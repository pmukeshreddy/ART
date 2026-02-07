# isort: off
import os

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
# NOTE: Do NOT set expandable_segments:True here. It prevents the CUDA
# allocator from returning memory to the driver when this process exits,
# which starves the SGLang process on shared GPUs. The default allocator
# behavior (release segments on empty_cache/process exit) is correct for
# our use case where memory must flow between training and inference processes.
# Support both A100 (8.0) and H100 (9.0) — hardcoding "9.0" breaks A100
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;9.0"
# isort: on

import json
import shutil
import sys
import time
from typing import Any, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
import torch

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.megatron.lora import apply_lora_adapters
# offload.py no longer used — process exits after each job instead of
# offloading to CPU and looping. All GPU memory is freed on process exit.
from art.megatron.provider import get_provider
from art.preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)

provider = get_provider(
    os.environ.get("MODEL_IDENTIFIER", "Qwen/Qwen3-30B-A3B-Instruct-2507")
)


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


provider.register_pre_wrap_hook(lambda x: freeze_model(x) or x)

model = provider.provide_distributed_model(
    ddp_config=DistributedDataParallelConfig(),
    data_parallel_random_init=False,
)

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

for module in model:
    while not isinstance(module, GPTModel) and hasattr(module, "module"):
        module = module.module
    if isinstance(module, GPTModel):
        _preprocess = module._preprocess

        def _preprocess_hook(*args, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore
            table = preproc_output[1]  # [S,B,1,D] type: ignore
            D = table.size(-1)  # type: ignore
            table_flat = table.view(table.size(0), D)  # type: ignore
            # position_ids: [B, S]
            position_ids = kwargs["position_ids"]
            B, S = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))  # [B*S, D]
            gathered = gathered.view(B, S, D).permute(1, 0, 2).contiguous()  # [S, B, D]
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = _preprocess_hook  # type: ignore[attr-defined]


apply_lora_adapters(model, provider)

optimizer = get_megatron_optimizer(
    config=OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
    ),
    model_chunks=model,  # type: ignore
)

if rank == 0:
    # Print the number of parameters in the optimizer, nicely formatted
    num_params = sum(
        p.numel()
        for group in optimizer.param_groups
        if not group["is_decoupled_lr"]
        for p in group["params"]
    )
    print(f"Number of parameters in optimizer: {num_params:,}")
    total_params = sum(p.numel() for m in model for p in m.parameters())
    percent = (num_params / total_params) * 100 if total_params > 0 else 0
    print(f"Optimizer parameters as percent of total: {percent:0.2f}%")


class TrainingJob(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig
    log_file_path: str = "/tmp/megatron_training_log.jsonl"


def print0(*values: Any) -> None:
    if rank == 0:
        print(*values)



def calculate_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> torch.Tensor:
    causal_mask = (
        torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )
        .unsqueeze(0)
        .expand(batch_size, seq_len, seq_len)
    )
    group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    parent_mask = parent_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    mask = causal_mask & (group_mask | parent_mask)
    return mask


# ═══════════════════════════════════════════════════════════════════
# SINGLE-JOB EXECUTION (veRL-style)
#
# Previous design: `while True` loop kept this process alive across
# ALL epochs. Each cycle accumulated:
#   - CUDA context overhead (~500MB per GPU, never freed)
#   - PyTorch allocator fragmentation (~500-800MB per cycle)
#   - Pinned buffer leaks from id(param) changes
#
# New design: Process initializes model, runs ONE training job, saves,
# and EXITS. The orchestrator (sglang_service.py) spawns a fresh
# process for each epoch. This costs ~10-15s startup per epoch but:
#   - CUDA contexts are fully freed when process exits
#   - Zero fragmentation accumulation between epochs
#   - No memory competition with SGLang process on shared GPUs
#   - Matches veRL's pattern: fresh process per training step
# ═══════════════════════════════════════════════════════════════════

# Model is already on GPU from initialization above — no offload/reload needed.
# Wait for a job file to appear.
jobs_dir = "/tmp/megatron_training_jobs"
os.makedirs(jobs_dir, exist_ok=True)
print0(f"[train.py] Waiting for training job in {jobs_dir}...")

job_path = None
max_wait_s = 600  # 10 min timeout
_wait_start = time.perf_counter()
while True:
    torch.distributed.barrier()
    job_names = sorted(
        job_name for job_name in os.listdir(jobs_dir) if job_name.endswith(".json")
    )
    if job_names:
        job_path = os.path.join(jobs_dir, job_names[0])
        break
    if time.perf_counter() - _wait_start > max_wait_s:
        print0(f"[train.py] No job after {max_wait_s}s, exiting.")
        torch.distributed.destroy_process_group()
        sys.exit(0)
    time.sleep(1)

wake_lock_path = "/tmp/megatron_vllm_waking"
while os.path.exists(wake_lock_path):
    time.sleep(0.2)

# Load job
with open(job_path, "rb") as f:
    job = TrainingJob.model_validate_json(f.read())
config = job.config
experimental_config = job.experimental_config
print0("Loaded job from", job_path)
print0("Job:", job)

# Load LoRA adapter weights
adapter_model_path = f"{job.lora_path}/adapter_model.safetensors"
if os.path.exists(adapter_model_path):
    print0("Loading adapter model from", adapter_model_path)
    adapter_model = load_file(adapter_model_path)
    with torch.no_grad():
        for chunk in model:
            for module in chunk.modules():
                if hasattr(module, "load_lora"):
                    module.load_lora(adapter_model)  # type: ignore
else:
    print0("No adapter model found at", adapter_model_path)
    adapter_model = {}
    with torch.no_grad():
        for chunk in model:
            for module in chunk.modules():
                if hasattr(module, "reset_lora_parameters"):
                    module.reset_lora_parameters()  # type: ignore

# Load optimizer state
optimizer_shard_path = os.path.join(
    job.optimizer_state_path, f"{rank + 1:02d}-of-{world_size:02d}.pt"
)
if os.path.exists(optimizer_shard_path):
    print(
        "Loading optimizer state from",
        optimizer_shard_path,
    )
    optimizer.load_state_dict(torch.load(optimizer_shard_path))
else:
    print(
        "No optimizer state found at",
        optimizer_shard_path,
        "— resetting optimizer for new run",
    )
    optimizer.optimizer.state.clear()
    optimizer.reload_model_params()

# Load training data
print0("Loading packed tensors from", job.disk_packed_tensors["dir"])
packed_tensors = packed_tensors_from_dir(**job.disk_packed_tensors)
num_sequences = job.disk_packed_tensors["num_sequences"]
dp_rank = ps.get_data_parallel_rank()
dp_world_size = ps.get_data_parallel_world_size()
tp_rank = ps.get_tensor_model_parallel_rank()
tp_world_size = ps.get_tensor_model_parallel_world_size()
print(f"[Rank {rank}] DP rank={dp_rank}/{dp_world_size}, TP rank={tp_rank}/{tp_world_size}, num_sequences={num_sequences}")
indices = list(
    range(
        dp_rank,
        num_sequences,
        dp_world_size,
    )
)
print(f"[Rank {rank}] Processing indices: {indices}")
# pad indices
if num_sequences % dp_world_size <= dp_rank > 0:
    indices.append(
        (list(range(num_sequences)) * (dp_world_size // num_sequences + 1))[dp_rank]
    )

# ── Training loop (single epoch) ──
print(f"[Rank {rank}] Starting training loop over {len(indices)} sequences")
for seq_idx, index in enumerate(indices):
    # Release fragmented reserved memory before each sequence to avoid OOM
    # on MoE models where grouped_gemm allocations cause fragmentation.
    torch.cuda.empty_cache()
    print(f"[Rank {rank}] Processing sequence {seq_idx+1}/{len(indices)}, index={index}")
    inputs = PackedTensors(  # type: ignore
        **{
            key: value[index : index + 1]
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )
    ref_logprobs = None
    device = next(model[0].parameters()).device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore
    attention_mask = ~calculate_mask(
        batch_size=inputs["tokens"].shape[0],
        seq_len=inputs["tokens"].shape[1],
        device=device,
        group_ids=inputs["group_ids"],
        parent_ids=inputs["parent_ids"],
    ).unsqueeze(1)  # add head dimension [B, H=1, S, S]
    attention_bias = torch.where(
        attention_mask,
        torch.tensor(
            float("-inf"), dtype=next(model[0].parameters()).dtype, device=device
        ),
        torch.tensor(0.0, dtype=next(model[0].parameters()).dtype, device=device),
    )
    print(f"[Rank {rank}] Running forward pass for index={index}...")
    new_logprobs: torch.Tensor = -model[0](
        input_ids=inputs["tokens"],
        position_ids=inputs["input_pos"],
        attention_mask=attention_mask,
        labels=shift_tensor(inputs["tokens"], 0),
        extra_block_kwargs={"attention_bias": attention_bias},
    )
    # Free mask tensors immediately — they're large ([B,1,S,S]) and no longer needed
    del attention_mask, attention_bias
    print(f"[Rank {rank}] Forward pass complete, computing loss...")
    loss = loss_fn(
        inputs,  # type: ignore
        new_logprobs,
        ref_logprobs,
        None,
        experimental_config,
    )
    del new_logprobs  # free logprobs before backward (recomputed via checkpointing)
    probs_corr = loss.probs_corr.item()
    print0("Correlation between old and new probabilities:", probs_corr)
    loss = loss.mean_policy_loss + config.beta * loss.mean_kl
    loss.backward()
    # Reduce LoRA grads
    start = time.perf_counter()
    num_grads = 0
    for chunk in model:
        for param in chunk.parameters():
            if param.grad is None:
                continue
            torch.distributed.all_reduce(
                param.grad,
                op=torch.distributed.ReduceOp.AVG,
                group=ps.get_data_parallel_group(),
            )
            num_grads += 1
    print0(
        f"Reduced {num_grads} LoRA grads in {(time.perf_counter() - start) * 1e3:.1f} ms"
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.learning_rate
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    optimizer.zero_grad()

    # Free the autograd graph immediately — with MoE models the intermediate
    # activations from 32+ experts per layer are huge and must be released
    # before the next sequence's forward pass.
    loss_val = loss.detach().clone()
    del loss, inputs
    torch.cuda.empty_cache()

    # Mean reduce loss across all ranks for logging
    torch.distributed.all_reduce(loss_val, op=torch.distributed.ReduceOp.AVG)

    if rank == 0:
        with open(job.log_file_path, "a") as log_file:
            log_msg = json.dumps(
                {
                    "loss": loss_val.item(),
                    "grad_norm": grad_norm,
                    "probs_corr": probs_corr,
                }
            )
            print("Logging", log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()

# ── Save checkpoint ──
sharded_state_dict = {}
for chunk in model:
    for module in chunk.modules():
        if hasattr(module, "sharded_lora_state_dict"):
            module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                module.sharded_lora_state_dict()  # type: ignore
            )
            for key, value in module_sharded_lora_state_dict.items():
                target_dtype = (
                    adapter_model[key].dtype
                    if key in adapter_model
                    else value.dtype
                )
                sharded_state_dict[key] = value.to(target_dtype)
shard_path = os.path.join(
    job.lora_path,
    f"adapter_model-{rank + 1:02d}-of-{world_size:02d}.safetensors",
)
print(f"[Rank {rank}] Saving adapter shard to {shard_path}")
save_file(sharded_state_dict, shard_path)
print(f"[Rank {rank}] Saving optimizer shard to {optimizer_shard_path}")
os.makedirs(job.optimizer_state_path, exist_ok=True)
torch.save(optimizer.state_dict(), optimizer_shard_path)

# ── Signal completion and EXIT ──
# No offload_to_cpu — process is about to exit, which frees ALL GPU memory
# (CUDA contexts, allocated tensors, reserved blocks, fragmented segments).
# This is the key architectural change: a fresh process next epoch starts
# with zero accumulated fragmentation.
print(f"[Rank {rank}] Waiting at final barrier...")
torch.distributed.barrier()
print(f"[Rank {rank}] Passed final barrier")
if rank == 0:
    os.remove(job_path)
    with open(job.log_file_path, "a") as log_file:
        log_file.write("all done\n")
        log_file.flush()
        os.fsync(log_file.fileno())  # Force write to disk
    print(f"[Rank 0] Wrote 'all done' to {job.log_file_path}")
    shutil.rmtree(job.disk_packed_tensors["dir"])

# Clean shutdown — destroy process group so torchrun doesn't report errors
print(f"[Rank {rank}] Training complete, exiting process.")
torch.distributed.destroy_process_group()
