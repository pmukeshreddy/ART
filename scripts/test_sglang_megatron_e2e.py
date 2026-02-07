#!/usr/bin/env python3
"""End-to-end test for SGLang + Megatron backend.

Tests the full RL cycle with:
- SGLang for inference (NOT vLLM, NOT unsloth)
- Megatron for distributed training (NOT unsloth)

Usage:
    source .venv/bin/activate
    python scripts/test_sglang_megatron_e2e.py --inference-only
    python scripts/test_sglang_megatron_e2e.py  # Full test with training

Requirements:
    - SGLang installed (pip install sglang[srt])
    - OR .venv-sglang-server with SGLang installed
    - For training: Megatron-Core with megatron.bridge
"""

import warnings
warnings.filterwarnings("ignore", message="resource_tracker:")

import os
import sys
import asyncio
import argparse
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_inference_only(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Quick test that only verifies SGLang inference works."""
    print("=" * 60)
    print("SGLang + Megatron Backend - Inference Only Test")
    print("=" * 60)
    
    # Step 1: Import
    print("\n[1/4] Importing modules...")
    try:
        import torch
        import art
        from art.megatron import SGLangMegatronBackend
        from openai import AsyncOpenAI
        print("  ✓ Imports OK")
        print(f"  CUDA devices: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Step 2: Initialize backend
    print("\n[2/4] Initializing SGLangMegatronBackend...")
    try:
        backend = SGLangMegatronBackend(
            sglang_config={
                "mem_fraction_static": 0.8,
                "server_timeout": 180.0,
            }
        )
        print("  ✓ Backend initialized")
    except Exception as e:
        print(f"  ✗ Backend init failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Register model and start server
    print("\n[3/4] Registering model and starting SGLang server...")
    try:
        model = art.TrainableModel(
            name="sglang-megatron-test",
            base_model=model_name,
            project="sglang-megatron-test",
        )
        await backend.register(model)
        print(f"  Model: {model.name}")
        print(f"  Base: {model.base_model}")
        
        base_url, api_key = await backend._prepare_backend_for_training(model, None)
        print(f"  Server URL: {base_url}")
        print("  ✓ Server started")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test inference
    print("\n[4/4] Testing inference...")
    try:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        model_inference_name = backend._model_inference_name(model)
        print(f"  Model name for inference: {model_inference_name}")
        
        for i in range(3):
            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=model_inference_name,
                messages=[{"role": "user", "content": f"Say the number {i+1}."}],
                max_tokens=10,
            )
            elapsed = time.perf_counter() - start
            print(f"  Request {i+1}: {response.choices[0].message.content!r} ({elapsed:.2f}s)")
        
        print("  ✓ Inference works!")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n" + "=" * 60)
    print("INFERENCE TEST PASSED!")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["pkill", "-9", "-f", "sglang"], capture_output=True)
    
    return True


async def test_e2e(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    skip_training: bool = False,
):
    """Run end-to-end test for SGLang + Megatron backend."""
    print("=" * 60)
    print("SGLang + Megatron Backend End-to-End Test")
    print("=" * 60)
    
    # Step 1: Import
    print("\n[1/6] Importing modules...")
    try:
        import torch
        import art
        from art.megatron import SGLangMegatronBackend
        from art.trajectories import Trajectory, TrajectoryGroup
        from openai import AsyncOpenAI
        print("  ✓ Imports OK")
        print(f"  CUDA devices: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Step 2: Initialize backend
    print("\n[2/6] Initializing SGLangMegatronBackend...")
    try:
        backend = SGLangMegatronBackend(
            sglang_config={
                "mem_fraction_static": 0.9,
                "server_timeout": 600.0,  # 10 min for large models
                # tensor_parallel_size: auto-detected based on GPU count
            }
        )
        print("  ✓ Backend initialized")
    except Exception as e:
        print(f"  ✗ Backend init failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Register model
    print("\n[3/6] Registering model...")
    try:
        model = art.TrainableModel(
            name="sglang-megatron-e2e-test",
            base_model=model_name,
            project="sglang-megatron-test",
        )
        await backend.register(model)
        print(f"  Model: {model.name}")
        print(f"  Base: {model.base_model}")
        print("  ✓ Model registered")
    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Start server and test inference
    print("\n[4/6] Starting SGLang server and testing inference...")
    try:
        base_url, api_key = await backend._prepare_backend_for_training(model, None)
        print(f"  Server URL: {base_url}")
        
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        model_inference_name = backend._model_inference_name(model)
        print(f"  Model name: {model_inference_name}")
        
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_inference_name,
            messages=[{"role": "user", "content": "Say 'test passed' in exactly two words."}],
            max_tokens=10,
        )
        elapsed = time.perf_counter() - start
        print(f"  Response: {response.choices[0].message.content}")
        print(f"  Latency: {elapsed:.2f}s")
        print("  ✓ Inference works")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Training
    if skip_training:
        print("\n[5/6] Skipping training (--skip-training flag set)...")
        print("  ✓ Training skipped")
    else:
        print("\n[5/6] Running Megatron training step...")
        try:
            # Create trajectories
            trajectories = []
            for i, (question, expected_reward) in enumerate([
                ("What is 2+2? Answer with just the number.", 1.0),
                ("What is 2+2? Answer with a wrong number.", 0.0),
            ]):
                response = await client.chat.completions.create(
                    model=model_inference_name,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=10,
                    logprobs=True,
                )
                choice = response.choices[0]
                
                traj = Trajectory(
                    messages_and_choices=[
                        {"role": "user", "content": question},
                        choice,
                    ],
                    reward=expected_reward,
                )
                trajectories.append(traj)
                print(f"  Trajectory {i+1}: '{choice.message.content}' -> reward={expected_reward}")
            
            trajectory_group = TrajectoryGroup(trajectories=trajectories)
            
            print("  Training with Megatron...")
            start = time.perf_counter()
            result = await backend.train(
                model,
                [trajectory_group],
                learning_rate=1e-5,
                verbose=True,
            )
            elapsed = time.perf_counter() - start
            print(f"  Step: {result.step}")
            print(f"  Metrics: {result.metrics}")
            print(f"  Training time: {elapsed:.2f}s")
            print("  ✓ Training complete")
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 6: Test inference after training
    print("\n[6/6] Testing inference after training...")
    try:
        model_inference_name = backend._model_inference_name(model)
        
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_inference_name,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
        )
        elapsed = time.perf_counter() - start
        print(f"  Response: {response.choices[0].message.content}")
        print(f"  Latency: {elapsed:.2f}s")
        print("  ✓ Post-training inference works")
    except Exception as e:
        print(f"  ✗ Post-training inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["pkill", "-9", "-f", "sglang"], capture_output=True)
    subprocess.run(["pkill", "-9", "megatron-service"], capture_output=True)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test SGLang + Megatron backend end-to-end"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to use for testing",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training step (inference only)",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run quick inference-only test",
    )
    
    args = parser.parse_args()
    
    if args.inference_only:
        success = asyncio.run(test_inference_only(args.model))
    else:
        success = asyncio.run(test_e2e(args.model, args.skip_training))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
