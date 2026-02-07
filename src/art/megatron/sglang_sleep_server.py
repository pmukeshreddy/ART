#!/usr/bin/env python3
"""
ART SGLang Sleep Server — drop-in replacement for `python -m sglang.launch_server`

Adds /art/sleep and /art/wake endpoints for GPU memory offloading during training.

How it works:
  This script starts a lightweight control HTTP server (on sglang_port + 100),
  then launches SGLang normally. The control server runs in the same process
  as SGLang, so it can directly access the model weights via gc.get_objects().

  /art/sleep:
    - Finds the LLM model in this process (largest nn.Module)
    - Moves all parameters + buffers from GPU → CPU RAM
    - Calls torch.cuda.empty_cache() to free GPU memory
    - GPU drops from ~76GB → ~50GB (model weights freed, KV cache pool stays)

  /art/wake:
    - Reloads parameters + buffers from CPU → GPU
    - Model is ready for inference again

  If the model is in a child process (SGLang multi-process mode), the sleep
  endpoint returns {"status": "error"} and the caller falls back to SIGSTOP-only.

Usage:
  python -m art.megatron.sglang_sleep_server \\
      --model-path Qwen/Qwen3-30B-A3B --port 8000 [other sglang args]

  # Control server is automatically on port 8100 (or --art-control-port N)
  curl -X POST http://127.0.0.1:8100/art/sleep
  curl -X POST http://127.0.0.1:8100/art/wake
  curl http://127.0.0.1:8100/art/status
"""

import gc
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch


# ═══════════════════════════════════════════════════════════════════
# Sleep/Wake Engine — runs in SGLang's process, accesses GPU directly
# ═══════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_sleeping = False
_cpu_stash: dict[str, torch.Tensor] = {}
_model_ref: list = [None]  # mutable container for model reference


def _find_model() -> torch.nn.Module | None:
    """Find the LLM model in this process by scanning gc objects.

    Returns the largest nn.Module (by parameter count), which should
    be the LLM. Caches the result for subsequent calls.
    """
    if _model_ref[0] is not None:
        # Verify model is still alive
        try:
            next(iter(_model_ref[0].parameters()))
            return _model_ref[0]
        except (StopIteration, RuntimeError):
            _model_ref[0] = None

    best, best_size = None, 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                sz = sum(p.nelement() for p in obj.parameters())
                if sz > best_size:
                    best, best_size = obj, sz
        except Exception:
            continue

    if best is not None:
        _model_ref[0] = best
        print(
            f"[ART Sleep] Found model: {type(best).__name__} "
            f"({best_size / 1e9:.1f}B params, "
            f"{sum(p.nelement() * p.element_size() for p in best.parameters()) / 1e9:.1f}GB)"
        )

    return best


def do_sleep() -> dict:
    """Offload model weights (parameters + buffers) from GPU to CPU.

    The KV cache pool stays on GPU — it's pre-allocated by SGLang's memory
    manager and can't be safely freed without breaking the pool invariants.
    Model weights typically account for ~25GB on a 30B model, so freeing
    them reduces GPU 0 from ~76GB to ~50GB.

    Returns dict with offload statistics.
    """
    global _sleeping

    with _lock:
        if _sleeping:
            return {"status": "already_sleeping"}

        t0 = time.perf_counter()
        model = _find_model()

        if model is None:
            return {
                "status": "error",
                "error": "model not found in this process (may be in child process)",
            }

        gpu_freed = 0

        # Offload all parameters to CPU
        for name, param in model.named_parameters():
            if param.device.type == "cuda":
                _cpu_stash[f"p:{name}"] = param.data.cpu()
                gpu_freed += param.data.nelement() * param.data.element_size()
                param.data = torch.empty(0, dtype=param.dtype, device="cpu")

        # Offload buffers (batchnorm stats, etc.)
        for name, buf in model.named_buffers():
            if buf.device.type == "cuda" and buf.nelement() > 0:
                _cpu_stash[f"b:{name}"] = buf.data.cpu()
                gpu_freed += buf.nelement() * buf.element_size()
                buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")

        # Free CUDA memory
        gc.collect()
        torch.cuda.empty_cache()

        _sleeping = True
        elapsed = time.perf_counter() - t0

        result = {
            "status": "sleeping",
            "gpu_freed_gb": round(gpu_freed / 1e9, 2),
            "params_offloaded": sum(1 for k in _cpu_stash if k.startswith("p:")),
            "buffers_offloaded": sum(1 for k in _cpu_stash if k.startswith("b:")),
            "offload_time_s": round(elapsed, 3),
        }
        print(f"[ART Sleep] ✓ Offloaded {result['gpu_freed_gb']}GB to CPU in {elapsed:.2f}s")
        return result


def do_wake() -> dict:
    """Reload model weights from CPU back to GPU.

    After reload, CUDA graphs may need to be recaptured (SGLang does this
    automatically on the next inference call with each batch size).

    Returns dict with reload statistics.
    """
    global _sleeping

    with _lock:
        if not _sleeping:
            return {"status": "already_awake"}

        t0 = time.perf_counter()
        model = _find_model()
        gpu_reloaded = 0

        if model is not None:
            # Reload parameters
            for name, param in model.named_parameters():
                key = f"p:{name}"
                if key in _cpu_stash:
                    param.data = _cpu_stash.pop(key).cuda()
                    gpu_reloaded += param.data.nelement() * param.data.element_size()

            # Reload buffers
            for name, buf in model.named_buffers():
                key = f"b:{name}"
                if key in _cpu_stash:
                    buf.data = _cpu_stash.pop(key).cuda()
                    gpu_reloaded += buf.data.nelement() * buf.data.element_size()

        _cpu_stash.clear()
        _sleeping = False
        elapsed = time.perf_counter() - t0

        result = {
            "status": "awake",
            "gpu_reloaded_gb": round(gpu_reloaded / 1e9, 2),
            "reload_time_s": round(elapsed, 3),
        }
        print(f"[ART Sleep] ✓ Reloaded {result['gpu_reloaded_gb']}GB to GPU in {elapsed:.2f}s")
        return result


# ═══════════════════════════════════════════════════════════════════
# Control HTTP Server — lightweight, runs on sglang_port + 100
# ═══════════════════════════════════════════════════════════════════


class _ControlHandler(BaseHTTPRequestHandler):
    """Handles /art/sleep, /art/wake, /art/status requests."""

    def do_POST(self):
        if self.path == "/art/sleep":
            self._json(200, do_sleep())
        elif self.path == "/art/wake":
            self._json(200, do_wake())
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/art/status":
            self._json(
                200,
                {
                    "sleeping": _sleeping,
                    "model_found": _model_ref[0] is not None,
                    "stash_size_gb": round(
                        sum(t.nelement() * t.element_size() for t in _cpu_stash.values()) / 1e9, 2
                    ),
                },
            )
        elif self.path == "/art/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress default access logs


def _start_control_server(port: int):
    """Start the control HTTP server in a daemon thread."""
    try:
        server = HTTPServer(("127.0.0.1", port), _ControlHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[ART Sleep] Control server on 127.0.0.1:{port}")
        return server
    except OSError as e:
        print(f"[ART Sleep] ⚠️ Could not start control server on port {port}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Main — parse args, start control server, launch SGLang
# ═══════════════════════════════════════════════════════════════════


def main():
    # Parse ART-specific args (strip before SGLang sees them)
    control_port = None
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--art-control-port":
            control_port = int(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = filtered_argv

    # Auto-detect control port from SGLang's --port arg
    if control_port is None:
        for j, arg in enumerate(filtered_argv):
            if arg == "--port" and j + 1 < len(filtered_argv):
                control_port = int(filtered_argv[j + 1]) + 100
                break
        if control_port is None:
            control_port = 8100

    # Start control server BEFORE SGLang loads (available during model load)
    _start_control_server(control_port)

    # Launch SGLang (this blocks until server exits)
    print(f"[ART Sleep] Launching SGLang server...")
    from sglang.launch_server import main as sglang_main

    sglang_main()


if __name__ == "__main__":
    main()
