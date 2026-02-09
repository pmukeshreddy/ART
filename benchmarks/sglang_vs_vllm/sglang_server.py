"""
SGLang server lifecycle management.

Handles launching, health-checking, and stopping the SGLang inference
server as an external process.

Key fixes based on SGLang docs & GitHub issues:
- --lora-paths format must be name=path (not bare path)
- --mem-fraction-static (not --gpu-memory-utilization)
- --max-running-requests (not --max-num-seqs)
- --enable-p2p-check prevents hangs on multi-GPU
- --disable-cuda-graph avoids cuda_fp8.h compilation failures
- --served-model-name sets the model name for OpenAI API requests
- Health endpoint is /health (GET, returns 200 when ready)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SGLangServerConfig:
    """Configuration for launching an SGLang server."""

    model_path: str
    served_model_name: str = ""  # defaults to model_path if empty
    port: int = 8200
    host: str = "0.0.0.0"
    tensor_parallel_size: int = 2
    mem_fraction_static: float = 0.85
    max_running_requests: int = 256
    dtype: str = "auto"
    trust_remote_code: bool = True
    python_executable: str = "python"
    log_file: str | None = None

    # LoRA — format must be "name=path"
    lora_paths: list[str] = field(default_factory=list)

    # Performance
    chunked_prefill_size: int = 32768
    disable_cuda_graph: bool = False  # set True if cuda_fp8.h missing
    enable_p2p_check: bool = True  # prevents multi-GPU hangs

    # Additional raw args
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.served_model_name:
            self.served_model_name = self.model_path


class SGLangServerError(Exception):
    """Raised when the SGLang server encounters an error."""


class SGLangServer:
    """
    Manages the lifecycle of an SGLang inference server process.
    """

    def __init__(self, config: SGLangServerConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._startup_time: float = 0.0
        self._shutdown_time: float = 0.0
        self._log_fh: Any = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def last_startup_time(self) -> float:
        return self._startup_time

    @property
    def last_shutdown_time(self) -> float:
        return self._shutdown_time

    # ------------------------------------------------------------------
    # Build launch command
    # ------------------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        c = self.config
        cmd = [
            c.python_executable, "-m", "sglang.launch_server",
            "--model-path", c.model_path,
            "--served-model-name", c.served_model_name,
            "--port", str(c.port),
            "--host", c.host,
            "--tp", str(c.tensor_parallel_size),
            "--mem-fraction-static", str(c.mem_fraction_static),
            "--max-running-requests", str(c.max_running_requests),
            "--dtype", c.dtype,
            "--chunked-prefill-size", str(c.chunked_prefill_size),
        ]
        if c.trust_remote_code:
            cmd.append("--trust-remote-code")
        if c.disable_cuda_graph:
            cmd.append("--disable-cuda-graph")
        if c.enable_p2p_check:
            cmd.append("--enable-p2p-check")
        # LoRA paths: each must be "name=path"
        for lp in c.lora_paths:
            cmd.extend(["--lora-paths", lp])
        cmd.extend(c.extra_args)
        return cmd

    # ------------------------------------------------------------------
    # Start / stop / health
    # ------------------------------------------------------------------

    async def start(self, timeout: int = 600) -> float:
        """Start server and block until healthy.  Returns startup seconds."""
        if self.is_running:
            logger.warning("Server already running — stopping first")
            await self.stop()

        await self._kill_port(self.config.port)

        cmd = self._build_cmd()
        logger.info("Starting SGLang: %s", " ".join(cmd))

        # Log file
        out_target: Any = subprocess.DEVNULL
        if self.config.log_file:
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
            self._log_fh = open(self.config.log_file, "a")
            out_target = self._log_fh

        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # Required for TP>1 — prevents false memory imbalance errors
        # Ref: https://verl.readthedocs.io/en/v0.5.x/workers/sglang_worker.html
        env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "True"

        t0 = time.perf_counter()
        self._process = subprocess.Popen(
            cmd,
            stdout=out_target,
            stderr=subprocess.STDOUT if self.config.log_file else subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )

        try:
            await self._wait_healthy(timeout)
        except Exception:
            await self.stop(timeout=10)
            raise

        self._startup_time = time.perf_counter() - t0
        logger.info("SGLang ready in %.2fs (pid=%s)", self._startup_time, self._process.pid)
        return self._startup_time

    async def stop(self, timeout: int = 60) -> float:
        """Stop server.  Returns shutdown seconds."""
        if not self.is_running:
            self._shutdown_time = 0.0
            return 0.0

        pid = self._process.pid  # type: ignore[union-attr]
        t0 = time.perf_counter()
        logger.info("Stopping SGLang (pid=%s)", pid)

        # SIGTERM → wait → SIGKILL
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                break
            await asyncio.sleep(0.5)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if self._process is not None:
                self._process.wait(timeout=10)

        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self._process = None

        await self._kill_port(self.config.port)
        await asyncio.sleep(1.0)

        self._shutdown_time = time.perf_counter() - t0
        logger.info("SGLang stopped in %.2fs", self._shutdown_time)
        return self._shutdown_time

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    async def flush_cache(self) -> bool:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/flush_cache",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _wait_healthy(self, timeout: int) -> None:
        deadline = time.perf_counter() + timeout
        interval = 2.0
        last_err: Exception | None = None

        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise SGLangServerError(
                    f"SGLang exited with code {self._process.returncode} during startup. "
                    f"Check: {self.config.log_file}"
                )
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{self.base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        if r.status == 200:
                            # Quick smoke test
                            await self._smoke_test()
                            return
            except Exception as e:
                last_err = e
            await asyncio.sleep(interval)
            interval = min(interval * 1.2, 10.0)

        raise SGLangServerError(
            f"SGLang not ready after {timeout}s. Last error: {last_err}"
        )

    async def _smoke_test(self) -> None:
        """One tiny request to confirm model is loaded."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.openai_base_url}/chat/completions",
                    json={
                        "model": self.config.served_model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        raise SGLangServerError(f"Smoke test: {r.status} {body[:200]}")
        except aiohttp.ClientError as e:
            raise SGLangServerError(f"Smoke test failed: {e}")

    @staticmethod
    async def _kill_port(port: int) -> None:
        try:
            p = await asyncio.create_subprocess_shell(
                f"lsof -ti:{port} | xargs -r kill -9 2>/dev/null || true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
        except Exception:
            pass
