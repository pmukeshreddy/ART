"""
SGLang + Megatron backend — drop-in replacement for MegatronBackend.

This mirrors art.megatron.backend.MegatronBackend but plugs in
SGLangMegatronService instead of MegatronService.  All higher-level
methods (train, register, etc.) are inherited from LocalBackend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from art.local.backend import LocalBackend
from art.local.service import ModelService
from art.model import TrainableModel
from art.utils.output_dirs import get_model_dir

from .sglang_megatron_service import SGLangMegatronService

if TYPE_CHECKING:
    pass


class SGLangMegatronBackend(LocalBackend):
    """
    Backend that uses SGLang for inference and Megatron for training.

    Extends LocalBackend so that all training, logging, and checkpoint
    management is inherited unchanged — only the inference engine differs.
    """

    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        sglang_python: str = "python",
        port: int = 8200,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256,
    ) -> None:
        super().__init__(in_process=in_process, path=path)
        self._sglang_python = sglang_python
        self._port = port
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_num_seqs = max_num_seqs

    async def _get_service(self, model: TrainableModel) -> ModelService:
        """
        Create an SGLangMegatronService instead of the default service.

        The service is cached by model name, just like LocalBackend.
        """
        from art.dev.get_model_config import get_model_config

        if model.name not in self._services:
            output_dir = get_model_dir(model=model, art_path=self._path)
            config = get_model_config(
                base_model=model.base_model,
                output_dir=output_dir,
                config=model._internal_config,
            )

            # Determine tensor parallel size from config or default
            tp = config.get("engine_args", {}).get(
                "tensor_parallel_size", self._tensor_parallel_size
            )

            service = SGLangMegatronService(
                model_name=model.name,
                base_model=model.base_model,
                output_dir=output_dir,
                sglang_python=self._sglang_python,
                port=self._port,
                tensor_parallel_size=tp,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_num_seqs=self._max_num_seqs,
            )

            # Note: We do NOT move to child process (unlike MegatronBackend)
            # because SGLang already runs as a separate process.
            self._services[model.name] = service  # type: ignore[assignment]

        return self._services[model.name]

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config=None,
    ) -> tuple[str, str]:
        """
        Start the SGLang server and return (base_url, api_key).
        """
        service = await self._get_service(model)
        assert isinstance(service, SGLangMegatronService)

        host, port = await service.start_openai_server()
        base_url = f"http://{host}:{port}/v1"
        api_key = "sglang-benchmark"

        return base_url, api_key
