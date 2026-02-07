"""OpenAI-compatible server functionality for vLLM."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from typing import Any, AsyncIterator, Coroutine, cast

from openai import AsyncOpenAI
from uvicorn.config import LOGGING_CONFIG

from ..dev.openai_server import OpenAIServerConfig


def _sanitize_lora_configs(lora_paths: list[str] | None) -> None:
    """
    Remove unsupported fields from adapter_config.json files.
    
    vLLM 0.15.x doesn't support 'long_lora_max_len' but PEFT includes it.
    This function removes such fields to prevent vLLM errors during init_static_loras().
    """
    if not lora_paths:
        return
    
    unsupported_fields = {'long_lora_max_len'}
    
    for lora_path in lora_paths:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check if any unsupported fields exist
            fields_to_remove = unsupported_fields & set(config.keys())
            if not fields_to_remove:
                continue
            
            # Remove unsupported fields
            for field in fields_to_remove:
                del config[field]
            
            # Write back cleaned config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except (json.JSONDecodeError, IOError):
            # Skip if config can't be read/written
            pass

# Version-compatible imports for vLLM
def _get_vllm_version() -> tuple[int, int]:
    """Get vLLM version as (major, minor) tuple."""
    try:
        import vllm
        version = getattr(vllm, '__version__', '0.0.0')
        parts = version.split('.')[:2]
        return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (0, 0)
    except (ImportError, ValueError):
        return (0, 0)

_VLLM_VERSION = _get_vllm_version()

# Import EngineClient - location varies by version
try:
    if _VLLM_VERSION >= (0, 15):
        from vllm.engine.protocol import EngineClient
    else:
        from vllm.engine.protocol import EngineClient
except ImportError:
    # Fallback for different vLLM structures
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine as EngineClient
    except ImportError:
        EngineClient = Any  # type: ignore

# Import CLI args - location varies by version
try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
except ImportError:
    try:
        from vllm.entrypoints.openai.api_server import make_arg_parser
        validate_parsed_serve_args = lambda x: x  # No-op for older versions
    except ImportError:
        make_arg_parser = None
        validate_parsed_serve_args = lambda x: x

# Import logger utilities
try:
    from vllm.logger import _DATE_FORMAT, _FORMAT
except ImportError:
    _DATE_FORMAT = "%m-%d %H:%M:%S"
    _FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"

try:
    from vllm.logging_utils import NewLineFormatter
except ImportError:
    # Fallback formatter
    class NewLineFormatter(logging.Formatter):  # type: ignore
        pass

# Import argument parser
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    try:
        from vllm.utils.argument_parser import FlexibleArgumentParser
    except ImportError:
        import argparse
        FlexibleArgumentParser = argparse.ArgumentParser  # type: ignore

_openai_serving_models: Any | None = None


async def openai_server_task(
    engine: EngineClient,
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    """
    Starts an asyncio task that runs an OpenAI-compatible server.

    Args:
        engine: The vLLM engine client.
        config: The configuration for the OpenAI-compatible server.

    Returns:
        A running asyncio task for the OpenAI-compatible server. Cancel the task
        to stop the server.
    """
    # Import patches before importing api_server
    from .patches import (
        patch_listen_for_disconnect,
        patch_tool_parser_manager,
        subclass_chat_completion_request,
    )

    # We must subclass ChatCompletionRequest before importing api_server
    # or logprobs will not always be returned
    subclass_chat_completion_request()
    
    # Import api_server (required for all versions)
    from vllm.entrypoints.openai import api_server
    
    # Capture the OpenAIServingModels instance so dynamically added LoRAs
    # are reflected in the model list.
    serving_models_cls = None
    try:
        from vllm.entrypoints.openai.serving_models import OpenAIServingModels as _cls
        serving_models_cls = _cls
    except ImportError:
        pass
    # Fallback for vLLM v1 / 0.15+ where the module moved
    if serving_models_cls is None:
        try:
            from vllm.entrypoints.openai.api_server import OpenAIServingModels as _cls2  # type: ignore
            serving_models_cls = _cls2
        except (ImportError, AttributeError):
            pass

    if serving_models_cls is not None:
        if not getattr(serving_models_cls, "_art_openai_serving_models_patched", False):
            serving_models_cls._art_openai_serving_models_patched = True  # type: ignore
            original_init = serving_models_cls.__init__

            def _init(self, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                global _openai_serving_models
                _openai_serving_models = self

            serving_models_cls.__init__ = _init  # ty:ignore[invalid-assignment]

    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    set_vllm_log_file(config.get("log_file", "vllm.log"))

    # Patch engine.add_lora to ensure lora_tensors attribute exists
    # This is needed for compatibility with Unsloth
    add_lora = engine.add_lora

    async def _add_lora(lora_request) -> bool:
        # Ensure lora_tensors attribute exists on the request
        if not hasattr(lora_request, "lora_tensors"):
            # For msgspec.Struct, we need to create a new instance with the attribute
            from vllm.lora.request import LoRARequest
            import inspect
            
            # Get the LoRARequest signature to determine which parameters it accepts
            sig = inspect.signature(LoRARequest)
            valid_params = set(sig.parameters.keys())
            
            # Build kwargs based on what the LoRARequest accepts
            kwargs: dict[str, Any] = {
                "lora_name": lora_request.lora_name,
                "lora_int_id": lora_request.lora_int_id,
                "lora_path": lora_request.lora_path,
            }
            
            # Only add optional params if they exist in the signature
            if "long_lora_max_len" in valid_params:
                kwargs["long_lora_max_len"] = getattr(lora_request, "long_lora_max_len", None)
            if "base_model_name" in valid_params:
                kwargs["base_model_name"] = getattr(lora_request, "base_model_name", None)
            
            lora_request = LoRARequest(**kwargs)
        added = await add_lora(lora_request)
        if added and _openai_serving_models is not None:
            _openai_serving_models.lora_requests[lora_request.lora_name] = lora_request
        return added

    engine.add_lora = _add_lora  # ty:ignore[invalid-assignment]

    @asynccontextmanager
    async def build_async_engine_client(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[EngineClient]:
        yield engine

    api_server.build_async_engine_client = build_async_engine_client
    openai_server_task = asyncio.create_task(_openai_server_coroutine(config))
    server_args = config.get("server_args", {})
    client = AsyncOpenAI(
        api_key=server_args.get("api_key"),
        base_url=f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1",
    )

    async def test_client() -> None:
        while True:
            try:
                async for _ in client.models.list():
                    return
            except:  # noqa: E722
                await asyncio.sleep(0.1)

    test_client_task = asyncio.create_task(test_client())
    try:
        timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 30.0))
        done, _ = await asyncio.wait(
            [openai_server_task, test_client_task],
            timeout=timeout,
            return_when="FIRST_COMPLETED",
        )
        if not done:
            raise TimeoutError(
                f"Unable to reach OpenAI-compatible server within {timeout} seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable."
            )
        for task in done:
            task.result()

        return openai_server_task
    except Exception:
        openai_server_task.cancel()
        test_client_task.cancel()
        raise


def _openai_server_coroutine(
    config: OpenAIServerConfig,
) -> Coroutine[Any, Any, None]:
    from vllm.entrypoints.openai import api_server

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    engine_args = config.get("engine_args", {})
    server_args = config.get("server_args", {})
    
    # Sanitize LoRA adapter configs to remove unsupported fields (e.g., long_lora_max_len)
    # before vLLM tries to load them during init_static_loras()
    lora_paths = engine_args.get("lora_modules") or server_args.get("lora_modules")
    if isinstance(lora_paths, str):
        lora_paths = [lora_paths]
    _sanitize_lora_configs(lora_paths)
    
    args = [
        *[
            f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
            for args in [engine_args, server_args]
            for key, value in args.items()
            for item in (value if isinstance(value, list) else [value])
            if item is not None
        ],
    ]
    namespace = parser.parse_args(args)
    assert namespace is not None
    validate_parsed_serve_args(namespace)
    return api_server.run_server(
        namespace,
        log_config=get_uvicorn_logging_config(config.get("log_file", "vllm.log")),
    )


def get_uvicorn_logging_config(path: str) -> dict[str, Any]:
    """
    Returns a Uvicorn logging config that writes to the given path.
    """
    return {
        **LOGGING_CONFIG,
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
            "access": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
        },
    }


def set_vllm_log_file(path: str) -> None:
    """
    Sets the vLLM log file to the given path.
    """

    # Create directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get the vLLM logger
    vllm_logger = logging.getLogger("vllm")

    # Remove existing handlers
    for handler in vllm_logger.handlers[:]:
        vllm_logger.removeHandler(handler)

    # Create a file handler
    file_handler = logging.FileHandler(path)

    # Use vLLM's NewLineFormatter which adds the fileinfo field
    formatter = NewLineFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    vllm_logger.addHandler(file_handler)

    # Set log level to filter out DEBUG messages
    vllm_logger.setLevel(logging.INFO)
