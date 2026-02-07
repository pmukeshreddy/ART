"""Monkey patches and modifications for vLLM."""

from typing import Any

# Cache for protocol module
_vllm_protocol_module = None
_vllm_version = None


def _get_vllm_version() -> str:
    """Get the installed vLLM version."""
    global _vllm_version
    if _vllm_version is None:
        try:
            import vllm
            _vllm_version = getattr(vllm, '__version__', '0.0.0')
        except ImportError:
            _vllm_version = '0.0.0'
    return _vllm_version


def _get_vllm_protocol_module():
    """Get the vLLM protocol module, handling different vLLM versions."""
    global _vllm_protocol_module
    if _vllm_protocol_module is not None:
        return _vllm_protocol_module
    
    version = _get_vllm_version()
    major_minor = tuple(int(x) for x in version.split('.')[:2]) if version else (0, 0)
    
    # vLLM 0.15.x+ has different structure
    if major_minor >= (0, 15):
        try:
            # In vLLM 0.15.x, try importing from the openai module directly
            from vllm.entrypoints.openai import serving_chat
            _vllm_protocol_module = serving_chat
            return _vllm_protocol_module
        except ImportError:
            pass
        
        try:
            # Alternative for 0.15.x
            import vllm.entrypoints.openai.api_server as api_server
            _vllm_protocol_module = api_server
            return _vllm_protocol_module
        except ImportError:
            pass
    
    # vLLM 0.6.x - 0.14.x
    if major_minor >= (0, 6):
        try:
            import vllm.entrypoints.openai.protocol as protocol
            _vllm_protocol_module = protocol
            return _vllm_protocol_module
        except ImportError:
            pass
    
    # vLLM 0.4.x - 0.5.x (legacy)
    try:
        import vllm.entrypoints.openai.protocol as protocol
        _vllm_protocol_module = protocol
        return _vllm_protocol_module
    except ImportError:
        pass
    
    # Return None instead of raising - let callers handle gracefully
    return None


def subclass_chat_completion_request() -> None:
    """
    Subclass ChatCompletionRequest so that logprobs are always returned.
    
    For vLLM 0.15.x+, this may not be needed as the API handles logprobs differently.
    """
    version = _get_vllm_version()
    major_minor = tuple(int(x) for x in version.split('.')[:2]) if version else (0, 0)
    
    # vLLM 0.15.x+ handles logprobs via request parameters, skip patching
    if major_minor >= (0, 15):
        # In vLLM 0.15.x, logprobs are handled via the API request parameters
        # No need to patch ChatCompletionRequest
        return
    
    protocol = _get_vllm_protocol_module()
    if protocol is None:
        return
    
    if not hasattr(protocol, 'ChatCompletionRequest'):
        return

    class ChatCompletionRequest(protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0

    protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]


def patch_listen_for_disconnect() -> None:
    """Patch listen_for_disconnect to handle edge cases."""
    version = _get_vllm_version()
    major_minor = tuple(int(x) for x in version.split('.')[:2]) if version else (0, 0)
    
    # vLLM 0.15.x+ has different disconnect handling
    if major_minor >= (0, 15):
        return
    
    async def patched_listen_for_disconnect(request):
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    break
        except UnboundLocalError:
            pass

    # Replace the original function
    try:
        import vllm.entrypoints.utils
        vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect  # ty:ignore[invalid-assignment]
    except (ModuleNotFoundError, AttributeError):
        # vLLM version doesn't have this module, skip patching
        pass


def patch_tool_parser_manager() -> None:
    """
    Patch ToolParserManager to support streaming tool call logprobs.
    """
    version = _get_vllm_version()
    major_minor = tuple(int(x) for x in version.split('.')[:2]) if version else (0, 0)
    
    # vLLM 0.15.x+ handles this differently
    if major_minor >= (0, 15):
        return
    
    try:
        # Try to get DeltaMessage from protocol module
        protocol = _get_vllm_protocol_module()
        DeltaMessage = None
        
        if protocol is not None:
            DeltaMessage = getattr(protocol, 'DeltaMessage', None)
        
        if DeltaMessage is None:
            # Try alternative import
            try:
                from vllm.entrypoints.openai.protocol import DeltaMessage
            except ImportError:
                return  # Can't patch without DeltaMessage
        
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

        get_tool_parser = ToolParserManager.get_tool_parser

        def patched_get_tool_parser(name: str) -> type:
            tool_parser_class = get_tool_parser(name)
            original = tool_parser_class.extract_tool_calls_streaming

            def patch(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                return original(*args, **kwargs) or DeltaMessage()

            tool_parser_class.extract_tool_calls_streaming = patch  # ty:ignore[invalid-assignment]
            return tool_parser_class

        ToolParserManager.get_tool_parser = patched_get_tool_parser  # ty:ignore[invalid-assignment]
    except (ModuleNotFoundError, AttributeError, ImportError):
        # vLLM version doesn't support this patching, skip
        pass
