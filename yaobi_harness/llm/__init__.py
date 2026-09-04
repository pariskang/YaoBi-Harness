"""Provider-neutral LLM access for the Yaobi harness."""

from .base import LLMClient, LLMError, LLMResponse, NullLLMClient, ToolCall, ToolSpec, extract_json
from .factory import build_client, describe_client
from .providers import AzureOpenAIClient, LiteLLMClient, MiniMaxClient, OpenAICompatibleClient, PoeClient

__all__ = [
    "LLMClient", "LLMError", "LLMResponse", "NullLLMClient", "ToolCall", "ToolSpec", "extract_json",
    "build_client", "describe_client",
    "AzureOpenAIClient", "LiteLLMClient", "MiniMaxClient", "OpenAICompatibleClient", "PoeClient",
]
