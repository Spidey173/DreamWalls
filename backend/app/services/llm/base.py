from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMClient(ABC):
    """
    Abstract interface for multi-provider LLM integration
    (Gemini, Anthropic, Groq).
    """

    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        """
        Send an image along with a vision prompt to the LLM and return raw response text.
        """
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a text prompt to the LLM and return raw response text.
        """
        pass
