from abc import ABC, abstractmethod

class BaseImageClient(ABC):
    """
    Abstract interface for exterior image-editing models.
    Supports easy switching between providers and models (e.g. gemini-2.5-flash-image vs imagen-3.0 vs mock).
    """
    is_mock: bool = False

    @abstractmethod
    async def generate_scheme_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> bytes:
        """
        Takes the input building photo bytes and prompt, returns repainted image bytes.
        """
        pass
