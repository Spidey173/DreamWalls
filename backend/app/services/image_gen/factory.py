import logging
from typing import Optional
from app.core.config import settings
from app.services.image_gen.base import BaseImageClient
from app.services.image_gen.gemini_client import GeminiImageClient
from app.services.image_gen.mock_client import MockImageClient

logger = logging.getLogger(__name__)

def get_image_client(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    allow_mock: Optional[bool] = None
) -> BaseImageClient:
    """
    Factory to instantiate image generation client.
    Allows easy switching between models via IMAGE_MODEL env var.
    In production mode (IMAGE_ALLOW_MOCK=False by default), raises ValueError
    if IMAGE_API_KEY is missing so callers can use precise mask recoloring.
    """
    target_model = model if model is not None else (settings.IMAGE_MODEL or "gemini-2.5-flash-image")
    target_key = api_key if api_key is not None else (settings.IMAGE_API_KEY or settings.GEMINI_API_KEY)
    mock_allowed = allow_mock if allow_mock is not None else settings.IMAGE_ALLOW_MOCK

    if target_model.lower() == "mock":
        if not mock_allowed:
            raise ValueError(
                "IMAGE_MODEL is set to 'mock' but IMAGE_ALLOW_MOCK is false. "
                "Enable IMAGE_ALLOW_MOCK=true in .env to use the mock client."
            )
        logger.info("Using MockImageClient as requested by IMAGE_MODEL='mock'")
        return MockImageClient(model=target_model)

    if not target_key:
        if not mock_allowed:
            raise ValueError(
                "IMAGE_API_KEY is not configured and IMAGE_ALLOW_MOCK is false. "
                "Set IMAGE_API_KEY in .env to use the real image model, or enable IMAGE_ALLOW_MOCK=true for demo testing."
            )
        logger.warning(
            f"IMAGE_API_KEY is not configured, but IMAGE_ALLOW_MOCK=True is enabled. Using MockImageClient for '{target_model}'."
        )
        return MockImageClient(model=target_model)

    logger.info(f"Instantiating GeminiImageClient for model '{target_model}'")
    return GeminiImageClient(api_key=target_key, model=target_model)
