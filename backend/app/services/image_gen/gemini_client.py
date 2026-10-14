import asyncio
import logging
from typing import Optional
from google import genai
from google.genai import types
from app.services.image_gen.base import BaseImageClient

logger = logging.getLogger(__name__)

class GeminiImageClient(BaseImageClient):
    """
    Client for Google Gemini / Imagen image editing models:
    - Supports conversational multimodal editing (e.g. gemini-2.5-flash-image, gemini-3.1-flash-image)
    - Supports Imagen image generation (e.g. imagen-3.0-generate-002)
    - Easily switchable via IMAGE_MODEL environment variable
    """
    is_mock: bool = False

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-image"):
        if not api_key:
            raise ValueError("API key must be provided for GeminiImageClient")
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate_scheme_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> bytes:
        def _call_api():
            m_lower = self.model.lower()
            if "imagen" in m_lower:
                logger.info(f"Calling Imagen 3 model '{self.model}' for exterior scheme rendering...")
                result = self.client.models.generate_images(
                    model=self.model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/jpeg",
                    )
                )
                if result.generated_images and len(result.generated_images) > 0:
                    return result.generated_images[0].image.image_bytes
                raise ValueError(f"No generated images returned by model '{self.model}'")
            else:
                logger.info(f"Calling Gemini multimodal image model '{self.model}' with prompt: {prompt[:80]}...")
                img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[img_part, prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    )
                )
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if part.inline_data and part.inline_data.data:
                                    logger.info(f"Successfully received image bytes from '{self.model}' ({len(part.inline_data.data)} bytes)")
                                    return part.inline_data.data
                raise ValueError(f"No image part found in response from model '{self.model}'")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call_api)
