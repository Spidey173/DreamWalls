from typing import Optional
from google import genai
from google.genai import types
from app.services.llm.base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        # google-genai supports async client or thread offloading
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ]
        )
        return response.text or ""

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        config = types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text or ""
