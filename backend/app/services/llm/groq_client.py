import base64
import logging
from typing import Optional
from groq import AsyncGroq
from app.services.llm.base import BaseLLMClient

logger = logging.getLogger("paintmywall.groq")

# Models known to only accept string content (no image_url)
TEXT_ONLY_PREFIXES = ("openai/", "allam", "whisper")

class GroqClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        vision_model: str = "qwen/qwen3.8-27b"
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided")
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.vision_model = vision_model

    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_data}"

        # If the primary model is text-only (like openai/gpt-oss-120b), route vision to vision_model
        target_model = self.vision_model if any(self.model.startswith(p) for p in TEXT_ONLY_PREFIXES) else self.model
        logger.info(f"Groq analyze_image using model: {target_model}")

        chat_completion = await self.client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content or ""

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(f"Groq generate_text using model: {self.model}")
        chat_completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content or ""
