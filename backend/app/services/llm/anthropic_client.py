import base64
from typing import Optional
import anthropic
from app.services.llm.base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        # Ensure media_type is supported by Anthropic: image/jpeg, image/png, image/gif, image/webp
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        first_content = message.content[0]
        return getattr(first_content, "text", "")

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        message = await self.client.messages.create(**kwargs)
        first_content = message.content[0]
        return getattr(first_content, "text", "")
