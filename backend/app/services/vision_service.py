import json
import re
import logging
from typing import Optional
from pydantic import ValidationError
from app.schemas.vision import RoomAnalysis
from app.services.llm.base import BaseLLMClient
from app.services.llm.factory import get_llm_client

logger = logging.getLogger("paintmywall.vision")

VISION_PROMPT = """Analyze this photo for an architectural paint visualization tool.
First, determine whether the image is an INTERIOR room or an EXTERIOR building.
You MUST output ONLY a valid JSON object (no markdown formatting, no code fences, no extra text) matching one of the two schemas below:

If INTERIOR:
{
  "scene_type": "interior",
  "room_type": "<e.g. living room, bedroom, kitchen, dining room, office>",
  "lighting": "<bright | dim | mixed>",
  "current_wall_color_hex": "<estimated hex code of main wall, e.g. #F4EBD9>",
  "floor_color_hex": "<estimated hex code of floor/carpet, e.g. #6B4D3C>",
  "dominant_furniture_colors": ["<hex1>", "<hex2>"],
  "style": "<e.g. Modern, Minimalist, Traditional, Scandinavian, Bohemian>",
  "mood": "<e.g. Warm and cozy, Fresh and airy, Sophisticated and moody>"
}

If EXTERIOR:
{
  "scene_type": "exterior",
  "building_type": "<e.g. house, villa, apartment, bungalow, commercial facade>",
  "lighting": "<bright | dim | mixed>",
  "light_conditions": "<sunny | overcast | golden_hour | mixed>",
  "current_wall_color_hex": "<estimated hex code of building facade/walls, e.g. #E8E2D5>",
  "surroundings": "<e.g. street with gate, lush garden, urban road>",
  "accent_colors": ["<trim_or_gate_hex>", "<accent_hex>"],
  "style": "<e.g. Contemporary, Modern, Colonial, Minimalist>",
  "mood": "<e.g. Elegant and welcoming, Stately, Vibrant, Clean and modern>"
}

Guidelines:
1. 'scene_type' MUST be exactly 'interior' or 'exterior'.
2. 'lighting' MUST be exactly: 'bright', 'dim', or 'mixed'.
3. For exterior, 'light_conditions' MUST be: 'sunny', 'overcast', 'golden_hour', or 'mixed'. Do NOT include 'floor_color_hex' or 'room_type' for exterior.
4. For interior, return 'dominant_furniture_colors'. For exterior, return 'accent_colors' (doors, gate, trim, accents).
5. All hex values must be valid 6-character hex strings starting with #.
"""

def extract_json(raw_text: str) -> str:
    """Extract JSON object from raw LLM response string."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
    return text

class VisionService:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or get_llm_client()

    async def analyze_room(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> RoomAnalysis:
        """
        Analyze room or building exterior photo using LLM vision.
        Validates with Pydantic; retries once on invalid JSON.
        """
        raw_output = await self.llm_client.analyze_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=VISION_PROMPT
        )

        try:
            cleaned_json = extract_json(raw_output)
            data = json.loads(cleaned_json)
            return RoomAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"First vision analysis attempt failed ({e}); retrying with correction prompt...")

            correction_prompt = f"""The previous response had a schema or parsing error: {e}.
Please analyze the image again and output STRICT, VALID JSON ONLY.
Must have 'scene_type': 'interior' | 'exterior'.
If interior: include room_type, floor_color_hex, current_wall_color_hex, lighting, style, mood, dominant_furniture_colors.
If exterior: include building_type, surroundings, light_conditions, current_wall_color_hex, lighting, style, mood, accent_colors.
Output JSON only.
"""
            retry_output = await self.llm_client.analyze_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=correction_prompt
            )

            try:
                cleaned_retry = extract_json(retry_output)
                retry_data = json.loads(cleaned_retry)
                return RoomAnalysis.model_validate(retry_data)
            except Exception as final_err:
                logger.error(f"Vision analysis retry also failed: {final_err}. Falling back to default baseline.")
                return RoomAnalysis(
                    scene_type="interior",
                    room_type="living room",
                    lighting="bright",
                    current_wall_color_hex="#EFEAE1",
                    floor_color_hex="#7A5230",
                    dominant_furniture_colors=["#2C3E50", "#95A5A6"],
                    style="Contemporary",
                    mood="Warm and welcoming"
                )

