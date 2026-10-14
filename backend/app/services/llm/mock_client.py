import json
import re
from typing import Optional
from app.services.llm.base import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client used during testing or when API keys are not supplied.
    Produces valid schema-compliant responses.
    """

    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        # Heuristic check on image to mock realistic scene detection
        is_exterior = False
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # Top 15% color check: if blue/gray daylight sky
                top_bgr = np.mean(img[:int(img.shape[0] * 0.15), :], axis=(0, 1))
                if top_bgr[0] > 180 and top_bgr[1] > 180:  # bright sky
                    is_exterior = True
        except Exception:
            pass

        if is_exterior:
            return json.dumps({
                "scene_type": "exterior",
                "building_type": "House",
                "lighting": "bright",
                "light_conditions": "sunny",
                "current_wall_color_hex": "#E8E2D5",
                "surroundings": "Suburban street with metal gate",
                "accent_colors": ["#4A5568", "#2D3748"],
                "style": "Contemporary",
                "mood": "Stately and welcoming"
            })

        return json.dumps({
            "scene_type": "interior",
            "room_type": "living room",
            "lighting": "bright",
            "current_wall_color_hex": "#EDE5D8",
            "floor_color_hex": "#8A5A36",
            "dominant_furniture_colors": ["#2A475E", "#C77966"],
            "style": "Contemporary",
            "mood": "Warm and inviting"
        })


    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Extract candidate IDs from prompt if available
        # The prompt will contain candidate IDs
        ids = [int(m) for m in re.findall(r'"id":\s*(\d+)', prompt)]
        selected_ids = ids[:4] if len(ids) >= 4 else (ids + [1, 2, 3, 4])[:4]
        
        selections = [
            {"id": selected_ids[0], "reason": "Brings warmth and harmonizes gracefully with the wood flooring."},
            {"id": selected_ids[1], "reason": "Offers an elegant neutral backdrop that accentuates ambient daylight."},
            {"id": selected_ids[2], "reason": "Creates an eye-catching complementary contrast with dominant furniture pieces."},
            {"id": selected_ids[3], "reason": "Delivers a soothing, serene tone that opens up the living space."}
        ]
        return json.dumps({"selections": selections})
