import cv2
import numpy as np
from app.services.image_gen.base import BaseImageClient

class MockImageClient(BaseImageClient):
    """
    Mock image client used strictly when IMAGE_ALLOW_MOCK=True is enabled.
    Returns the UNMODIFIED ORIGINAL PHOTO with an unmistakable, prominent
    'MOCK - NO IMAGE API KEY' banner across the image so it is never
    confused with real AI generation output.
    """
    is_mock: bool = True

    def __init__(self, model: str = "mock-image-model"):
        self.model = model

    async def generate_scheme_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode input image bytes in MockImageClient")

        h, w = img_bgr.shape[:2]
        out_bgr = img_bgr.copy()

        # Create an unmistakable, prominent semi-transparent dark banner across the image center
        banner_h = max(70, int(h * 0.12))
        y_center = int(h * 0.45)
        y1 = max(0, y_center - banner_h // 2)
        y2 = min(h, y_center + banner_h // 2)

        overlay = out_bgr.copy()
        # Dark crimson bar (#8B0000 -> BGR 0, 0, 139)
        overlay[y1:y2, :] = [20, 20, 160]
        cv2.addWeighted(overlay, 0.85, out_bgr, 0.15, 0, out_bgr)

        # Border lines for the banner
        cv2.line(out_bgr, (0, y1), (w, y1), (255, 255, 255), 2)
        cv2.line(out_bgr, (0, y2), (w, y2), (255, 255, 255), 2)

        # Text 1: Primary Warning
        text1 = "MOCK - NO IMAGE API KEY"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.6, w / 750.0)
        thickness = max(2, int(font_scale * 2.2))
        (text_w1, text_h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        tx1 = max(10, (w - text_w1) // 2)
        ty1 = y_center - 5

        # Drop shadow
        cv2.putText(out_bgr, text1, (tx1 + 2, ty1 + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out_bgr, text1, (tx1, ty1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Text 2: Sub-info
        text2 = "Demo Mode: Unpainted Original Photo (IMAGE_ALLOW_MOCK=true)"
        sub_scale = font_scale * 0.52
        sub_thick = 1
        (text_w2, text_h2), _ = cv2.getTextSize(text2, font, sub_scale, sub_thick)
        tx2 = max(10, (w - text_w2) // 2)
        ty2 = y_center + text_h1 + 8
        cv2.putText(out_bgr, text2, (tx2, ty2), font, sub_scale, (220, 220, 220), sub_thick, cv2.LINE_AA)

        _, encoded = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return encoded.tobytes()
