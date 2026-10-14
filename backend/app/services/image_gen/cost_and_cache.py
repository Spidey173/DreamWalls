import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)

class ImageCostTracker:
    """
    Cost controls for external image-editing generation models:
    - Enforces global daily cap (from settings.IMAGE_GLOBAL_DAILY_CAP)
    - Enforces per-IP daily limit (from settings.IMAGE_PER_IP_DAILY_LIMIT)
    - Logs estimated cost per request based on selected model
    """
    def __init__(self):
        self.global_daily_counts: Dict[str, int] = {}
        self.ip_daily_counts: Dict[str, Dict[str, int]] = {}

    def _get_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_model_cost_per_image(self, model_name: str) -> float:
        m = model_name.lower()
        if "mock" in m:
            return 0.0
        if "flash" in m:
            return 0.02
        if "imagen" in m:
            return 0.04
        return 0.03

    def check_limits(self, client_ip: str) -> Tuple[bool, str]:
        today = self._get_today_str()
        global_count = self.global_daily_counts.get(today, 0)
        if global_count >= settings.IMAGE_GLOBAL_DAILY_CAP:
            msg = f"Global daily cap of {settings.IMAGE_GLOBAL_DAILY_CAP} image generations reached."
            logger.warning(f"Cost control rejection: {msg}")
            return False, msg

        ip_counts = self.ip_daily_counts.setdefault(client_ip, {})
        ip_count = ip_counts.get(today, 0)
        if ip_count >= settings.IMAGE_PER_IP_DAILY_LIMIT:
            msg = f"Per-IP daily limit of {settings.IMAGE_PER_IP_DAILY_LIMIT} image generations reached for {client_ip}."
            logger.warning(f"Cost control rejection: {msg}")
            return False, msg

        return True, ""

    def record_generation(self, client_ip: str, model_name: str, count: int = 1):
        today = self._get_today_str()
        self.global_daily_counts[today] = self.global_daily_counts.get(today, 0) + count

        ip_counts = self.ip_daily_counts.setdefault(client_ip, {})
        ip_counts[today] = ip_counts.get(today, 0) + count

        unit_cost = self.get_model_cost_per_image(model_name)
        total_cost = unit_cost * count

        logger.info(
            f"Image generation cost logged: ${total_cost:.4f} for {count} image(s) "
            f"using '{model_name}' (IP: {client_ip}, Daily IP Total: {ip_counts[today]}, Global Total: {self.global_daily_counts[today]})"
        )

    def get_stats(self) -> dict:
        today = self._get_today_str()
        return {
            "date": today,
            "global_count": self.global_daily_counts.get(today, 0),
            "global_cap": settings.IMAGE_GLOBAL_DAILY_CAP,
            "per_ip_limit": settings.IMAGE_PER_IP_DAILY_LIMIT,
        }

class ImageSchemeCache:
    """
    Caches repainted scheme results by (image_hash, scheme_ids, model_name)
    to eliminate duplicate API charges and latency.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.IMAGE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_cache_key(
        self,
        image_bytes: bytes,
        main_id: int,
        accent_id: int,
        trim_id: int,
        model_name: str
    ) -> str:
        img_sha = hashlib.sha256(image_bytes).hexdigest()[:16]
        model_slug = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        return f"{img_sha}_m{main_id}_a{accent_id}_t{trim_id}_{model_slug}"

    def get(self, cache_key: str) -> Optional[Tuple[np.ndarray, dict]]:
        img_path = self.cache_dir / f"{cache_key}.jpg"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if img_path.exists() and meta_path.exists():
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    logger.info(f"Image scheme cache hit for key: {cache_key}")
                    return img_bgr, meta
            except Exception as e:
                logger.warning(f"Failed to read image cache for key {cache_key}: {e}")
        return None

    def set(self, cache_key: str, image_bgr: np.ndarray, meta: dict):
        img_path = self.cache_dir / f"{cache_key}.jpg"
        meta_path = self.cache_dir / f"{cache_key}.json"
        try:
            cv2.imwrite(str(img_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved image scheme to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to write image cache for key {cache_key}: {e}")

# Global singleton instances
cost_tracker = ImageCostTracker()
scheme_cache = ImageSchemeCache()
