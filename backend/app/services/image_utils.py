import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger("paintmywall.image_utils")

def detect_and_crop_camera_bar(
    image_bgr: np.ndarray,
    max_bar_ratio: float = 0.08
) -> Tuple[np.ndarray, int]:
    """
    Detects if the bottom of the photo has a camera info bar / device watermark banner
    (common in phone camera shots, e.g. white or black strip with date/model stamp).
    
    Returns:
        (cropped_image, num_rows_cropped)
    """
    h, w = image_bgr.shape[:2]
    max_rows = int(h * max_bar_ratio)
    
    if max_rows < 10:
        return image_bgr, 0

    # Inspect the bottom pad (last 15 rows or max_rows)
    bottom_pad = image_bgr[-min(15, h):, :]
    is_white_pad = float(np.mean(bottom_pad >= 245)) > 0.85
    is_black_pad = float(np.mean(bottom_pad <= 20)) > 0.85

    if not (is_white_pad or is_black_pad):
        return image_bgr, 0

    target_val = 245 if is_white_pad else 20
    comp_op = (lambda r: r >= target_val) if is_white_pad else (lambda r: r <= target_val)

    bar_start_y = h
    for y in range(h - 1, h - max_rows, -1):
        row = image_bgr[y, :]
        match_pct = float(np.mean(comp_op(row)))
        # In a banner with small text/watermark, >75% of the row matches the background color
        if match_pct > 0.75:
            bar_start_y = y
        else:
            # Reached actual photographic content
            break

    num_rows_to_crop = h - bar_start_y
    if num_rows_to_crop >= 10:
        logger.info(f"Detected camera watermark/info bar: cropping {num_rows_to_crop} rows from bottom ({h} -> {bar_start_y})")
        cropped = image_bgr[:bar_start_y, :].copy()
        return cropped, num_rows_to_crop

    return image_bgr, 0
