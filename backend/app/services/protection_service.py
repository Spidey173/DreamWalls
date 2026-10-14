import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

from app.core.config import settings
from app.services.segmentation_service import SegmentationService

logger = logging.getLogger("paintmywall.protection")

class ProtectionService:
    """
    Manages auto-protection of non-paintable exterior surfaces using SegFormer:
    - Sky
    - Road / Ground / Earth / Sidewalk / Grass / Path
    - Vegetation / Trees / Plants
    - Vehicles (Cars, Trucks, Buses, Vans, Bikes)

    Hard constraint:
    Protected areas are dilated by ~2px and subtracted from ALL zones.
    Hatched overlay is generated for user visibility.
    Sky pixel fraction inside any zone is verified to be < 1% (strictly 0.0%).
    """

    SKY_CLASSES = ["sky"]
    GROUND_CLASSES = [
        "road", "sidewalk", "earth", "grass", "path", "sand", "dirt track", "land", "floor"
    ]
    VEG_CLASSES = [
        "tree", "plant", "palm", "flower"
    ]
    VEHICLE_CLASSES = [
        "car", "bus", "truck", "van", "boat", "bicycle", "minibike", "airplane"
    ]

    def __init__(self, segmentation_service: Optional[SegmentationService] = None):
        self.seg_service = segmentation_service or SegmentationService()

    def compute_protection_masks(
        self,
        image_bgr: np.ndarray,
        dilation_kernel_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs SegFormer ONNX once to extract:
        1. protected_mask (dilated by ~2px): binary uint8 [0, 255]
        2. sky_mask: binary uint8 [0, 255]
        3. hatched_overlay: 4-channel BGRA image with diagonal hatch lines over protected areas.
        """
        orig_h, orig_w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.seg_service.session is None:
            logger.warning("SegFormer session not loaded; returning empty protection masks.")
            empty = np.zeros((orig_h, orig_w), dtype=np.uint8)
            overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            return empty, empty, overlay

        # Standard 512x512 ImageNet input
        resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        outputs = self.seg_service.session.run(["logits"], {"pixel_values": input_tensor})
        logits = outputs[0][0]  # (150, 128, 128)

        sky_ids = self.seg_service._resolve_class_ids(self.SKY_CLASSES)
        ground_ids = self.seg_service._resolve_class_ids(self.GROUND_CLASSES)
        veg_ids = self.seg_service._resolve_class_ids(self.VEG_CLASSES)
        veh_ids = self.seg_service._resolve_class_ids(self.VEHICLE_CLASSES)
        prot_ids = sky_ids + ground_ids + veg_ids + veh_ids

        # Paintable classes (wall, building, house, skyscraper)
        paint_ids = self.seg_service._resolve_class_ids(["wall", "building", "house", "skyscraper"])

        all_rel_ids = list(dict.fromkeys(paint_ids + prot_ids))
        other_ids = [i for i in range(logits.shape[0]) if i not in all_rel_ids]
        other_max = np.max(logits[other_ids], axis=0) if other_ids else np.full((128, 128), -1e4, dtype=np.float32)

        # Upsample logits to full resolution BEFORE argmax
        num_channels = len(all_rel_ids) + 1
        upsampled = np.zeros((num_channels, orig_h, orig_w), dtype=np.float32)
        for idx, cid in enumerate(all_rel_ids):
            upsampled[idx] = cv2.resize(logits[cid], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        upsampled[-1] = cv2.resize(other_max, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        argmax_map = np.argmax(upsampled, axis=0)
        id_map = np.array([all_rel_ids[idx] if idx < len(all_rel_ids) else -1 for idx in range(num_channels)])
        full_class_map = id_map[argmax_map]

        sky_mask = np.isin(full_class_map, sky_ids).astype(np.uint8) * 255
        prot_raw = np.isin(full_class_map, prot_ids).astype(np.uint8) * 255

        # Dilate protected area by ~2px (5x5 kernel gives 2px radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        protected_mask = cv2.dilate(prot_raw, kernel, iterations=1)

        # Generate hatched pattern overlay
        hatched_overlay = self.generate_hatched_overlay(protected_mask)

        logger.info(
            f"Auto-protect computed: sky={np.mean(sky_mask > 0)*100:.1f}%, "
            f"protected_dilated={np.mean(protected_mask > 0)*100:.1f}%"
        )
        return protected_mask, sky_mask, hatched_overlay

    @staticmethod
    def generate_hatched_overlay(
        protected_mask: np.ndarray,
        hatch_spacing: int = 14,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Creates an RGBA image where protected pixels display a 45-degree diagonal hatched pattern.
        """
        h, w = protected_mask.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Draw diagonal lines
        diag_canvas = np.zeros((h, w), dtype=np.uint8)
        for c in range(-h, w + h, hatch_spacing):
            cv2.line(diag_canvas, (c, 0), (c + h, h), 255, line_width)

        hatched_pixels = (diag_canvas > 0) & (protected_mask > 0)
        # Coral/amber warning color: BGRA = (40, 50, 220, 180)
        overlay[hatched_pixels] = [40, 50, 220, 180]
        return overlay

    @staticmethod
    def clean_selection_mask(
        mask: np.ndarray,
        protected_mask: Optional[np.ndarray] = None,
        min_blob_area_fraction: float = 0.001,  # 0.1% of image area
        smooth_ksize: int = 3
    ) -> np.ndarray:
        """
        Post-selection processing required after every edit:
        1. Fill holes.
        2. Remove disconnected blobs under 0.1% of total image area.
        3. Smooth edges lightly (morphological close/open).
        4. Hard constraint: subtract protected_mask.
        """
        if mask is None or not np.any(mask > 0):
            return np.zeros_like(mask) if mask is not None else np.zeros((512, 512), dtype=np.uint8)

        bin_mask = (mask > 127).astype(np.uint8) * 255
        h, w = bin_mask.shape[:2]
        total_area = h * w

        # 1. Fill holes by flood-filling inverted mask from outer border
        padded_mask = cv2.copyMakeBorder(bin_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        inv_mask = cv2.bitwise_not(padded_mask)
        ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(inv_mask, ff_mask, (0, 0), 0)
        # Any remaining 255s in inv_mask were enclosed internal holes
        mask_filled = cv2.bitwise_or(padded_mask, inv_mask)[1:-1, 1:-1]

        # 2. Remove blobs under 0.1% of image area
        min_pixels = max(10, int(min_blob_area_fraction * total_area))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
        cleaned = np.zeros_like(bin_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
                cleaned[labels == i] = 255

        # 3. Smooth edges lightly
        if smooth_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_ksize, smooth_ksize))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 4. Hard constraint: subtract protected area
        if protected_mask is not None:
            cleaned[protected_mask > 0] = 0

        return cleaned

    @staticmethod
    def calculate_sky_fraction(zone_mask: np.ndarray, sky_mask: np.ndarray) -> float:
        """
        Calculates the fraction of sky pixels inside a zone mask:
        fraction = sum(zone & sky) / sum(zone)
        """
        zone_pixels = np.sum(zone_mask > 0)
        if zone_pixels == 0:
            return 0.0
        sky_in_zone = np.sum((zone_mask > 0) & (sky_mask > 0))
        return float(sky_in_zone / zone_pixels)
