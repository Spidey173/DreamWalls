import cv2
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

from app.services.sam_service import SamService
import app.services.sam_service as sam_mod
from app.services.protection_service import ProtectionService

logger = logging.getLogger("paintmywall.auto_regions")


@dataclass
class CandidateRegion:
    region_id: int
    mask: np.ndarray                 # binary uint8 [0, 255]
    pixel_count: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]    # normalized (x, y) in [0, 1]


class AutoRegionsService:
    """
    Pre-segments an exterior photo into candidate regions using MobileSAM
    with a dense grid of point prompts.
    Provides tap lookup returning the smallest candidate region containing the tap point.
    """

    def __init__(self, sam_service: Optional[SamService] = None, protection_service: Optional[ProtectionService] = None):
        self.sam_service = sam_service or SamService()
        self.protection_service = protection_service or ProtectionService()

    def compute_candidate_regions(
        self,
        image_bgr: np.ndarray,
        cache_key: str,
        grid_size: int = 16,
        protected_mask: Optional[np.ndarray] = None
    ) -> List[CandidateRegion]:
        """
        Runs MobileSAM on a grid of positive points, dedupes by IoU,
        filters out protected sky/ground areas, and returns sorted candidate regions.
        """
        self.sam_service._ensure_models_loaded()
        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Compute or retrieve image embedding (cached)
        embedding, (h, w) = self.sam_service.get_or_compute_embedding(cache_key, image_bgr)

        # 2. Generate grid of candidate tap points
        xs = np.linspace(0.05, 0.95, grid_size)
        ys = np.linspace(0.05, 0.95, grid_size)
        valid_points: List[Tuple[float, float]] = []

        for y in ys:
            for x in xs:
                px = int(round(x * (orig_w - 1)))
                py = int(round(y * (orig_h - 1)))
                if protected_mask is not None:
                    if protected_mask[py, px] > 128:
                        continue  # Skip points landing on sky, ground, vegetation
                valid_points.append((float(x), float(y)))

        scale = 1024.0 / max(orig_h, orig_w)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
        orig_im_size = np.array([float(orig_h), float(orig_w)], dtype=np.float32)

        # 3. Decode masks for all valid points
        raw_components: List[Tuple[np.ndarray, int, Tuple[int, int, int, int], Tuple[float, float]]] = []

        for x, y in valid_points:
            coords = np.array([[[x * orig_w * scale, y * orig_h * scale]]], dtype=np.float32)
            labels = np.array([[1.0]], dtype=np.float32)

            dec_out = sam_mod._decoder_session.run(["masks"], {
                "image_embeddings": embedding,
                "point_coords": coords,
                "point_labels": labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
                "orig_im_size": orig_im_size
            })

            raw_m = (dec_out[0][0, 0] > 0.0).astype(np.uint8) * 255
            cleaned = self.protection_service.clean_selection_mask(raw_m, protected_mask=protected_mask)

            if np.sum(cleaned > 0) < 100:
                continue

            # Also decompose connected components to separate disconnected slabs/pillars
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
            for lbl in range(1, num_labels):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area >= 300:  # Minimum useful architectural feature size
                    x1 = int(stats[lbl, cv2.CC_STAT_LEFT])
                    y1 = int(stats[lbl, cv2.CC_STAT_TOP])
                    bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                    bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                    bbox = (x1, y1, x1 + bw, y1 + bh)
                    cent = (float(centroids[lbl][0] / orig_w), float(centroids[lbl][1] / orig_h))
                    comp_mask = (labels_im == lbl).astype(np.uint8) * 255
                    raw_components.append((comp_mask, area, bbox, cent))

        # 4. Deduplicate candidate masks by IoU > 0.70 (smallest first)
        raw_components.sort(key=lambda item: item[1])
        unique_regions: List[CandidateRegion] = []
        reg_id = 1

        for comp_mask, area, bbox, cent in raw_components:
            m_bool = comp_mask > 0

            # Drop if > 50% overlap with protected area
            if protected_mask is not None:
                prot_overlap = np.sum(m_bool & (protected_mask > 0))
                if (prot_overlap / area) > 0.50:
                    continue

            duplicate = False
            for kept in unique_regions:
                k_bool = kept.mask > 0
                inter = np.sum(m_bool & k_bool)
                union = np.sum(m_bool | k_bool)
                iou = (inter / union) if union > 0 else 0.0
                if iou > 0.70:
                    duplicate = True
                    break

            if not duplicate:
                unique_regions.append(CandidateRegion(
                    region_id=reg_id,
                    mask=comp_mask,
                    pixel_count=area,
                    bbox=bbox,
                    centroid=cent
                ))
                reg_id += 1

        # Re-sort ascending by pixel count for "smallest containing region" lookup
        unique_regions.sort(key=lambda r: r.pixel_count)
        return unique_regions

    def find_region_at_tap(
        self,
        candidates: List[CandidateRegion],
        tap_x: float,
        tap_y: float,
        image_shape: Tuple[int, int]
    ) -> Optional[CandidateRegion]:
        """
        Finds the smallest candidate region containing the normalized (tap_x, tap_y).
        Returns None if tap falls outside all candidate regions.
        """
        orig_h, orig_w = image_shape[:2]
        px = int(np.clip(round(tap_x * (orig_w - 1)), 0, orig_w - 1))
        py = int(np.clip(round(tap_y * (orig_h - 1)), 0, orig_h - 1))

        # Candidates are already sorted ascending by area, so first match is smallest
        for cand in candidates:
            if cand.mask[py, px] > 0:
                return cand

        return None

    def save_regions(self, session_dir: Path, candidates: List[CandidateRegion]) -> None:
        """
        Saves candidate regions to compressed NumPy archive in session_dir.
        """
        if not candidates:
            return

        session_dir.mkdir(parents=True, exist_ok=True)
        archive_path = session_dir / "auto_regions.npz"

        masks_stack = np.stack([c.mask > 0 for c in candidates], axis=0)  # shape (N, H, W) bool
        bboxes = np.array([list(c.bbox) for c in candidates], dtype=np.int32)
        areas = np.array([c.pixel_count for c in candidates], dtype=np.int64)
        centroids = np.array([[c.centroid[0], c.centroid[1]] for c in candidates], dtype=np.float32)
        region_ids = np.array([c.region_id for c in candidates], dtype=np.int32)

        np.savez_compressed(
            str(archive_path),
            masks=masks_stack,
            bboxes=bboxes,
            areas=areas,
            centroids=centroids,
            region_ids=region_ids
        )
        logger.info(f"Saved {len(candidates)} auto-regions to {archive_path}")

    def load_regions(self, session_dir: Path) -> Optional[List[CandidateRegion]]:
        """
        Loads candidate regions from session directory if cached.
        """
        archive_path = session_dir / "auto_regions.npz"
        if not archive_path.exists():
            return None

        try:
            with np.load(str(archive_path)) as data:
                masks_stack = data["masks"]
                bboxes = data["bboxes"]
                areas = data["areas"]
                centroids = data["centroids"]
                region_ids = data["region_ids"]

                candidates: List[CandidateRegion] = []
                for i in range(len(region_ids)):
                    m = (masks_stack[i].astype(np.uint8) * 255)
                    cand = CandidateRegion(
                        region_id=int(region_ids[i]),
                        mask=m,
                        pixel_count=int(areas[i]),
                        bbox=(int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3])),
                        centroid=(float(centroids[i][0]), float(centroids[i][1]))
                    )
                    candidates.append(cand)

                candidates.sort(key=lambda r: r.pixel_count)
                return candidates
        except Exception as e:
            logger.warning(f"Failed to load cached auto_regions from {archive_path}: {e}")
            return None

    def load_or_compute(
        self,
        session_dir: Path,
        image_bgr: np.ndarray,
        cache_key: str,
        protected_mask: Optional[np.ndarray] = None,
        grid_size: int = 16
    ) -> Tuple[List[CandidateRegion], float]:
        """
        Loads cached candidate regions or computes them, returning (candidates, compute_time_ms).
        """
        cached = self.load_regions(session_dir)
        if cached is not None:
            return cached, 0.0

        t0 = time.time()
        computed = self.compute_candidate_regions(
            image_bgr=image_bgr,
            cache_key=cache_key,
            grid_size=grid_size,
            protected_mask=protected_mask
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        self.save_regions(session_dir, computed)
        return computed, elapsed_ms
