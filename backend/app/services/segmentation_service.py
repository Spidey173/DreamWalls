import hashlib
import os
import json
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Optional, Literal, Set, Dict, List
import logging

from app.core.config import settings

logger = logging.getLogger("paintmywall.segmentation")

_shared_session: Optional[ort.InferenceSession] = None
_shared_model_path: Optional[Path] = None

EXTERIOR_PAINT_CLASS_NAMES = ["wall", "building", "house", "skyscraper"]
EXTERIOR_PROTECTED_CLASS_NAMES = [
    "sky", "tree", "plant", "door", "windowpane", "window",
    "fence", "gate", "railing", "road", "sidewalk", "path",
    "grass", "earth", "ground", "car", "bus", "truck", "person",
    "signboard", "pole", "light", "awning", "streetlight"
]
EXTERIOR_INDICATOR_CLASS_NAMES = ["sky", "tree", "road", "grass", "earth"]

class SegmentationService:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else settings.get_segformer_model_path()
        self.session: Optional[ort.InferenceSession] = None
        self._init_session()
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self._load_id2label()

    def _init_session(self):
        global _shared_session, _shared_model_path

        # If a shared session is already initialized with this model, reuse it
        if _shared_session is not None and _shared_model_path == self.model_path:
            self.session = _shared_session
            return

        paths_to_try = [self.model_path]
        b2_path = settings.MODEL_DIR / "segformer_b2_ade20k.onnx"
        b0_path = settings.MODEL_DIR / "segformer_b0_ade20k.onnx"
        for candidate in [b2_path, b0_path]:
            if candidate != self.model_path and candidate.exists():
                paths_to_try.append(candidate)

        for p in paths_to_try:
            if p and p.exists():
                try:
                    # Constrain thread pools to avoid excessive memory overhead on cloud containers
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    self.session = ort.InferenceSession(
                        str(p),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"]
                    )
                    self.model_path = p
                    _shared_session = self.session
                    _shared_model_path = p
                    logger.info(f"Loaded SegFormer ONNX session from {p} with constrained thread pool")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize SegFormer ONNX session from {p}: {e}")
                    self.session = None
        logger.warning("No SegFormer ONNX model could be loaded.")

    def _load_id2label(self):
        label_file = settings.DATA_DIR / "ade20k_id2label.json"
        if label_file.exists():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.id2label = {int(k): str(v).strip() for k, v in data.items()}
                    self.label2id = {v.lower(): k for k, v in self.id2label.items()}
                logger.info(f"Loaded {len(self.id2label)} ADE20K classes from {label_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to read {label_file}: {e}")

        # Fallback to standard 150 ADE20K labels if file is unavailable
        logger.warning("ADE20K label file not found; using built-in baseline mappings.")
        self.id2label = {0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree", 5: "ceiling", 6: "road"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _resolve_class_ids(self, class_names: List[str]) -> List[int]:
        """
        Maps class names to ADE20K class IDs.
        Skips missing class names with a warning without crashing.
        """
        resolved: List[int] = []
        for name in class_names:
            key = name.strip().lower()
            if key in self.label2id:
                resolved.append(self.label2id[key])
            else:
                logger.warning(f"Class name '{name}' not found in ADE20K id2label; skipping without crashing.")
        return resolved

    def compute_image_hash(self, image_bytes: bytes, surface_type: str = "auto") -> str:
        # Incorporate model path and surface_type to avoid stale cached masks
        model_name = self.model_path.name if self.model_path else "default"
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        return f"{model_name}_{surface_type}_{content_hash}"

    def get_cached_mask_path(self, image_hash: str) -> Path:
        return settings.CACHE_DIR / f"{image_hash}_mask.png"

    @staticmethod
    def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
        """
        Fast Guided Filter for boundary refinement.
        Aligns coarse segmentation mask 'p' to sharp photorealistic edges in 'I'.
        Both I and p must be float32 in range [0.0, 1.0].
        """
        ksize = (2 * r + 1, 2 * r + 1)
        if len(I.shape) == 3:
            I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        else:
            I_gray = I

        mean_I = cv2.boxFilter(I_gray, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I_gray * p, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I_gray * I_gray, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

        q = mean_a * I_gray + mean_b
        return np.clip(q, 0.0, 1.0)

    @staticmethod
    def snap_interior_boundaries(mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        """
        Architectural Boundary Snapping for Interior Wall Segmentation:
        1. Baseboard / Skirting Line: Detects straight horizontal edge in lower 70%-96% of frame
           to prevent paint bleeding onto floor skirting tiles.
        2. Ceiling Line: Detects straight horizontal ceiling junction in top 3%-20% of frame.
        3. Corner Junctions: Detects sharp vertical wall transitions at left and right corners.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        h, w = gray.shape[:2]
        refined = mask.copy()

        # 1. Baseboard / Skirting Line snap
        y_bot_min, y_bot_max = int(h * 0.70), int(h * 0.96)
        quads = [(0.20, 0.40), (0.40, 0.60), (0.60, 0.80)]
        skirting_candidates = []
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        for x1, x2 in quads:
            prof = np.mean(sobel_y[y_bot_min:y_bot_max, int(w * x1):int(w * x2)], axis=1)
            peak = int(np.argmax(prof))
            if prof[peak] > 35.0:
                skirting_candidates.append(y_bot_min + peak)

        if skirting_candidates and (max(skirting_candidates) - min(skirting_candidates) <= 6):
            skirting_y = int(np.median(skirting_candidates))
            refined[skirting_y:, :] = 0

        # 2. Ceiling Line snap
        y_top_min, y_top_max = int(h * 0.03), int(h * 0.20)
        ceiling_candidates = []
        for x1, x2 in quads:
            prof = np.mean(sobel_y[y_top_min:y_top_max, int(w * x1):int(w * x2)], axis=1)
            peak = int(np.argmax(prof))
            if prof[peak] > 35.0:
                ceiling_candidates.append(y_top_min + peak)

        if ceiling_candidates and (max(ceiling_candidates) - min(ceiling_candidates) <= 6):
            ceiling_y = int(np.median(ceiling_candidates))
            refined[:ceiling_y, :] = 0

        # 3. Right vertical wall corner junction
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        x_r_min, x_r_max = int(w * 0.88), int(w * 0.99)
        prof_r = np.mean(sobel_x[int(h * 0.25):int(h * 0.75), x_r_min:x_r_max], axis=0)
        peak_r = int(np.argmax(prof_r))
        if prof_r[peak_r] > 50.0:
            corner_r = x_r_min + peak_r
            refined[:, corner_r:] = 0

        # 4. Left vertical wall corner junction (e.g. window reveal or perpendicular wall)
        x_l_min, x_l_max = int(w * 0.065), int(w * 0.12)
        prof_l = np.mean(sobel_x[int(h * 0.25):int(h * 0.75), x_l_min:x_l_max], axis=0)
        peak_l = int(np.argmax(prof_l))
        if prof_l[peak_l] > 50.0:
            corner_l = x_l_min + peak_l
            refined[:, :corner_l] = 0

        return refined

    def detect_walls(
        self,
        image_bytes: bytes,
        surface_type: Literal["interior", "exterior", "auto"] = "auto",
        force_recompute: bool = False,
        guided_radius: int = 8,
        guided_eps: float = 1e-3,
        feather_radius: int = 3
    ) -> Tuple[np.ndarray, Path]:
        """
        Produce a high-resolution, edge-refined binary paint mask:
        1. Runs SegFormer (B0 or B2) inference at 512x512 to obtain 150-channel logits.
        2. Discovers surface type (interior or exterior) if 'auto'.
        3. Bilinearly upsamples logits to original image resolution BEFORE argmax.
        4. For exterior: paints wall + building + house (+ skyscraper), subtracting protected classes.
        5. For interior: paints wall (class 0) with fallback to wall + building + ceiling.
        6. Refines mask edges with a guided filter using the original image as guide.
        7. Returns (mask_array, mask_path).
        """
        img_hash = self.compute_image_hash(image_bytes, surface_type=surface_type)
        mask_path = self.get_cached_mask_path(img_hash)

        # Check cache
        if mask_path.exists() and not force_recompute:
            logger.info(f"Loading cached wall mask from {mask_path}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask, mask_path

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image bytes into valid image.")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.session is not None:
            # SegFormer expects 512x512 normalized with ImageNet mean and std
            resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            resized = resized.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (resized - mean) / std

            # Transpose (512, 512, 3) -> (1, 3, 512, 512)
            input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

            # Run inference
            outputs = self.session.run(["logits"], {"pixel_values": input_tensor})
            logits = outputs[0][0]  # Shape: (150, 128, 128)

            # Determine surface type if auto
            active_surface = surface_type
            if active_surface == "auto":
                coarse_preds = np.argmax(logits, axis=0)
                indicator_ids = self._resolve_class_ids(EXTERIOR_INDICATOR_CLASS_NAMES)
                indicator_coverage = float(np.isin(coarse_preds, indicator_ids).mean())
                building_id = self.label2id.get("building", 1)
                wall_id = self.label2id.get("wall", 0)
                building_coverage = float((coarse_preds == building_id).mean())
                wall_coverage = float((coarse_preds == wall_id).mean())

                if indicator_coverage > 0.15 or (building_coverage > 0.15 and wall_coverage < 0.05):
                    active_surface = "exterior"
                    logger.info(f"Auto-detected scene as EXTERIOR (indicators: {indicator_coverage*100:.1f}%, building: {building_coverage*100:.1f}%)")
                else:
                    active_surface = "interior"
                    logger.info(f"Auto-detected scene as INTERIOR (indicators: {indicator_coverage*100:.1f}%, wall: {wall_coverage*100:.1f}%)")

            if active_surface == "exterior":
                # Exterior masking:
                # Paint classes: wall, building, house, skyscraper
                paint_ids = self._resolve_class_ids(EXTERIOR_PAINT_CLASS_NAMES)
                protected_ids = self._resolve_class_ids(EXTERIOR_PROTECTED_CLASS_NAMES)

                # Union of relevant classes to upsample
                rel_ids = list(dict.fromkeys(paint_ids + protected_ids))
                paint_indices_in_rel = [i for i, cid in enumerate(rel_ids) if cid in paint_ids]
                other_ids = [i for i in range(logits.shape[0]) if i not in rel_ids]

                # Compute maximum over remaining classes at coarse resolution
                other_max = np.max(logits[other_ids], axis=0) if other_ids else np.full((128, 128), -1e4, dtype=np.float32)

                # Bilinearly upsample relevant channels + other_max to full image resolution BEFORE argmax
                num_channels = len(rel_ids) + 1
                selected_logits = np.zeros((num_channels, orig_h, orig_w), dtype=np.float32)
                for idx, cid in enumerate(rel_ids):
                    selected_logits[idx] = cv2.resize(logits[cid], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                selected_logits[-1] = cv2.resize(other_max, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                # Full-resolution argmax over the upsampled logits
                argmax_map = np.argmax(selected_logits, axis=0)
                raw_paint = np.isin(argmax_map, paint_indices_in_rel).astype(np.float32)

            else:
                # Interior masking:
                # Class 0 is wall in ADE20K
                wall_id = self.label2id.get("wall", 0)
                wall_logit = logits[wall_id]
                rest_indices = [idx for idx in range(logits.shape[0]) if idx != wall_id]
                other_max = np.max(logits[rest_indices], axis=0)
                margin = wall_logit - other_max

                # Bilinearly interpolate margin directly to full resolution
                upsampled_margin = cv2.resize(margin, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                raw_paint = (upsampled_margin > 0.0).astype(np.float32)

                # Fallback if SegFormer found very few wall pixels (< 5%)
                if float(np.mean(raw_paint)) < 0.05:
                    building_id = self.label2id.get("building", 1)
                    ceiling_id = self.label2id.get("ceiling", 5)
                    alt_max = np.maximum(logits[wall_id], np.maximum(logits[building_id], logits[ceiling_id]))
                    fallback_rest = [idx for idx in range(logits.shape[0]) if idx not in (wall_id, building_id, ceiling_id)]
                    fb_rest_max = np.max(logits[fallback_rest], axis=0)
                    fb_margin = cv2.resize(alt_max - fb_rest_max, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    if float(np.mean(fb_margin > 0.0)) >= 0.10:
                        raw_paint = (fb_margin > 0.0).astype(np.float32)

            # Step 3: Refine mask edges with guided filter using original image as guide
            guide_gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            refined_wall = self.guided_filter(guide_gray, raw_paint, r=guided_radius, eps=guided_eps)

            # Threshold refined mask to binary [0, 255]
            binary_mask = (refined_wall > 0.5).astype(np.uint8) * 255

            # Interior Architectural Snapping:
            # Prevents bleed over baseboard/skirting board tiles, jagged ceiling edges,
            # and perpendicular window reveals by snapping to verified architectural edge gradients.
            if active_surface == "interior":
                binary_mask = self.snap_interior_boundaries(binary_mask, img_bgr)

            # Step 4: Small feathering for smooth edge transition
            if feather_radius > 0:
                k = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
                full_mask = cv2.GaussianBlur(binary_mask, (k, k), sigmaX=1.0, sigmaY=1.0)
            else:
                full_mask = binary_mask

        else:
            logger.warning("SegFormer session not available; using geometric fallback wall mask.")
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            top = int(orig_h * 0.05)
            bottom = int(orig_h * 0.70)
            full_mask[top:bottom, :] = 255

        # Cache mask to disk
        cv2.imwrite(str(mask_path), full_mask)
        logger.info(f"Saved computed wall mask to {mask_path}")

        return full_mask, mask_path

