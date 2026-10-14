import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

from app.core.config import settings

logger = logging.getLogger("paintmywall.sam")

_encoder_session: Optional[ort.InferenceSession] = None
_decoder_session: Optional[ort.InferenceSession] = None
_embedding_cache: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}

class SamService:
    def __init__(
        self,
        encoder_path: Optional[Path] = None,
        decoder_path: Optional[Path] = None
    ):
        self.encoder_path = encoder_path or (settings.MODEL_DIR / "mobile_sam_encoder.onnx")
        self.decoder_path = decoder_path or (settings.MODEL_DIR / "mobile_sam_decoder.onnx")

    def _ensure_models_loaded(self):
        global _encoder_session, _decoder_session
        if _encoder_session is None or _decoder_session is None:
            if not self.encoder_path.exists() or not self.decoder_path.exists():
                raise FileNotFoundError(
                    f"MobileSAM ONNX models not found at {self.encoder_path} and {self.decoder_path}"
                )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            _encoder_session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            _decoder_session = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded MobileSAM encoder and decoder ONNX sessions.")

    def get_or_compute_embedding(
        self,
        cache_key: str,
        image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Computes or reuses cached image embedding for ultra-fast point prompts (~15ms).
        """
        global _embedding_cache
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        self._ensure_models_loaded()
        orig_h, orig_w = image_bgr.shape[:2]
        scale = 1024.0 / max(orig_h, orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized.astype(np.float32)

        outputs = _encoder_session.run(["image_embeddings"], {"input_image": padded})
        embedding = outputs[0]  # Shape: (1, 256, 64, 64)

        _embedding_cache[cache_key] = (embedding, (orig_h, orig_w))
        return embedding, (orig_h, orig_w)

    def segment_with_points(
        self,
        image_bgr: np.ndarray,
        cache_key: str,
        points: List[Dict[str, float]],  # list of {"x": float [0,1], "y": float [0,1], "label": 1 | 0}
        refine_edges: bool = True
    ) -> np.ndarray:
        """
        Segment object using multiple positive (1) and negative (0) tap points.
        Returns binary uint8 mask (0 or 255) at original image resolution.
        """
        self._ensure_models_loaded()
        orig_h, orig_w = image_bgr.shape[:2]
        embedding, (orig_h, orig_w) = self.get_or_compute_embedding(cache_key, image_bgr)

        if not points:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

        scale = 1024.0 / max(orig_h, orig_w)

        coords_list = []
        labels_list = []
        for p in points:
            # Normalized [0, 1] -> original pixels -> scaled to 1024 system
            px = float(p.get("x", 0.5)) * orig_w * scale
            py = float(p.get("y", 0.5)) * orig_h * scale
            lbl = float(p.get("label", 1.0))
            coords_list.append([px, py])
            labels_list.append(lbl)

        point_coords = np.array([coords_list], dtype=np.float32)
        point_labels = np.array([labels_list], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)
        orig_im_size = np.array([float(orig_h), float(orig_w)], dtype=np.float32)

        dec_out = _decoder_session.run(["masks", "iou_predictions"], {
            "image_embeddings": embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size
        })

        raw_masks = dec_out[0]  # shape (1, num_masks, orig_h, orig_w)
        # Select best mask based on highest logit
        raw_mask = raw_masks[0, 0]
        binary_mask = (raw_mask > 0.0).astype(np.uint8) * 255

        if refine_edges:
            guide_gray = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            from app.services.segmentation_service import SegmentationService
            refined = SegmentationService.guided_filter(guide_gray, binary_mask.astype(np.float32) / 255.0, r=6, eps=1e-3)
            binary_mask = (refined > 0.5).astype(np.uint8) * 255

        return binary_mask
