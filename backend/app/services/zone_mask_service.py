import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from app.services.protection_service import ProtectionService

logger = logging.getLogger("paintmywall.zone_masks")

class ZoneMaskService:
    """
    Manages 3 non-overlapping zone masks per session:
    - 'main': Main exterior walls / body
    - 'accent': Borders, projecting slabs, architectural pop-outs, window frames
    - 'trim_gate': Gates, exterior doors, grills, metal trims

    Zone Priority Rule:
    - Last-edited layer wins: when updating the active layer, it takes pixels
      from all other layers (other_masks[active_mask > 0] = 0).
    - ensure_no_overlap is retained as a safety net.

    Auto-Protection Constraint:
    - Protected pixels (sky, ground, vegetation, vehicles) are strictly subtracted
      from ALL zones.
    - Sky fraction in each zone is tracked and verified (< 1.0%).
    """
    LAYERS = ["main", "accent", "trim_gate"]
    LAYER_NAMES = {"main": "Main Wall", "accent": "Borders & Slabs", "trim_gate": "Gate & Trims"}

    @staticmethod
    def get_mask_filename(layer: str) -> str:
        return f"mask_{layer}.png"

    def init_session_masks(
        self,
        session_dir: Path,
        wall_mask: np.ndarray,
        protected_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Initializes masks for a new session:
        - main: initialized with the auto-detected wall mask (minus protected areas)
        - accent: empty mask of same dimensions
        - trim_gate: empty mask of same dimensions
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        h, w = wall_mask.shape[:2]

        main_mask = (wall_mask > 127).astype(np.uint8) * 255
        if protected_mask is not None:
            main_mask[protected_mask > 0] = 0

        # Post-clean main_mask
        main_mask = ProtectionService.clean_selection_mask(main_mask, protected_mask=protected_mask)

        accent_mask = np.zeros((h, w), dtype=np.uint8)
        trim_gate_mask = np.zeros((h, w), dtype=np.uint8)

        masks = {
            "main": main_mask,
            "accent": accent_mask,
            "trim_gate": trim_gate_mask
        }
        self.save_masks(session_dir, masks)
        return masks

    def load_masks(self, session_dir: Path, fallback_shape: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        Loads all 3 layer masks from session_dir.
        If any layer mask file is missing, creates an empty mask.
        """
        masks = {}
        for layer in self.LAYERS:
            fpath = session_dir / self.get_mask_filename(layer)
            if fpath.exists():
                mask = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[layer] = mask
                    continue

            # Missing mask file fallback
            if "main" in masks:
                ref_h, ref_w = masks["main"].shape[:2]
            elif fallback_shape is not None:
                ref_h, ref_w = fallback_shape
            else:
                ref_h, ref_w = (512, 512)
            masks[layer] = np.zeros((ref_h, ref_w), dtype=np.uint8)

        return masks

    def save_masks(self, session_dir: Path, masks: Dict[str, np.ndarray]):
        """
        Saves all 3 layer masks to session_dir as PNGs.
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        for layer, mask in masks.items():
            fpath = session_dir / self.get_mask_filename(layer)
            cv2.imwrite(str(fpath), mask)

    def load_protection_artifacts(self, session_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Loads cached protected_mask.png, sky_mask.png, and protected_hatched.png if present.
        """
        prot_p = session_dir / "protected_mask.png"
        sky_p = session_dir / "sky_mask.png"
        hatch_p = session_dir / "protected_hatched.png"

        prot_m = cv2.imread(str(prot_p), cv2.IMREAD_GRAYSCALE) if prot_p.exists() else None
        sky_m = cv2.imread(str(sky_p), cv2.IMREAD_GRAYSCALE) if sky_p.exists() else None
        hatch_m = cv2.imread(str(hatch_p), cv2.IMREAD_UNCHANGED) if hatch_p.exists() else None

        return prot_m, sky_m, hatch_m

    def save_protection_artifacts(
        self,
        session_dir: Path,
        protected_mask: np.ndarray,
        sky_mask: np.ndarray,
        hatched_overlay: Optional[np.ndarray] = None
    ):
        """
        Saves protected_mask.png, sky_mask.png, and protected_hatched.png.
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(session_dir / "protected_mask.png"), protected_mask)
        cv2.imwrite(str(session_dir / "sky_mask.png"), sky_mask)
        if hatched_overlay is not None:
            cv2.imwrite(str(session_dir / "protected_hatched.png"), hatched_overlay)

    def update_layer(
        self,
        masks: Dict[str, np.ndarray],
        active_layer: str,
        new_mask: np.ndarray,
        protected_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Update the active layer mask with 'last-edited layer wins' rule:
        1. Clean selection and enforce auto-protection subtraction on new_mask.
        2. The active layer takes pixels from all other layers.
        3. For all other layers L != active_layer, any pixel where new_mask > 0 is cleared.
        4. Strict zero-overlap safety net.
        """
        if active_layer not in self.LAYERS:
            raise ValueError(f"Unknown layer '{active_layer}'. Must be one of {self.LAYERS}")

        binary_new = (new_mask > 127).astype(np.uint8) * 255

        # Hard constraint: subtract protected area from active layer
        if protected_mask is not None:
            binary_new[protected_mask > 0] = 0

        # Post-selection cleanup: fill holes, remove small blobs < 0.1%, smooth edges lightly
        binary_new = ProtectionService.clean_selection_mask(binary_new, protected_mask=protected_mask)

        updated_masks = {}

        # Last-edited layer wins: clear claimed pixels from other layers
        active_pixels = binary_new > 0
        for layer, m in masks.items():
            if layer == active_layer:
                updated_masks[layer] = binary_new.copy()
            else:
                m_copy = m.copy()
                m_copy[active_pixels] = 0
                if protected_mask is not None:
                    m_copy[protected_mask > 0] = 0
                updated_masks[layer] = m_copy

        # Safety net pass to ensure strictly zero overlap
        updated_masks = self.ensure_no_overlap(
            updated_masks,
            priority_order=[active_layer] + [l for l in self.LAYERS if l != active_layer]
        )
        return updated_masks

    def ensure_no_overlap(
        self,
        masks: Dict[str, np.ndarray],
        priority_order: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Safety net: verifies that no pixel belongs to more than one layer.
        Priority order determines which layer retains the pixel on overlap.
        Default priority order: trim_gate > accent > main.
        """
        if priority_order is None:
            priority_order = ["trim_gate", "accent", "main"]

        cleaned = {}
        claimed = None

        for layer in priority_order:
            if layer not in masks:
                continue
            m = (masks[layer] > 127).astype(np.uint8) * 255
            if claimed is None:
                claimed = (m > 0)
                cleaned[layer] = m
            else:
                # Subtract already claimed pixels
                m[claimed] = 0
                claimed = claimed | (m > 0)
                cleaned[layer] = m

        # Ensure all standard layers exist in dict
        for layer in self.LAYERS:
            if layer not in cleaned:
                if claimed is not None:
                    cleaned[layer] = np.zeros(claimed.shape, dtype=np.uint8)
                else:
                    cleaned[layer] = np.zeros((512, 512), dtype=np.uint8)

        return cleaned

    def get_combined_mask(self, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns the union of all 3 layers.
        Useful for combined facade analysis and backward compatibility.
        """
        combined = None
        for layer in self.LAYERS:
            if layer in masks:
                m = (masks[layer] > 127).astype(np.uint8) * 255
                if combined is None:
                    combined = m.copy()
                else:
                    combined = cv2.bitwise_or(combined, m)
        if combined is None:
            return np.zeros((512, 512), dtype=np.uint8)
        return combined

    def get_coverage_pct(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Returns coverage percentage of each layer relative to total image pixels.
        """
        coverage = {}
        for layer in self.LAYERS:
            m = masks.get(layer)
            if m is not None and m.size > 0:
                coverage[layer] = round(float(np.mean(m > 127) * 100), 2)
            else:
                coverage[layer] = 0.0
        return coverage

    def get_sky_fractions(self, masks: Dict[str, np.ndarray], sky_mask: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Calculates sky pixel fraction for every zone layer.
        Requirement: must be < 1% (0.01).
        """
        fractions = {}
        if sky_mask is None:
            return {layer: 0.0 for layer in self.LAYERS}

        for layer in self.LAYERS:
            m = masks.get(layer)
            if m is not None and np.any(m > 0):
                fractions[layer] = round(ProtectionService.calculate_sky_fraction(m, sky_mask) * 100, 3)
            else:
                fractions[layer] = 0.0
        return fractions

    def reset_layer(
        self,
        masks: Dict[str, np.ndarray],
        layer: str,
        protected_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Resets a specific layer to empty without affecting other layers.
        """
        updated = {k: v.copy() for k, v in masks.items()}
        if layer in updated:
            updated[layer] = np.zeros_like(updated[layer])
        if protected_mask is not None:
            for l_name in updated:
                updated[l_name][protected_mask > 0] = 0
        return updated

    def reset_all(
        self,
        session_dir: Path,
        fallback_shape: Optional[Tuple[int, int]] = None,
        protected_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Resets all zones back to default:
        - main is restored from mask.png (minus protected areas)
        - accent and trim_gate are cleared to empty
        """
        masks = self.load_masks(session_dir, fallback_shape=fallback_shape)
        orig_mask_path = session_dir / "mask.png"
        if orig_mask_path.exists():
            base_m = cv2.imread(str(orig_mask_path), cv2.IMREAD_GRAYSCALE)
            if base_m is not None:
                if protected_mask is not None:
                    base_m[protected_mask > 0] = 0
                masks["main"] = ProtectionService.clean_selection_mask(base_m, protected_mask=protected_mask)
        for l_name in ["accent", "trim_gate"]:
            if l_name in masks:
                masks[l_name] = np.zeros_like(masks["main"])
        self.save_masks(session_dir, masks)
        return masks
