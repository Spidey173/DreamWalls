import cv2
import numpy as np
from typing import Tuple, Union, Literal, Optional, List, Dict
from pathlib import Path
from app.services.shortlist_service import hex_to_rgb

class RecolorService:
    def feather_mask(self, mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
        """
        Feathers the binary mask edges slightly using Gaussian blur.
        Returns a float32 mask normalized to [0.0, 1.0].
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        if blur_radius <= 0:
            return (mask > 127).astype(np.float32)

        ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=1.5, sigmaY=1.5)
        return blurred.astype(np.float32) / 255.0

    def get_shade_lab(self, hex_color: str) -> Tuple[float, float, float]:
        """
        Convert hex color to OpenCV LAB values (L in [0, 255], a in [0, 255], b in [0, 255]).
        """
        r, g, b = hex_to_rgb(hex_color)
        rgb_pixel = np.uint8([[[r, g, b]]])
        bgr_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2BGR)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        l_val = float(lab_pixel[0, 0, 0])
        a_val = float(lab_pixel[0, 0, 1])
        b_val = float(lab_pixel[0, 0, 2])
        return (l_val, a_val, b_val)

    @staticmethod
    def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 25, eps: float = 0.04) -> np.ndarray:
        """
        Fast Guided Filter for illumination smoothing.
        I: guide image (float32 [0, 1])
        p: input illumination / shading field (float32)
        r: local filter radius
        eps: regularization parameter
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
        return q

    @staticmethod
    def soft_highlight_rolloff(
        L: np.ndarray,
        knee: float = 210.0,
        max_val: float = 252.0
    ) -> np.ndarray:
        """
        Smooth C^1 continuous highlight compression curve:
        - For L <= knee: completely linear (slope = 1.0), preserving exact shade tone.
        - For L > knee: smoothly compresses towards max_val using a hyperbolic tangent shoulder,
          preventing harsh clipping, white blowouts, and washed-out glowing artifacts.
        """
        L = np.maximum(L, 0.0)
        knee = min(knee, max_val - 1.0)
        shoulder = knee + (max_val - knee) * np.tanh((L - knee) / max(max_val - knee, 1e-4))
        rolled = np.where(L <= knee, L, shoulder)
        return np.clip(rolled, 0.0, 255.0)

    @staticmethod
    def get_layer_feather_radius(layer: str, img_shape: Tuple[int, int]) -> int:
        """
        Feather 1-2px for accent/trim (scaled with image size); keep larger (3-5px) for main wall.
        """
        h, w = img_shape[:2]
        scale = max(h, w) / 1000.0
        if layer in ("accent", "trim_gate"):
            return max(1, int(round(1.5 * scale)))
        else:
            return max(3, int(round(4.0 * scale)))

    def prepare_shading_context(
        self,
        image_bgr: np.ndarray,
        wall_mask: np.ndarray,
        surface_type: Literal["interior", "exterior", "auto"] = "auto",
        r_large: Optional[int] = None,
        eps_large: Optional[float] = None,
        r_mid: Optional[int] = None,
        eps_mid: Optional[float] = None,
        gamma_mid: Optional[float] = None,
        texture_strength: Optional[float] = None,
        feather_radius: int = 5,
        shared_facade_mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Precompute 3-layer illumination and texture decomposition once for the image and mask.
        Layer 1: Large-scale macro lighting (kept intact for 3D realism)
        Layer 2: Mid-frequency stains/blotches (compressed with gamma to remove defects)
        Layer 3: Fine micro-texture (preserved at tunable strength for authentic plaster/brick grain)

        Recolor math:
        Normalize shading against the 90th percentile of the zone's lit pixels
        (computed over the whole facade for consistency across zones), not the mean.
        Target L = shade_L * (L_pixel / L_ref).
        """
        # Feather mask to [0.0, 1.0]
        alpha = self.feather_mask(wall_mask, blur_radius=feather_radius)
        alpha_3d = np.expand_dims(alpha, axis=-1)

        # Convert original image to float32 LAB
        orig_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_orig = orig_lab[:, :, 0]
        guide = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)

        # Tuned defaults per scene type
        if surface_type == "exterior":
            r_large = 20 if r_large is None else r_large
            eps_large = 0.02 if eps_large is None else eps_large
            r_mid = 6 if r_mid is None else r_mid
            eps_mid = 0.005 if eps_mid is None else eps_mid
            gamma_mid = 0.65 if gamma_mid is None else gamma_mid
            texture_strength = 1.0 if texture_strength is None else texture_strength
        else:
            # Interior scenes:
            # 1. Broad smooth macro illumination to preserve 3D daylight/shadow shape (r=35)
            # 2. Suppress mid-frequency water stains and cement blotches (gamma_mid = 0.0)
            # 3. Delicate micro-texture (r_fine=3, strength=0.35) for authentic plaster tooth
            r_large = 35 if r_large is None else r_large
            eps_large = 0.04 if eps_large is None else eps_large
            r_mid = 3 if r_mid is None else r_mid
            eps_mid = 0.005 if eps_mid is None else eps_mid
            gamma_mid = 0.0 if gamma_mid is None else gamma_mid
            texture_strength = 0.35 if texture_strength is None else texture_strength

        # 1. Macro Illumination (broad daylight/shadow field)
        l_large = self.guided_filter(guide, l_orig, r=r_large, eps=eps_large)
        
        # Shared reference normalization:
        # For exterior: 90th percentile of lit pixels over whole facade
        # For interior: 55th percentile of lit pixels (diffuse ambient wall lighting) to prevent
        # direct window sunbeams from dragging ambient walls down into dark, muddy grey.
        pct_norm = 90.0 if surface_type == "exterior" else 55.0

        if shared_facade_mask is not None and np.any(shared_facade_mask > 50):
            facade_bool = shared_facade_mask > 50
            lit_bool = facade_bool & (l_orig > 30)
            if not np.any(lit_bool):
                lit_bool = facade_bool
            l_ref = float(np.percentile(l_large[lit_bool], pct_norm))
        else:
            mask_bool = alpha > 0.2
            lit_bool = mask_bool & (l_orig > 30)
            if not np.any(lit_bool):
                lit_bool = mask_bool
            l_ref = float(np.percentile(l_large[lit_bool], pct_norm)) if np.any(lit_bool) else 128.0
            
        f_macro = l_large / max(l_ref, 1.0)

        # 2. Mid-Frequency Stains (water stains, paint discoloration, plaster unevenness)
        if gamma_mid > 0.0:
            l_mid = self.guided_filter(guide, l_orig, r=r_mid, eps=eps_mid)
            r_mid_ratio = l_mid / np.maximum(l_large, 1.0)
            f_mid = np.power(np.maximum(r_mid_ratio, 1e-4), gamma_mid)
        else:
            f_mid = 1.0

        # 3. Fine Micro-Texture (authentic tooth, plaster grain, mortar lines)
        l_fine = self.guided_filter(guide, l_orig, r=r_mid, eps=eps_mid)
        d_fine = l_orig / np.maximum(l_fine, 1.0)
        f_fine = 1.0 + texture_strength * (d_fine - 1.0)

        # Combined modulation factor across all layers
        shading_combined = f_macro * f_mid * f_fine

        return {
            "wall_mask": wall_mask,
            "alpha": alpha,
            "alpha_3d": alpha_3d,
            "orig_lab": orig_lab,
            "shading_combined": shading_combined,
            "shading_ratio": f_macro,
            "micro_texture": f_fine,
            "surface_type": surface_type,
            "orig_bgr": image_bgr,
            "orig_bgr_float": image_bgr.astype(np.float32),
            "l_ref": l_ref
        }

    def compute_repainted_layer_bgr(
        self,
        context: dict,
        target_hex: str,
        highlight_knee: Optional[float] = None,
        highlight_max: float = 252.0
    ) -> np.ndarray:
        """
        Computes repainted BGR pixels BEFORE blending with any background.
        Target L = shade_L * (L_pixel / L_ref) with a soft shoulder instead of hard clipping.
        For dark colors (target_L < 90), compresses shadow crushing so colors stay rich and vibrant.
        Preserves natural window sunlight highlights with realistic chromatic adaptation.
        """
        target_l, target_a, target_b = self.get_shade_lab(target_hex)
        is_interior = context.get("surface_type") != "exterior"

        if is_interior and "shading_ratio" in context:
            shading_ratio = context["shading_ratio"]
            micro_texture = context["micro_texture"]

            # Shadow floor compression for deep/dark colors to prevent black void crushing
            if target_l < 90.0:
                ratio_mapped = np.where(
                    shading_ratio < 1.0,
                    1.0 + (shading_ratio - 1.0) * 0.45,
                    1.0 + (shading_ratio - 1.0) * 0.85
                )
            else:
                ratio_mapped = shading_ratio

            l_new = target_l * ratio_mapped * micro_texture

            # Window sunlight chromatic adaptation (warm amber bounce in direct sun patches)
            l_orig = context["orig_lab"][:, :, 0]
            sun_weight = np.clip((l_orig - 175.0) / 45.0, 0.0, 1.0)
            a_channel = target_a + sun_weight * 2.0
            b_channel = target_b + sun_weight * 6.0
        else:
            l_new = target_l * context["shading_combined"]
            a_channel = target_a
            b_channel = target_b

        knee = highlight_knee if highlight_knee is not None else max(target_l, 210.0)
        l_repainted = self.soft_highlight_rolloff(l_new, knee=knee, max_val=highlight_max)

        repainted_lab = np.zeros_like(context["orig_lab"])
        repainted_lab[:, :, 0] = l_repainted
        repainted_lab[:, :, 1] = a_channel
        repainted_lab[:, :, 2] = b_channel

        repainted_bgr = cv2.cvtColor(
            np.clip(repainted_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        ).astype(np.float32)
        return repainted_bgr

    def recolor_with_context(
        self,
        context: dict,
        target_hex: str,
        highlight_knee: Optional[float] = None,
        highlight_max: float = 252.0
    ) -> np.ndarray:
        """Instantaneous single-layer recolor with bit-identical unmasked pixel preservation."""
        repainted_bgr = self.compute_repainted_layer_bgr(
            context, target_hex, highlight_knee=highlight_knee, highlight_max=highlight_max
        )

        result = context["orig_bgr"].copy()
        alpha_3d = context["alpha_3d"]
        active = context["alpha"] > 0.001
        if np.any(active):
            blended = (alpha_3d * repainted_bgr) + ((1.0 - alpha_3d) * context["orig_bgr_float"])
            result[active] = np.clip(blended[active], 0, 255).astype(np.uint8)

        return result

    def prepare_multi_layer_context(
        self,
        image_bgr: np.ndarray,
        masks: Dict[str, np.ndarray],
        surface_type: str = "exterior"
    ) -> Dict[str, dict]:
        """
        Precompute shading context for each non-empty layer using a shared facade reference.
        Reused across all 4 schemes.
        """
        # Shared facade mask: union of all non-empty masks
        facade_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for m in masks.values():
            if m is not None and np.any(m > 0):
                facade_mask = cv2.bitwise_or(facade_mask, (m > 127).astype(np.uint8) * 255)

        contexts = {}
        for layer_name in ["main", "accent", "trim_gate"]:
            mask = masks.get(layer_name)
            if mask is not None and np.any(mask > 0):
                feather_r = self.get_layer_feather_radius(layer_name, image_bgr.shape[:2])
                contexts[layer_name] = self.prepare_shading_context(
                    image_bgr=image_bgr,
                    wall_mask=mask,
                    surface_type=surface_type,
                    feather_radius=feather_r,
                    shared_facade_mask=facade_mask
                )
        return contexts

    def recolor_multi_layer(
        self,
        image_bgr: np.ndarray,
        layer_contexts: Dict[str, dict],
        scheme_hexes: Dict[str, str],  # {"main": "#hex", "accent": "#hex", "trim_gate": "#hex"}
        layer_order: Optional[List[str]] = None,
        highlight_knee: Optional[float] = None,
        highlight_max: float = 252.0,
        facade_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sequential layer blending onto the RUNNING RESULT:
        result = alpha * repainted + (1 - alpha) * result
        Ensures zone edges blend with neighbouring zones instead of showing unpainted wall!
        Enforces 100% byte-identity for all unselected pixels outside the facade mask.
        """
        if layer_order is None:
            layer_order = ["main", "accent", "trim_gate"]

        running_result = image_bgr.copy()
        running_float = running_result.astype(np.float32)

        for layer in layer_order:
            if layer not in layer_contexts or layer not in scheme_hexes:
                continue
            ctx = layer_contexts[layer]
            alpha_3d = ctx["alpha_3d"]
            active = ctx["alpha"] > 0.001
            if not np.any(active):
                continue

            # 1. Repainted pixels before blending
            repainted_bgr = self.compute_repainted_layer_bgr(
                ctx, scheme_hexes[layer], highlight_knee=highlight_knee, highlight_max=highlight_max
            )

            # 2. Blend sequentially onto running_result
            blended = (alpha_3d * repainted_bgr) + ((1.0 - alpha_3d) * running_float)
            running_float[active] = blended[active]
            running_result[active] = np.clip(running_float[active], 0, 255).astype(np.uint8)

        # Enforce exact byte-identity for all unselected pixels outside facade_mask
        if facade_mask is None:
            facade_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            for ctx in layer_contexts.values():
                if "wall_mask" in ctx and ctx["wall_mask"] is not None:
                    facade_mask = cv2.bitwise_or(facade_mask, (ctx["wall_mask"] > 127).astype(np.uint8) * 255)

        if facade_mask is not None:
            unselected = (facade_mask == 0)
            running_result[unselected] = image_bgr[unselected]

        return running_result

    def recolor(
        self,
        image_bgr: np.ndarray,
        wall_mask: np.ndarray,
        target_hex: str,
        surface_type: Literal["interior", "exterior", "auto"] = "auto",
        r_large: Optional[int] = None,
        eps_large: Optional[float] = None,
        r_mid: Optional[int] = None,
        eps_mid: Optional[float] = None,
        gamma_mid: Optional[float] = None,
        texture_strength: Optional[float] = None,
        highlight_knee: Optional[float] = None,
        highlight_max: float = 252.0,
        feather_radius: int = 5
    ) -> np.ndarray:
        ctx = self.prepare_shading_context(
            image_bgr=image_bgr,
            wall_mask=wall_mask,
            surface_type=surface_type,
            r_large=r_large,
            eps_large=eps_large,
            r_mid=r_mid,
            eps_mid=eps_mid,
            gamma_mid=gamma_mid,
            texture_strength=texture_strength,
            feather_radius=feather_radius
        )
        return self.recolor_with_context(
            context=ctx,
            target_hex=target_hex,
            highlight_knee=highlight_knee,
            highlight_max=highlight_max
        )

    def recolor_to_bytes(
        self,
        image_bytes: bytes,
        wall_mask: np.ndarray,
        target_hex: str,
        **kwargs
    ) -> bytes:
        """Helper to recolor directly from and to bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image bytes")
        
        result_bgr = self.recolor(img_bgr, wall_mask, target_hex, **kwargs)
        success, encoded = cv2.imencode(".jpg", result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise ValueError("Failed to encode repainted image")
        return encoded.tobytes()

