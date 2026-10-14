import asyncio
import base64
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.models.paint import Paint
from app.schemas.recolor import SchemeResult, SchemeShadeInfo
from app.schemas.selection import SchemeSelectionItem
from app.services.image_gen.factory import get_image_client
from app.services.image_gen.cost_and_cache import cost_tracker, scheme_cache
from app.services.shortlist_service import hex_to_rgb, rgb_to_lab
from app.services.recolor_service import RecolorService

logger = logging.getLogger(__name__)

def calculate_delta_e_rgb(rgb1: Tuple[int, int, int], hex2: str) -> float:
    """Computes CIE76 deltaE between an RGB tuple and target hex color."""
    r2, g2, b2 = hex_to_rgb(hex2)
    l1, a1, b1 = rgb_to_lab(rgb1[0], rgb1[1], rgb1[2])
    l2, a2, b2 = rgb_to_lab(r2, g2, b2)
    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

class ImageSchemeService:
    """
    Orchestrates exterior scheme rendering via image-editing AI models:
    - Builds strict role-mapped prompts with real Asian Paints catalog shades
    - Runs 4 generations in parallel with per-call timeouts and retries
    - Composites via SegFormer building mask so background is exactly original
    - Logs outside-mask drift before compositing
    - Checks median repainted main wall color and warns if deltaE > 12
    - Enforces per-IP and global daily cost limits, caches by (image, schemes)
    - Supports per-scheme regeneration and fallback to precise mask recolor
    """

    def __init__(self):
        self.recolor_service = RecolorService()

    def build_prompt(self, main_paint: Paint, accent_paint: Paint, trim_paint: Paint) -> str:
        return (
            f"Repaint this building's exterior. "
            f"Main walls: {main_paint.name} ({main_paint.hex}). "
            f"Projecting slabs, window surrounds, side panels and tower: {accent_paint.name} ({accent_paint.hex}). "
            f"Gate: {trim_paint.name} ({trim_paint.hex}). "
            f"Keep the structure, sky, road, ground, vegetation and text unchanged. "
            f"Do not add lights or objects. Photorealistic."
        )

    def composite_with_mask(
        self,
        original_bgr: np.ndarray,
        generated_bgr: np.ndarray,
        building_mask: np.ndarray,
        s_idx: int
    ) -> Tuple[np.ndarray, float, float]:
        """
        Composites generated image strictly inside the building mask.
        Sky, road, ground, trees outside the building remain 100% original.
        Logs pixel drift before compositing.
        """
        orig_h, orig_w = original_bgr.shape[:2]
        gen_h, gen_w = generated_bgr.shape[:2]

        # Verify aspect ratio and dimensions, resize if generated size differs
        if (gen_h, gen_w) != (orig_h, orig_w):
            logger.info(
                f"Scheme {s_idx}: Resizing generated image from ({gen_w}x{gen_h}) to original ({orig_w}x{orig_h})"
            )
            generated_bgr = cv2.resize(generated_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        outside_mask = (building_mask == 0)
        if np.any(outside_mask):
            diff = np.abs(generated_bgr[outside_mask].astype(float) - original_bgr[outside_mask].astype(float))
            mean_delta = float(np.mean(diff))
            pct_changed = float(np.mean(np.linalg.norm(diff, axis=-1) > 10.0) * 100.0)
        else:
            mean_delta = 0.0
            pct_changed = 0.0

        logger.info(
            f"Scheme {s_idx} post-process: raw AI image drift outside building mask: "
            f"mean delta={mean_delta:.2f}, {pct_changed:.1f}% pixels changed before compositing"
        )

        composite_bgr = original_bgr.copy()
        mask_binary = (building_mask > 0)
        composite_bgr[mask_binary] = generated_bgr[mask_binary]

        return composite_bgr, mean_delta, pct_changed

    def check_main_wall_color(
        self,
        composite_bgr: np.ndarray,
        main_mask: np.ndarray,
        requested_hex: str,
        s_idx: int
    ) -> Tuple[float, str]:
        """
        Samples median color of the repainted main wall inside the main zone mask.
        Computes deltaE against the requested catalog hex. Warns in logs if deltaE > 12.
        """
        sample_mask = (main_mask > 128)
        if not np.any(sample_mask):
            sample_mask = (main_mask > 0)

        if not np.any(sample_mask):
            logger.warning(f"Scheme {s_idx}: main wall mask is empty, skipping color check")
            return 0.0, requested_hex

        wall_pixels_bgr = composite_bgr[sample_mask]
        median_b = float(np.median(wall_pixels_bgr[:, 0]))
        median_g = float(np.median(wall_pixels_bgr[:, 1]))
        median_r = float(np.median(wall_pixels_bgr[:, 2]))

        median_rgb = (int(round(median_r)), int(round(median_g)), int(round(median_b)))
        sampled_hex = f"#{median_rgb[0]:02X}{median_rgb[1]:02X}{median_rgb[2]:02X}"

        delta_e = calculate_delta_e_rgb(median_rgb, requested_hex)
        logger.info(
            f"Scheme {s_idx} Main Wall Colour Check: requested={requested_hex}, "
            f"sampled_median={sampled_hex}, deltaE={delta_e:.2f}"
        )

        if delta_e > 12.0:
            logger.warning(
                f"Scheme {s_idx} color deviation warning: deltaE {delta_e:.2f} > 12.0 "
                f"between requested {requested_hex} and generated main wall {sampled_hex}"
            )

        return delta_e, sampled_hex

    async def _generate_single_with_retry(
        self,
        client,
        img_bytes: bytes,
        mime_type: str,
        prompt: str,
        s_idx: int,
        timeout: float
    ) -> bytes:
        """Runs single generation with timeout and 1 retry."""
        max_attempts = 2
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    client.generate_scheme_image(img_bytes, mime_type, prompt),
                    timeout=timeout
                )
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"Scheme {s_idx} generation attempt {attempt}/{max_attempts} failed: {e}"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(1.0)
        raise last_exc or RuntimeError(f"Scheme {s_idx} generation failed after {max_attempts} attempts")

    async def render_single_scheme(
        self,
        s_idx: int,
        scheme_item: SchemeSelectionItem,
        all_paints: Dict[int, Paint],
        original_bgr: np.ndarray,
        cropped_image_bytes: bytes,
        zone_masks: Dict[str, np.ndarray],
        building_mask: np.ndarray,
        client_ip: str,
        session_dir: Path,
        layer_contexts: Optional[List[Dict]] = None
    ) -> SchemeResult:
        """
        Renders a single scheme using the AI image editing model, with caching,
        cost controls, compositing, color check, and graceful fallback.
        """
        mp = all_paints.get(scheme_item.main_id)
        ap = all_paints.get(scheme_item.accent_id)
        tp = all_paints.get(scheme_item.trim_gate_id)

        if not (mp and ap and tp):
            raise ValueError(f"Scheme {s_idx} missing catalog paint models in database")

        shades_info = [
            SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
            SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
            SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
        ]

        model_name = settings.IMAGE_MODEL or "gemini-2.5-flash-image"

        # Instantiate client with error handling for production mode
        try:
            client = get_image_client()
        except ValueError as ve:
            logger.warning(f"Image client creation rejected: {ve}. Using precise mode fallback.")
            return self._fallback_precise_scheme(
                s_idx, scheme_item, shades_info, original_bgr, zone_masks, layer_contexts, session_dir,
                fallback_reason=str(ve)
            )

        is_mock_client = getattr(client, "is_mock", False)

        # 1. Check cache (only for real API clients, never cache mocks)
        if not is_mock_client:
            cache_key = scheme_cache.compute_cache_key(
                cropped_image_bytes, mp.id, ap.id, tp.id, model_name
            )
            cached_hit = scheme_cache.get(cache_key)
            if cached_hit is not None:
                comp_bgr, meta = cached_hit
                scheme_fname = f"scheme_{s_idx}.jpg"
                cv2.imwrite(str(session_dir / scheme_fname), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

                return SchemeResult(
                    scheme_index=s_idx,
                    reason=scheme_item.reason,
                    shades=shades_info,
                    image_url=comp_url,
                    status="success",
                    mode="ai",
                    mock=False,
                    delta_e=meta.get("delta_e"),
                    drift_outside_pct=meta.get("drift_outside_pct")
                )

            # 2. Check cost controls & daily limits
            allowed, reason_msg = cost_tracker.check_limits(client_ip)
            if not allowed:
                logger.warning(f"Cost limit hit for Scheme {s_idx}: {reason_msg}. Falling back to precise mode.")
                return self._fallback_precise_scheme(
                    s_idx, scheme_item, shades_info, original_bgr, zone_masks, layer_contexts, session_dir,
                    fallback_reason=reason_msg
                )

        # 3. Build Prompt
        prompt = self.build_prompt(mp, ap, tp)

        # 4. Generate Image
        raw_gen_bytes = await self._generate_single_with_retry(
            client=client,
            img_bytes=cropped_image_bytes,
            mime_type="image/jpeg",
            prompt=prompt,
            s_idx=s_idx,
            timeout=settings.IMAGE_TIMEOUT_SECONDS
        )

        nparr = np.frombuffer(raw_gen_bytes, np.uint8)
        gen_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if gen_bgr is None:
            raise ValueError(f"Could not decode generated image bytes for Scheme {s_idx}")

        if is_mock_client:
            # Mock mode: client returns original photo with "MOCK - NO IMAGE API KEY" banner
            # Do NOT composite fake rectangles; do NOT compute deltaE or drift metrics
            composite_bgr = gen_bgr
            delta_e = None
            pct_changed = None
            logger.info(f"Scheme {s_idx}: rendered via MockImageClient (mock=True). deltaE and drift skipped.")
        else:
            # 5. Composite with SegFormer building mask
            composite_bgr, mean_delta, pct_changed = self.composite_with_mask(
                original_bgr=original_bgr,
                generated_bgr=gen_bgr,
                building_mask=building_mask,
                s_idx=s_idx
            )

            # 6. Color Check (deltaE vs requested hex)
            delta_e, sampled_hex = self.check_main_wall_color(
                composite_bgr=composite_bgr,
                main_mask=zone_masks.get("main", building_mask),
                requested_hex=mp.hex,
                s_idx=s_idx
            )

            # 7. Record cost & save cache
            cost_tracker.record_generation(client_ip, model_name, count=1)
            scheme_cache.set(cache_key, composite_bgr, {
                "delta_e": round(delta_e, 2),
                "drift_outside_pct": round(pct_changed, 2),
                "sampled_hex": sampled_hex,
                "requested_hex": mp.hex
            })

        # Save to session disk
        scheme_fname = f"scheme_{s_idx}.jpg"
        cv2.imwrite(str(session_dir / scheme_fname), composite_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        _, enc_comp = cv2.imencode(".jpg", composite_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

        return SchemeResult(
            scheme_index=s_idx,
            reason=scheme_item.reason,
            shades=shades_info,
            image_url=comp_url,
            status="success",
            mode="ai",
            mock=is_mock_client,
            delta_e=round(delta_e, 2) if delta_e is not None else None,
            drift_outside_pct=round(pct_changed, 2) if pct_changed is not None else None
        )

    def _fallback_precise_scheme(
        self,
        s_idx: int,
        scheme_item: SchemeSelectionItem,
        shades_info: List[SchemeShadeInfo],
        original_bgr: np.ndarray,
        zone_masks: Dict[str, np.ndarray],
        layer_contexts: Optional[List[Dict]],
        session_dir: Path,
        fallback_reason: str = "Fallback to precise recolor mode"
    ) -> SchemeResult:
        """Falls back to algorithmic mask recoloring ('precise mode')."""
        if layer_contexts is None:
            layer_contexts = self.recolor_service.prepare_multi_layer_context(
                image_bgr=original_bgr,
                masks=zone_masks,
                surface_type="exterior"
            )

        mp_hex = next(s.hex for s in shades_info if s.role == "main")
        ap_hex = next(s.hex for s in shades_info if s.role == "accent")
        tp_hex = next(s.hex for s in shades_info if s.role == "trim_gate")

        scheme_hexes = {"main": mp_hex, "accent": ap_hex, "trim_gate": tp_hex}
        comp_bgr = self.recolor_service.recolor_multi_layer(original_bgr, layer_contexts, scheme_hexes)

        scheme_fname = f"scheme_{s_idx}.jpg"
        cv2.imwrite(str(session_dir / scheme_fname), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

        return SchemeResult(
            scheme_index=s_idx,
            reason=scheme_item.reason,
            shades=shades_info,
            image_url=comp_url,
            status="fallback",
            mode="precise",
            mock=False,
            delta_e=None,
            drift_outside_pct=None,
            error_message=fallback_reason
        )

    async def render_schemes_parallel(
        self,
        original_bgr: np.ndarray,
        cropped_image_bytes: bytes,
        zone_masks: Dict[str, np.ndarray],
        selected_schemes: List[SchemeSelectionItem],
        all_paints: Dict[int, Paint],
        client_ip: str,
        session_dir: Path
    ) -> Tuple[List[SchemeResult], str, Optional[str]]:
        """
        Executes parallel generation of 4 exterior schemes.
        - If all succeed: returns 4 AI schemes with mode="ai".
        - If some fail: returns successful AI schemes, plus failed ones with 'Regenerate' option.
        - If all fail: cleanly falls back to precise mask recolor mode with message.
        """
        # Combined building facade mask
        main_m = zone_masks.get("main", np.zeros(original_bgr.shape[:2], dtype=np.uint8))
        accent_m = zone_masks.get("accent", np.zeros(original_bgr.shape[:2], dtype=np.uint8))
        trim_m = zone_masks.get("trim_gate", np.zeros(original_bgr.shape[:2], dtype=np.uint8))
        building_mask = (main_m | accent_m | trim_m)

        # Precompute layer contexts for fallback
        layer_contexts = self.recolor_service.prepare_multi_layer_context(
            image_bgr=original_bgr,
            masks=zone_masks,
            surface_type="exterior"
        )

        tasks = []
        for s_idx, scheme_item in enumerate(selected_schemes, 1):
            tasks.append(
                self.render_single_scheme(
                    s_idx=s_idx,
                    scheme_item=scheme_item,
                    all_paints=all_paints,
                    original_bgr=original_bgr,
                    cropped_image_bytes=cropped_image_bytes,
                    zone_masks=zone_masks,
                    building_mask=building_mask,
                    client_ip=client_ip,
                    session_dir=session_dir,
                    layer_contexts=layer_contexts
                )
            )

        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        final_schemes: List[SchemeResult] = []
        success_count = 0

        for s_idx, res in enumerate(results_or_exceptions, 1):
            scheme_item = selected_schemes[s_idx - 1]
            mp = all_paints.get(scheme_item.main_id)
            ap = all_paints.get(scheme_item.accent_id)
            tp = all_paints.get(scheme_item.trim_gate_id)

            shades_info = [
                SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
                SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
                SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
            ]

            if isinstance(res, Exception):
                logger.error(f"Scheme {s_idx} failed generation: {res}")
                # Provide fallback precise preview so UI can render with Regenerate button
                fallback_res = self._fallback_precise_scheme(
                    s_idx=s_idx,
                    scheme_item=scheme_item,
                    shades_info=shades_info,
                    original_bgr=original_bgr,
                    zone_masks=zone_masks,
                    layer_contexts=layer_contexts,
                    session_dir=session_dir,
                    fallback_reason=str(res)
                )
                fallback_res.status = "failed"
                fallback_res.error_message = f"Generation failed: {res}. Click Regenerate to retry."
                final_schemes.append(fallback_res)
            else:
                if res.status == "success":
                    success_count += 1
                final_schemes.append(res)

        if success_count == 0:
            logger.warning("All 4 AI image generations failed. Falling back completely to precise mode.")
            return final_schemes, "precise", "AI image generation unavailable, fell back to precise recolor mode."
        elif success_count < len(selected_schemes):
            logger.info(f"{success_count}/{len(selected_schemes)} AI image generations succeeded. Partial failure handling active.")
            return final_schemes, "ai", f"{len(selected_schemes) - success_count} scheme(s) had generation issues; click Regenerate to retry."
        else:
            logger.info("All 4 AI exterior schemes generated successfully.")
            return final_schemes, "ai", None
