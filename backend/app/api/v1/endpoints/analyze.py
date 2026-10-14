import uuid
import json
import base64
from pathlib import Path
from typing import List
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.rate_limiter import rate_limiter
from app.schemas.recolor import AnalysisAndRecolorResponse, RepaintedShadeResult
from app.services.vision_service import VisionService
from app.services.shortlist_service import ShortlistService
from app.services.selection_service import SelectionService
from app.services.segmentation_service import SegmentationService
from app.services.recolor_service import RecolorService

router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@router.post("/analyze", response_model=AnalysisAndRecolorResponse)
async def analyze_and_recolor(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Rate limiting
    rate_limiter.check(request)

    # 2. Validate MIME type
    content_type = file.content_type.lower() if file.content_type else ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type '{content_type}'. Allowed: JPEG, PNG, WEBP."
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 10 MB limit."
        )

    # Verify valid image decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format; could not decode."
        )

    # User Change 9: Detect and crop camera info bar from phone photos
    from app.services.image_utils import detect_and_crop_camera_bar
    img_bgr, cropped_rows = detect_and_crop_camera_bar(img_bgr)
    if cropped_rows > 0:
        _, encoded_cropped = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        image_bytes = encoded_cropped.tobytes()

    # Ensure working resolution is capped at 2000px max dimension
    max_dim = 2000
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, encoded_scaled = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        image_bytes = encoded_scaled.tobytes()

    session_id = str(uuid.uuid4())
    session_dir = settings.UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save original image
    orig_filename = "original.jpg"
    orig_path = session_dir / orig_filename
    cv2.imwrite(str(orig_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3. Vision Analysis (downscale copy for vision to save bandwidth & memory, keeping original for recolor)
    vision_service = VisionService()
    h_curr, w_curr = img_bgr.shape[:2]
    if max(h_curr, w_curr) > 768:
        scale_v = 768 / max(h_curr, w_curr)
        img_vision = cv2.resize(img_bgr, (int(w_curr * scale_v), int(h_curr * scale_v)), interpolation=cv2.INTER_AREA)
        _, encoded_vision = cv2.imencode(".jpg", img_vision, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        vision_bytes = encoded_vision.tobytes()
    else:
        vision_bytes = image_bytes
    room_analysis = await vision_service.analyze_room(vision_bytes, mime_type="image/jpeg")

    # 4. Shortlist (done in CODE in LAB/HSL space, with lighting adjustment)
    shortlist_service = ShortlistService()
    shortlist = shortlist_service.shortlist_shades(db, room_analysis, min_count=20, max_count=40)

    # 5. Wall Detection
    segmentation_service = SegmentationService()
    wall_mask, _ = segmentation_service.detect_walls(
        image_bytes,
        surface_type=room_analysis.scene_type
    )

    # Save session mask (backward compat)
    mask_filename = "mask.png"
    mask_path = session_dir / mask_filename
    cv2.imwrite(str(mask_path), wall_mask)

    from app.services.zone_mask_service import ZoneMaskService
    from app.services.scheme_service import SchemeService
    from app.schemas.recolor import SchemeResult, SchemeShadeInfo
    from app.models.paint import Paint

    zone_service = ZoneMaskService()
    recolor_service = RecolorService()

    # Initialize 3 layer masks
    zone_masks = zone_service.init_session_masks(session_dir, wall_mask)
    coverage_pct = zone_service.get_coverage_pct(zone_masks)

    # Encode mask data URLs
    mask_urls = {}
    for layer_name, m in zone_masks.items():
        _, enc_m = cv2.imencode(".png", m)
        mask_urls[layer_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"

    results: List[RepaintedShadeResult] = []
    schemes_results: Optional[List[SchemeResult]] = None
    meta_payload = {
        "session_id": session_id,
        "room_analysis": room_analysis.model_dump(),
        "scene_type": room_analysis.scene_type,
        "coverage_pct": coverage_pct
    }

    mode: str = "precise"
    fallback_reason: Optional[str] = None

    if room_analysis.scene_type == "exterior":
        # EXTERIOR FLOW: Mask-based Multi-Layer Recolor (Primary Mode, No Image API)
        scheme_service = SchemeService()
        selected_schemes = await scheme_service.select_schemes(db, room_analysis, shortlist)

        recolor_service = RecolorService()
        zone_svc = ZoneMaskService()
        all_paints = {p.id: p for p in db.query(Paint).all()}

        from app.services.protection_service import ProtectionService
        prot_svc = ProtectionService()
        protected_mask, sky_mask, hatched_overlay = prot_svc.compute_protection_masks(img_bgr)
        zone_svc.save_protection_artifacts(session_dir, protected_mask, sky_mask, hatched_overlay)

        # Initialize session masks with hard auto-protection constraint (sky, ground, veg, vehicles subtracted)
        zone_masks = zone_svc.init_session_masks(session_dir, wall_mask, protected_mask=protected_mask)
        coverage_by_layer = zone_svc.get_coverage_pct(zone_masks)
        sky_fraction_by_layer = zone_svc.get_sky_fractions(zone_masks, sky_mask)

        # Encode masks & protection artifacts as data URLs
        mask_urls = {}
        for layer_name, m in zone_masks.items():
            _, enc_m = cv2.imencode(".png", m)
            mask_urls[layer_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"

        _, enc_hatch = cv2.imencode(".png", hatched_overlay)
        protected_hatched_url = f"data:image/png;base64,{base64.b64encode(enc_hatch).decode('utf-8')}"
        _, enc_prot = cv2.imencode(".png", protected_mask)
        protected_mask_url = f"data:image/png;base64,{base64.b64encode(enc_prot).decode('utf-8')}"

        combined_facade_mask = zone_svc.get_combined_mask(zone_masks)

        # Precompute multi-layer shading contexts once (cached across all 4 schemes)
        layer_contexts = recolor_service.prepare_multi_layer_context(
            image_bgr=img_bgr,
            masks=zone_masks,
            surface_type="exterior"
        )

        schemes_results = []
        schemes_meta = []
        for s_idx, s in enumerate(selected_schemes, start=1):
            mp = all_paints.get(s.main_id)
            ap = all_paints.get(s.accent_id)
            tp = all_paints.get(s.trim_gate_id)
            if not (mp and ap and tp):
                continue

            scheme_hexes = {"main": mp.hex, "accent": ap.hex, "trim_gate": tp.hex}
            comp_bgr = recolor_service.recolor_multi_layer(
                img_bgr, layer_contexts, scheme_hexes, facade_mask=combined_facade_mask
            )

            # Save scheme output
            out_path = session_dir / f"scheme_{s_idx}.jpg"
            cv2.imwrite(str(out_path), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

            shades_info = [
                SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
                SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
                SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
            ]
            scheme_res = SchemeResult(
                scheme_index=s_idx,
                reason=s.reason,
                shades=shades_info,
                image_url=comp_url,
                mode="precise",
                mock=False,
                status="success"
            )
            schemes_results.append(scheme_res)

            # Backward compatibility for clients expecting results array
            results.append(RepaintedShadeResult(
                shade_id=mp.id,
                shade_name=mp.name,
                shade_code=mp.shade_code,
                hex=mp.hex,
                finish=mp.finish,
                color_family=mp.color_family,
                reason=s.reason,
                image_url=comp_url,
                price=None
            ))

            schemes_meta.append({
                "scheme_index": s_idx,
                "reason": s.reason,
                "main_id": mp.id,
                "accent_id": ap.id,
                "trim_gate_id": tp.id,
                "status": "success",
                "mode": "precise",
                "delta_e": None,
                "drift_outside_pct": None,
                "error_message": None
            })

        mode = "precise"
        fallback_reason = None
        meta_payload["schemes"] = schemes_meta
        meta_payload["mode"] = "precise"
        meta_payload["mock"] = False

    else:
        # INTERIOR FLOW: Single-Shade Recommendations (backward compatible)
        selection_service = SelectionService()
        selected_shades = await selection_service.select_shades(db, room_analysis, shortlist)

        shading_ctx = recolor_service.prepare_shading_context(
            image_bgr=img_bgr,
            wall_mask=wall_mask,
            surface_type=room_analysis.scene_type,
            feather_radius=5
        )

        shades_meta = []
        for idx, (paint, reason) in enumerate(selected_shades, 1):
            repainted_bgr = recolor_service.recolor_with_context(
                context=shading_ctx,
                target_hex=paint.hex
            )
            shade_filename = f"shade_{idx}.jpg"
            shade_path = session_dir / shade_filename
            cv2.imwrite(str(shade_path), repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_shade = cv2.imencode(".jpg", repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            shade_b64 = base64.b64encode(enc_shade).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{shade_b64}"

            results.append(RepaintedShadeResult(
                shade_id=paint.id,
                shade_name=paint.name,
                shade_code=paint.shade_code,
                hex=paint.hex,
                finish=paint.finish,
                color_family=paint.color_family,
                reason=reason,
                image_url=image_url,
                price=paint.price
            ))

            shades_meta.append({
                "shade_id": paint.id,
                "shade_name": paint.name,
                "shade_code": paint.shade_code,
                "hex": paint.hex,
                "finish": paint.finish,
                "color_family": paint.color_family,
                "reason": reason,
                "price": paint.price
            })

        meta_payload["shades"] = shades_meta

    # Save session metadata
    meta_path = session_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    # Encode original image and mask as data URLs
    _, enc_orig = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    orig_b64 = base64.b64encode(enc_orig).decode("utf-8")
    original_data_url = f"data:image/jpeg;base64,{orig_b64}"

    mask_data_url = mask_urls["main"]

    import gc
    gc.collect()

    is_mock = any(bool(s.mock) for s in schemes_results) if schemes_results else False

    return AnalysisAndRecolorResponse(
        session_id=session_id,
        original_image_url=original_data_url,
        mask_url=mask_data_url,
        masks=mask_urls,
        coverage_pct=coverage_pct,
        coverage_by_layer=coverage_by_layer if "coverage_by_layer" in locals() else None,
        sky_fraction_by_layer=sky_fraction_by_layer if "sky_fraction_by_layer" in locals() else None,
        protected_mask_url=protected_mask_url if "protected_mask_url" in locals() else None,
        protected_hatched_url=protected_hatched_url if "protected_hatched_url" in locals() else None,
        room_analysis=room_analysis,
        results=results,
        schemes=schemes_results,
        mode=mode,
        mock=is_mock,
        fallback_reason=fallback_reason
    )
