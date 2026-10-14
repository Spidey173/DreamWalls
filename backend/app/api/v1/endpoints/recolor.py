import json
import base64
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
from pydantic import BaseModel

from app.core.config import settings
from app.schemas.recolor import RepaintedShadeResult, RegenerateSchemePayload, SchemeResult
from app.services.recolor_service import RecolorService

router = APIRouter()

class TapPoint(BaseModel):
    x: float  # Normalized 0.0 - 1.0
    y: float  # Normalized 0.0 - 1.0
    label: int = 1  # 1 = positive (foreground), 0 = negative (background)

class TapSegmentPayload(BaseModel):
    session_id: str
    layer: str = "main"  # "main" | "accent" | "trim_gate"
    points: List[TapPoint]

class ResetZonesPayload(BaseModel):
    session_id: str
    layer: Optional[str] = None  # None = reset all, or "main" | "accent" | "trim_gate"

class RecolorPayload(BaseModel):
    session_id: str
    layer: Optional[str] = "main"
    custom_mask_base64: Optional[str] = None
    custom_shades: Optional[dict] = None  # e.g. {"main": 12, "accent": "L102", "trim_gate": "#1F2937"}
    scheme_index: Optional[int] = 1
    masks: Optional[dict] = None

class SelectZonePayload(BaseModel):
    session_id: str
    layer: str = "main"  # "main" | "accent" | "trim_gate"
    tool: str = "magic_wand"  # "magic_wand" | "polygon" | "rectangle" | "brush" | "sam" | "mask" | "auto_region"
    tolerance: float = 12.0
    operation: str = "add"  # "add" | "subtract" | "replace"
    point: Optional[TapPoint] = None
    points: Optional[List[TapPoint]] = None
    box: Optional[dict] = None  # {"x1": float, "y1": float, "x2": float, "y2": float}
    brush_radius: int = 25
    custom_mask_base64: Optional[str] = None
    custom_shades: Optional[dict] = None
    scheme_index: Optional[int] = 1

@router.get("/catalog-shades")
def get_catalog_shades(
    color_family: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 150
):
    """
    Public catalog endpoint allowing users to browse and pick Asian Paints shades
    for building zones (Main, Accents, Trims).
    """
    from app.models.paint import Paint
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        query = db.query(Paint)
        if color_family and color_family != "All":
            query = query.filter(Paint.color_family == color_family)
        if search:
            s_term = f"%{search.strip()}%"
            query = query.filter((Paint.name.ilike(s_term)) | (Paint.shade_code.ilike(s_term)))
        paints = query.limit(limit).all()
        return [p.to_dict() for p in paints]
    finally:
        db.close()

@router.post("/reset-zones")
async def reset_zones(payload: ResetZonesPayload):
    """
    Resets zone masks:
    - If layer is specified, clears that zone layer to empty.
    - If layer is None or 'all', restores 'main' from initial wall mask and empties others.
    Re-renders the 4 exterior schemes instantly with updated masks.
    """
    session_dir = settings.UPLOAD_DIR / payload.session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{payload.session_id}' not found."
        )

    orig_path = session_dir / "original.jpg"
    meta_path = session_dir / "meta.json"
    if not orig_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original image or session metadata missing."
        )

    orig_bgr = cv2.imread(str(orig_path))
    from app.services.zone_mask_service import ZoneMaskService
    from app.services.recolor_service import RecolorService
    from app.schemas.recolor import SchemeResult, SchemeShadeInfo
    from app.models.paint import Paint
    from app.core.database import SessionLocal

    zone_svc = ZoneMaskService()
    recolor_service = RecolorService()

    prot_mask, sky_mask, hatch_overlay = zone_svc.load_protection_artifacts(session_dir)
    if prot_mask is None or sky_mask is None:
        from app.services.protection_service import ProtectionService
        prot_svc = ProtectionService()
        prot_mask, sky_mask, hatch_overlay = prot_svc.compute_protection_masks(orig_bgr)
        zone_svc.save_protection_artifacts(session_dir, prot_mask, sky_mask, hatch_overlay)

    if payload.layer:
        zone_masks = zone_svc.load_masks(session_dir, fallback_shape=orig_bgr.shape[:2])
        zone_masks = zone_svc.reset_layer(zone_masks, payload.layer, protected_mask=prot_mask)
        zone_svc.save_masks(session_dir, zone_masks)
    else:
        zone_masks = zone_svc.reset_all(session_dir, fallback_shape=orig_bgr.shape[:2], protected_mask=prot_mask)

    combined_mask = zone_svc.get_combined_mask(zone_masks)
    cv2.imwrite(str(session_dir / "mask.png"), combined_mask)
    coverage_pct = zone_svc.get_coverage_pct(zone_masks)
    sky_fraction_by_layer = zone_svc.get_sky_fractions(zone_masks, sky_mask)

    mask_urls = {}
    for l_name, m in zone_masks.items():
        _, enc_m = cv2.imencode(".png", m)
        mask_urls[l_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"
    _, enc_comb = cv2.imencode(".png", combined_mask)
    comb_data_url = f"data:image/png;base64,{base64.b64encode(enc_comb).decode('utf-8')}"

    _, enc_hatch = cv2.imencode(".png", hatch_overlay)
    protected_hatched_url = f"data:image/png;base64,{base64.b64encode(enc_hatch).decode('utf-8')}"
    _, enc_prot = cv2.imencode(".png", prot_mask)
    protected_mask_url = f"data:image/png;base64,{base64.b64encode(enc_prot).decode('utf-8')}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    schemes_results: List[SchemeResult] = []
    legacy_results = []
    if "schemes" in meta_data and meta_data["schemes"]:
        db = SessionLocal()
        all_paints = {p.id: p for p in db.query(Paint).all()}
        db.close()

        layer_contexts = recolor_service.prepare_multi_layer_context(
            image_bgr=orig_bgr,
            masks=zone_masks,
            surface_type="exterior"
        )

        for s_meta in meta_data["schemes"]:
            s_idx = s_meta["scheme_index"]
            mp = all_paints.get(s_meta["main_id"])
            ap = all_paints.get(s_meta["accent_id"])
            tp = all_paints.get(s_meta["trim_gate_id"])
            if not (mp and ap and tp):
                continue

            scheme_hexes = {"main": mp.hex, "accent": ap.hex, "trim_gate": tp.hex}
            comp_bgr = recolor_service.recolor_multi_layer(
                orig_bgr, layer_contexts, scheme_hexes, facade_mask=combined_mask
            )
            cv2.imwrite(str(session_dir / f"scheme_{s_idx}.jpg"), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

            shades_info = [
                SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
                SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
                SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
            ]
            schemes_results.append(SchemeResult(
                scheme_index=s_idx,
                reason=s_meta["reason"],
                shades=shades_info,
                image_url=comp_url,
                mode="precise",
                mock=False,
                status="success"
            ))

        for s in schemes_results:
            main_sh = next((sh for sh in s.shades if sh.role == "main"), s.shades[0])
            legacy_results.append({
                "shade_id": main_sh.shade_id,
                "shade_name": main_sh.shade_name,
                "shade_code": main_sh.shade_code,
                "hex": main_sh.hex,
                "finish": main_sh.finish,
                "color_family": main_sh.color_family,
                "reason": s.reason,
                "image_url": s.image_url,
                "price": None
            })

    return {
        "session_id": payload.session_id,
        "mask_url": comb_data_url,
        "masks": mask_urls,
        "coverage_by_layer": coverage_pct,
        "coverage_pct": round(float(np.mean(combined_mask > 128) * 100), 2),
        "sky_fraction_by_layer": sky_fraction_by_layer,
        "protected_mask_url": protected_mask_url,
        "protected_hatched_url": protected_hatched_url,
        "schemes": [s.model_dump() for s in schemes_results],
        "results": legacy_results if "schemes" in meta_data else []
    }

@router.post("/select-zone")
async def select_zone(payload: SelectZonePayload):
    """
    Unified architectural selection endpoint:
    - Supports magic_wand (default), polygon, rectangle, brush, sam, custom_mask.
    - Tolerance slider for magic wand flood-fill.
    - Operation: add, subtract, replace.
    - Auto-protection: strictly subtracts sky, road/ground, vegetation, and vehicles.
    - Post-selection: fills holes, removes small blobs (< 0.1% area), smoothes edges lightly.
    - Last-edited layer wins (non-overlapping).
    - Verifies sky_fraction < 1.0% in all zones.
    - Re-renders schemes instantly with cached shading contexts and byte-identical unselected pixels.
    """
    session_dir = settings.UPLOAD_DIR / payload.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{payload.session_id}' not found.")

    orig_path = session_dir / "original.jpg"
    meta_path = session_dir / "meta.json"
    if not orig_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=400, detail="Original image or metadata missing.")

    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to load original image.")

    from app.services.zone_mask_service import ZoneMaskService
    from app.services.selection_service import SelectionService
    from app.services.protection_service import ProtectionService
    from app.services.recolor_service import RecolorService
    from app.schemas.recolor import SchemeResult, SchemeShadeInfo
    from app.models.paint import Paint
    from app.core.database import SessionLocal

    zone_svc = ZoneMaskService()
    sel_svc = SelectionService()
    recolor_service = RecolorService()

    # 1. Load or compute protection artifacts
    prot_mask, sky_mask, hatch_overlay = zone_svc.load_protection_artifacts(session_dir)
    if prot_mask is None or sky_mask is None or hatch_overlay is None:
        prot_svc = ProtectionService()
        prot_mask, sky_mask, hatch_overlay = prot_svc.compute_protection_masks(orig_bgr)
        zone_svc.save_protection_artifacts(session_dir, prot_mask, sky_mask, hatch_overlay)

    # 2. Load existing zone masks
    zone_masks = zone_svc.load_masks(session_dir, fallback_shape=orig_bgr.shape[:2])
    target_layer = payload.layer if payload.layer in zone_svc.LAYERS else "main"
    active_mask = zone_masks[target_layer]

    # 3. Apply tool
    op = payload.operation if payload.operation in ["add", "subtract", "replace"] else "add"
    auto_region_count: int = 0
    selected_region_id: Optional[int] = None
    auto_region_compute_time_ms: float = 0.0

    if payload.tool == "magic_wand":
        pt = payload.point.model_dump() if payload.point else {"x": 0.5, "y": 0.5}
        new_layer_mask = sel_svc.magic_wand(
            image_bgr=orig_bgr,
            seed_point=pt,
            tolerance=payload.tolerance,
            protected_mask=prot_mask,
            active_mask=active_mask,
            operation=op
        )
    elif payload.tool == "polygon":
        pts = [p.model_dump() for p in (payload.points or [])]
        new_layer_mask = sel_svc.polygon(
            image_shape=orig_bgr.shape[:2],
            points=pts,
            protected_mask=prot_mask,
            active_mask=active_mask,
            operation=op
        )
    elif payload.tool == "rectangle":
        box = payload.box or {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        new_layer_mask = sel_svc.rectangle(
            image_shape=orig_bgr.shape[:2],
            box=box,
            protected_mask=prot_mask,
            active_mask=active_mask,
            operation=op
        )
    elif payload.tool == "brush":
        pts = [p.model_dump() for p in (payload.points or [])]
        new_layer_mask = sel_svc.brush(
            image_shape=orig_bgr.shape[:2],
            points=pts,
            radius=payload.brush_radius,
            protected_mask=prot_mask,
            active_mask=active_mask,
            operation=op if op in ["add", "subtract"] else "add"
        )
    elif payload.tool == "sam":
        pts = [p.model_dump() for p in (payload.points or [])]
        if not pts and payload.point:
            pt = payload.point.model_dump()
            if "label" not in pt:
                pt["label"] = 1.0
            pts = [pt]
        new_layer_mask = sel_svc.sam_selection(
            image_bgr=orig_bgr,
            cache_key=payload.session_id,
            points=pts,
            protected_mask=prot_mask,
            active_mask=active_mask,
            operation=op
        )
    elif payload.tool == "auto_region":
        from app.services.auto_regions_service import AutoRegionsService
        ar_svc = AutoRegionsService()
        candidates, auto_region_compute_time_ms = ar_svc.load_or_compute(
            session_dir=session_dir,
            image_bgr=orig_bgr,
            cache_key=payload.session_id,
            protected_mask=prot_mask
        )
        auto_region_count = len(candidates)
        pt = payload.point.model_dump() if payload.point else {"x": 0.5, "y": 0.5}
        cand = ar_svc.find_region_at_tap(candidates, pt["x"], pt["y"], orig_bgr.shape[:2])
        if cand is not None:
            selected_region_id = cand.region_id
            new_layer_mask = sel_svc._apply_operation(
                active_mask, cand.mask, op, orig_bgr.shape[:2]
            )
        else:
            selected_region_id = None
            new_layer_mask = active_mask
    elif payload.custom_mask_base64:
        raw_b64 = payload.custom_mask_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        mask_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(mask_bytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if decoded is not None:
            if decoded.shape[:2] != orig_bgr.shape[:2]:
                decoded = cv2.resize(decoded, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            cleaned = ProtectionService.clean_selection_mask(decoded, protected_mask=prot_mask)
            new_layer_mask = sel_svc._apply_operation(active_mask, cleaned, op, orig_bgr.shape[:2])
        else:
            new_layer_mask = active_mask
    else:
        new_layer_mask = active_mask

    # 4. Update layer with 'last-edited layer wins' and zero-overlap constraint
    zone_masks = zone_svc.update_layer(zone_masks, target_layer, new_layer_mask, protected_mask=prot_mask)
    zone_svc.save_masks(session_dir, zone_masks)

    combined_mask = zone_svc.get_combined_mask(zone_masks)
    cv2.imwrite(str(session_dir / "mask.png"), combined_mask)
    coverage_by_layer = zone_svc.get_coverage_pct(zone_masks)
    sky_fraction_by_layer = zone_svc.get_sky_fractions(zone_masks, sky_mask)

    # 5. Encode masks & protection artifacts
    mask_urls = {}
    for l_name, m in zone_masks.items():
        _, enc_m = cv2.imencode(".png", m)
        mask_urls[l_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"
    _, enc_comb = cv2.imencode(".png", combined_mask)
    comb_data_url = f"data:image/png;base64,{base64.b64encode(enc_comb).decode('utf-8')}"

    _, enc_hatch = cv2.imencode(".png", hatch_overlay)
    protected_hatched_url = f"data:image/png;base64,{base64.b64encode(enc_hatch).decode('utf-8')}"
    _, enc_prot = cv2.imencode(".png", prot_mask)
    protected_mask_url = f"data:image/png;base64,{base64.b64encode(enc_prot).decode('utf-8')}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 6. Re-render schemes instantly with cached shading contexts
    schemes_results: List[SchemeResult] = []
    legacy_results = []
    if "schemes" in meta_data and meta_data["schemes"]:
        db = SessionLocal()
        all_paints = {p.id: p for p in db.query(Paint).all()}
        db.close()

        # If custom shades provided, update metadata
        if payload.custom_shades:
            target_s_idx = payload.scheme_index or 1
            all_by_code = {p.shade_code: p for p in all_paints.values()}
            for s_meta in meta_data["schemes"]:
                if s_meta.get("scheme_index") == target_s_idx:
                    for role_key in ["main", "accent", "trim_gate"]:
                        if role_key in payload.custom_shades:
                            val = payload.custom_shades[role_key]
                            matched_p = None
                            if isinstance(val, int) and val in all_paints:
                                matched_p = all_paints[val]
                            elif isinstance(val, str):
                                if val in all_by_code:
                                    matched_p = all_by_code[val]
                                else:
                                    for p in all_paints.values():
                                        if p.shade_code.lower() == val.lower() or p.hex.lower() == val.lower():
                                            matched_p = p
                                            break
                            if matched_p:
                                s_meta[f"{role_key}_id"] = matched_p.id
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

        layer_contexts = recolor_service.prepare_multi_layer_context(
            image_bgr=orig_bgr,
            masks=zone_masks,
            surface_type="exterior"
        )

        for s_meta in meta_data["schemes"]:
            s_idx = s_meta["scheme_index"]
            mp = all_paints.get(s_meta["main_id"])
            ap = all_paints.get(s_meta["accent_id"])
            tp = all_paints.get(s_meta["trim_gate_id"])
            if not (mp and ap and tp):
                continue

            scheme_hexes = {"main": mp.hex, "accent": ap.hex, "trim_gate": tp.hex}
            comp_bgr = recolor_service.recolor_multi_layer(
                orig_bgr, layer_contexts, scheme_hexes, facade_mask=combined_mask
            )
            cv2.imwrite(str(session_dir / f"scheme_{s_idx}.jpg"), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

            shades_info = [
                SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
                SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
                SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
            ]
            schemes_results.append(SchemeResult(
                scheme_index=s_idx,
                reason=s_meta["reason"],
                shades=shades_info,
                image_url=comp_url,
                mode="precise",
                mock=False,
                status="success"
            ))

        for s in schemes_results:
            main_sh = next((sh for sh in s.shades if sh.role == "main"), s.shades[0])
            legacy_results.append({
                "shade_id": main_sh.shade_id,
                "shade_name": main_sh.shade_name,
                "shade_code": main_sh.shade_code,
                "hex": main_sh.hex,
                "finish": main_sh.finish,
                "color_family": main_sh.color_family,
                "reason": s.reason,
                "image_url": s.image_url,
                "price": None
            })

    return {
        "session_id": payload.session_id,
        "active_layer": target_layer,
        "mask_url": comb_data_url,
        "masks": mask_urls,
        "coverage_by_layer": coverage_by_layer,
        "coverage_pct": round(float(np.mean(combined_mask > 128) * 100), 2),
        "sky_fraction_by_layer": sky_fraction_by_layer,
        "protected_mask_url": protected_mask_url,
        "protected_hatched_url": protected_hatched_url,
        "auto_region_count": auto_region_count,
        "selected_region_id": selected_region_id,
        "auto_region_compute_time_ms": round(auto_region_compute_time_ms, 2),
        "schemes": [s.model_dump() for s in schemes_results],
        "results": legacy_results if "schemes" in meta_data else []
    }


@router.post("/tap-segment")
async def tap_segment(payload: TapSegmentPayload):
    """
    Interactive tap-to-select zone segmentation using MobileSAM ONNX:
    - User taps on the active layer ('main', 'accent', or 'trim_gate')
    - Active layer takes pixels from other layers ('last-edited layer wins')
    - Returns all 3 masks and coverage percentages
    """
    session_dir = settings.UPLOAD_DIR / payload.session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{payload.session_id}' not found."
        )

    orig_path = session_dir / "original.jpg"
    if not orig_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original image missing from session directory."
        )

    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load original image."
        )

    from app.services.sam_service import SamService
    from app.services.zone_mask_service import ZoneMaskService

    sam = SamService()
    zone_svc = ZoneMaskService()

    # Load protection artifacts
    prot_mask, sky_mask, hatch_overlay = zone_svc.load_protection_artifacts(session_dir)

    points_dicts = [p.model_dump() for p in payload.points]
    binary_mask = sam.segment_with_points(
        image_bgr=orig_bgr,
        cache_key=payload.session_id,
        points=points_dicts,
        refine_edges=True
    )

    # Load existing session masks
    existing_masks = zone_svc.load_masks(session_dir, fallback_shape=orig_bgr.shape[:2])

    # Update the active layer with 'last-edited layer wins' rule and auto-protection
    updated_masks = zone_svc.update_layer(
        existing_masks, payload.layer, binary_mask, protected_mask=prot_mask
    )
    zone_svc.save_masks(session_dir, updated_masks)

    # Save combined mask as mask.png for backward compatibility
    combined_mask = zone_svc.get_combined_mask(updated_masks)
    cv2.imwrite(str(session_dir / "mask.png"), combined_mask)

    # Encode all 3 masks as data URLs
    mask_data_urls = {}
    for l_name, m in updated_masks.items():
        _, enc_m = cv2.imencode(".png", m)
        mask_data_urls[l_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"

    _, enc_comb = cv2.imencode(".png", combined_mask)
    comb_data_url = f"data:image/png;base64,{base64.b64encode(enc_comb).decode('utf-8')}"

    coverage_by_layer = zone_svc.get_coverage_pct(updated_masks)
    total_coverage = round(float(np.mean(combined_mask > 128) * 100), 2)
    sky_fraction_by_layer = zone_svc.get_sky_fractions(updated_masks, sky_mask)

    return {
        "session_id": payload.session_id,
        "active_layer": payload.layer,
        "mask_url": comb_data_url,
        "masks": mask_data_urls,
        "coverage_pct": total_coverage,
        "coverage_by_layer": coverage_by_layer,
        "sky_fraction_by_layer": sky_fraction_by_layer
    }

@router.post("/recolor-mask")
async def update_mask_and_recolor(payload: RecolorPayload):
    """
    Re-runs recoloring when user edits zone masks via interactive brush/tap.
    Supports both exterior multi-layer schemes and interior single-shades.
    """
    session_dir = settings.UPLOAD_DIR / payload.session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{payload.session_id}' not found."
        )

    orig_path = session_dir / "original.jpg"
    meta_path = session_dir / "meta.json"
    if not orig_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original image or session metadata missing."
        )

    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load original room image."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    from app.services.zone_mask_service import ZoneMaskService
    from app.services.recolor_service import RecolorService
    from app.schemas.recolor import SchemeResult, SchemeShadeInfo
    from app.models.paint import Paint
    from app.core.database import SessionLocal

    zone_svc = ZoneMaskService()
    recolor_service = RecolorService()

    # Load protection artifacts
    prot_mask, sky_mask, hatch_overlay = zone_svc.load_protection_artifacts(session_dir)

    # Load current layer masks
    zone_masks = zone_svc.load_masks(session_dir, fallback_shape=orig_bgr.shape[:2])

    # If custom_mask_base64 was passed, update specified layer
    if payload.custom_mask_base64:
        try:
            raw_b64 = payload.custom_mask_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",")[1]
            mask_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(mask_bytes, np.uint8)
            mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                orig_h, orig_w = orig_bgr.shape[:2]
                if mask_img.shape[:2] != (orig_h, orig_w):
                    mask_img = cv2.resize(mask_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                target_layer = payload.layer if payload.layer in zone_svc.LAYERS else "main"
                zone_masks = zone_svc.update_layer(zone_masks, target_layer, mask_img, protected_mask=prot_mask)
                zone_svc.save_masks(session_dir, zone_masks)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mask data: {e}"
            )

    combined_mask = zone_svc.get_combined_mask(zone_masks)
    cv2.imwrite(str(session_dir / "mask.png"), combined_mask)
    coverage_pct = zone_svc.get_coverage_pct(zone_masks)
    sky_fraction_by_layer = zone_svc.get_sky_fractions(zone_masks, sky_mask)

    mask_urls = {}
    for l_name, m in zone_masks.items():
        _, enc_m = cv2.imencode(".png", m)
        mask_urls[l_name] = f"data:image/png;base64,{base64.b64encode(enc_m).decode('utf-8')}"
    _, enc_comb = cv2.imencode(".png", combined_mask)
    comb_data_url = f"data:image/png;base64,{base64.b64encode(enc_comb).decode('utf-8')}"

    # Check if exterior scheme session
    if "schemes" in meta_data and meta_data["schemes"]:
        db = SessionLocal()
        all_paints = {p.id: p for p in db.query(Paint).all()}
        db.close()

        layer_contexts = recolor_service.prepare_multi_layer_context(
            image_bgr=orig_bgr,
            masks=zone_masks,
            surface_type="exterior"
        )

        # If user customized zone shades, apply them to the target scheme
        if payload.custom_shades and meta_data.get("schemes"):
            target_s_idx = payload.scheme_index or 1
            all_by_code = {p.shade_code: p for p in all_paints.values()}
            for s_meta in meta_data["schemes"]:
                if s_meta.get("scheme_index") == target_s_idx:
                    for role_key in ["main", "accent", "trim_gate"]:
                        if role_key in payload.custom_shades:
                            val = payload.custom_shades[role_key]
                            matched_p = None
                            if isinstance(val, int) and val in all_paints:
                                matched_p = all_paints[val]
                            elif isinstance(val, str):
                                if val in all_by_code:
                                    matched_p = all_by_code[val]
                                else:
                                    for p in all_paints.values():
                                        if p.shade_code.lower() == val.lower() or p.hex.lower() == val.lower():
                                            matched_p = p
                                            break
                            if matched_p:
                                s_meta[f"{role_key}_id"] = matched_p.id
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

        schemes_results: List[SchemeResult] = []
        for s_meta in meta_data["schemes"]:
            s_idx = s_meta["scheme_index"]
            mp = all_paints.get(s_meta["main_id"])
            ap = all_paints.get(s_meta["accent_id"])
            tp = all_paints.get(s_meta["trim_gate_id"])
            if not (mp and ap and tp):
                continue

            scheme_hexes = {"main": mp.hex, "accent": ap.hex, "trim_gate": tp.hex}
            comp_bgr = recolor_service.recolor_multi_layer(
                orig_bgr, layer_contexts, scheme_hexes, facade_mask=combined_mask
            )
            cv2.imwrite(str(session_dir / f"scheme_{s_idx}.jpg"), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_comp = cv2.imencode(".jpg", comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            comp_url = f"data:image/jpeg;base64,{base64.b64encode(enc_comp).decode('utf-8')}"

            shades_info = [
                SchemeShadeInfo(role="main", shade_id=mp.id, shade_name=mp.name, shade_code=mp.shade_code, hex=mp.hex, finish=mp.finish, color_family=mp.color_family),
                SchemeShadeInfo(role="accent", shade_id=ap.id, shade_name=ap.name, shade_code=ap.shade_code, hex=ap.hex, finish=ap.finish, color_family=ap.color_family),
                SchemeShadeInfo(role="trim_gate", shade_id=tp.id, shade_name=tp.name, shade_code=tp.shade_code, hex=tp.hex, finish=tp.finish, color_family=tp.color_family),
            ]
            schemes_results.append(SchemeResult(
                scheme_index=s_idx,
                reason=s_meta["reason"],
                shades=shades_info,
                image_url=comp_url
            ))

        legacy_results = []
        for s in schemes_results:
            main_sh = next((sh for sh in s.shades if sh.role == "main"), s.shades[0])
            legacy_results.append({
                "shade_id": main_sh.shade_id,
                "shade_name": main_sh.shade_name,
                "shade_code": main_sh.shade_code,
                "hex": main_sh.hex,
                "finish": main_sh.finish,
                "color_family": main_sh.color_family,
                "reason": s.reason,
                "image_url": s.image_url,
                "price": None
            })

        return {
            "session_id": payload.session_id,
            "mask_url": comb_data_url,
            "masks": mask_urls,
            "coverage_pct": coverage_pct,
            "coverage_by_layer": coverage_pct,
            "sky_fraction_by_layer": sky_fraction_by_layer,
            "schemes": [s.model_dump() for s in schemes_results],
            "results": legacy_results
        }

    else:
        # Interior single-shade flow
        shades = meta_data.get("shades", [])
        results: List[RepaintedShadeResult] = []
        shading_ctx = recolor_service.prepare_shading_context(
            image_bgr=orig_bgr,
            wall_mask=combined_mask,
            feather_radius=5
        )

        for idx, s in enumerate(shades, 1):
            repainted_bgr = recolor_service.recolor_with_context(
                context=shading_ctx,
                target_hex=s["hex"]
            )
            shade_filename = f"shade_{idx}.jpg"
            shade_path = session_dir / shade_filename
            cv2.imwrite(str(shade_path), repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            _, enc_shade = cv2.imencode(".jpg", repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            shade_b64 = base64.b64encode(enc_shade).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{shade_b64}"

            results.append(RepaintedShadeResult(
                shade_id=s["shade_id"],
                shade_name=s["shade_name"],
                shade_code=s["shade_code"],
                hex=s["hex"],
                finish=s["finish"],
                color_family=s["color_family"],
                reason=s["reason"],
                image_url=image_url,
                price=s.get("price")
            ))

        return {
            "session_id": payload.session_id,
            "mask_url": comb_data_url,
            "masks": mask_urls,
            "coverage_pct": coverage_pct,
            "results": results
        }

@router.post("/schemes/regenerate", response_model=SchemeResult)
async def regenerate_scheme(request: Request, payload: RegenerateSchemePayload):
    """
    Regenerates a single scheme (1-4) using the AI image editing model with fresh retries.
    """
    session_dir = settings.UPLOAD_DIR / payload.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    orig_path = session_dir / "original.jpg"
    meta_path = session_dir / "meta.json"
    if not orig_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=400, detail="Original image or metadata missing.")

    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to load original image.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    schemes_meta = meta_data.get("schemes", [])
    target_meta = next((s for s in schemes_meta if s["scheme_index"] == payload.scheme_index), None)
    if not target_meta:
        raise HTTPException(status_code=404, detail=f"Scheme #{payload.scheme_index} not found in session.")

    from app.services.zone_mask_service import ZoneMaskService
    from app.services.image_gen.service import ImageSchemeService
    from app.schemas.selection import SchemeSelectionItem
    from app.models.paint import Paint
    from app.core.database import SessionLocal

    zone_svc = ZoneMaskService()
    zone_masks = zone_svc.load_masks(session_dir, fallback_shape=orig_bgr.shape[:2])
    building_mask = zone_svc.get_combined_mask(zone_masks)

    db = SessionLocal()
    all_paints = {p.id: p for p in db.query(Paint).all()}
    db.close()

    _, enc_orig = cv2.imencode(".jpg", orig_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    orig_bytes = enc_orig.tobytes()

    client_ip = request.client.host if request.client else "unknown"
    image_scheme_svc = ImageSchemeService()

    scheme_item = SchemeSelectionItem(
        main_id=target_meta["main_id"],
        accent_id=target_meta["accent_id"],
        trim_gate_id=target_meta["trim_gate_id"],
        reason=target_meta["reason"]
    )

    try:
        new_scheme_res = await image_scheme_svc.render_single_scheme(
            s_idx=payload.scheme_index,
            scheme_item=scheme_item,
            all_paints=all_paints,
            original_bgr=orig_bgr,
            cropped_image_bytes=orig_bytes,
            zone_masks=zone_masks,
            building_mask=building_mask,
            client_ip=client_ip,
            session_dir=session_dir
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {e}")

    # Update metadata
    target_meta["status"] = new_scheme_res.status
    target_meta["mode"] = new_scheme_res.mode
    target_meta["mock"] = new_scheme_res.mock
    target_meta["delta_e"] = new_scheme_res.delta_e
    target_meta["drift_outside_pct"] = new_scheme_res.drift_outside_pct
    target_meta["error_message"] = new_scheme_res.error_message

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    return new_scheme_res
