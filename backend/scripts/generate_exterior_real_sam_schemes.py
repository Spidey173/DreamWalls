import cv2
import numpy as np
import json
import math
import asyncio
from pathlib import Path

from app.services.sam_service import SamService
from app.services.zone_mask_service import ZoneMaskService
from app.services.recolor_service import RecolorService
from app.services.scheme_service import SchemeService
from app.services.shortlist_service import hex_to_rgb, rgb_to_lab
from app.models.paint import Paint
from app.core.database import SessionLocal
from app.schemas.vision import RoomAnalysis

ARTIFACT_DIR = Path("/Users/spidey./.gemini/antigravity-ide/brain/c909709f-8cbb-4cab-bbfa-8a619c166b78")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def measure_top_quartile_delta_e(repainted_bgr: np.ndarray, orig_bgr: np.ndarray, mask: np.ndarray, target_hex: str):
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
    l_orig = orig_lab[:, :, 0]
    zone_bool = mask > 128
    lit_bool = zone_bool & (l_orig > 30)
    if not np.any(lit_bool):
        lit_bool = zone_bool

    q75 = np.percentile(l_orig[lit_bool], 75)
    top_q_mask = zone_bool & (l_orig >= q75)

    rep_rgb = cv2.cvtColor(repainted_bgr, cv2.COLOR_BGR2RGB)[top_q_mask]
    if len(rep_rgb) == 0:
        return None, None

    med_r = float(np.median(rep_rgb[:, 0]))
    med_g = float(np.median(rep_rgb[:, 1]))
    med_b = float(np.median(rep_rgb[:, 2]))

    l1, a1, b1 = rgb_to_lab(int(round(med_r)), int(round(med_g)), int(round(med_b)))
    r2, g2, b2 = hex_to_rgb(target_hex)
    l2, a2, b2 = rgb_to_lab(r2, g2, b2)

    delta_e = math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
    return delta_e, (l1, a1, b1)

async def main():
    print("=" * 70)
    print("REAL MOBILE-SAM ONNX SEGMENTATION & 4 EXTERIOR SCHEMES GENERATOR")
    print("Strictly NO hand-coded rectangles or synthetic masks!")
    print("=" * 70)

    # 1. Load exterior_house.jpg
    img_path = Path("test_photos/exterior_house.jpg")
    if not img_path.exists():
        img_path = Path("../test_photos/exterior_house.jpg")
    assert img_path.exists(), f"Image not found at {img_path}"

    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]
    print(f"Loaded {img_path}: {w}x{h} px")

    # 2. Real SAM Tap Coordinates (Fractions of width / height)
    # Strictly real MobileSAM inference (no synthetic masks or hand-coded bounding boxes)
    main_tap_pts = [
        {"x": 0.50, "y": 0.30, "label": 1}
    ]
    accent_tap_pts = [
        {"x": 0.55, "y": 0.38, "label": 1},
        {"x": 0.55, "y": 0.30, "label": 0},
        {"x": 0.55, "y": 0.45, "label": 0}
    ]
    gate_tap_pts = [
        {"x": 0.35, "y": 0.52, "label": 1},
        {"x": 0.35, "y": 0.35, "label": 0},
        {"x": 0.35, "y": 0.65, "label": 0}
    ]

    print("\n[Step 1] Real MobileSAM ONNX Inference...")
    print(f"  - Main Facade Tap: x={main_tap_pts[0]['x']}, y={main_tap_pts[0]['y']} (label=1)")
    print(f"  - Accent Slabs Taps: pos=({accent_tap_pts[0]['x']}, {accent_tap_pts[0]['y']}), negs=({accent_tap_pts[1]['x']}, {accent_tap_pts[1]['y']}), ({accent_tap_pts[2]['x']}, {accent_tap_pts[2]['y']})")
    print(f"  - Gate Taps: pos=({gate_tap_pts[0]['x']}, {gate_tap_pts[0]['y']}), negs=({gate_tap_pts[1]['x']}, {gate_tap_pts[1]['y']}), ({gate_tap_pts[2]['x']}, {gate_tap_pts[2]['y']})")

    sam = SamService()
    sam._ensure_models_loaded()
    zone_svc = ZoneMaskService()
    recolor_svc = RecolorService()

    m_main = sam.segment_with_points(orig_bgr, cache_key="real_exterior_gen", points=main_tap_pts, refine_edges=True)
    m_accent = sam.segment_with_points(orig_bgr, cache_key="real_exterior_gen", points=accent_tap_pts, refine_edges=True)
    m_gate = sam.segment_with_points(orig_bgr, cache_key="real_exterior_gen", points=gate_tap_pts, refine_edges=True)

    # 3. Non-overlapping layer update: last-edited layer wins
    masks = {"main": m_main, "accent": np.zeros((h, w), dtype=np.uint8), "trim_gate": np.zeros((h, w), dtype=np.uint8)}
    masks = zone_svc.update_layer(masks, "accent", m_accent)
    masks = zone_svc.update_layer(masks, "trim_gate", m_gate)
    m_comb = zone_svc.get_combined_mask(masks)

    coverage = zone_svc.get_coverage_pct(masks)
    print("\n[Step 2] Non-Overlapping Zone Layer Coverage:")
    for role, pct in coverage.items():
        px = np.sum(masks[role] > 128)
        print(f"  - {role:10s}: {pct:5.2f}% ({px:,} px)")
    print(f"  - Total Facade Coverage: {np.mean(m_comb > 128) * 100:.2f}% ({np.sum(m_comb > 128):,} px)")

    # Save zone masks as artifacts
    cv2.imwrite(str(ARTIFACT_DIR / "mask_main.png"), masks["main"])
    cv2.imwrite(str(ARTIFACT_DIR / "mask_accent.png"), masks["accent"])
    cv2.imwrite(str(ARTIFACT_DIR / "mask_trim_gate.png"), masks["trim_gate"])
    cv2.imwrite(str(ARTIFACT_DIR / "mask_combined.png"), m_comb)

    # Tinted overlay for visual inspection
    tinted = orig_bgr.copy()
    # Blue for main, Amber for accent, Green for trim
    tinted[masks["main"] > 128] = cv2.addWeighted(orig_bgr[masks["main"] > 128], 0.6, np.full_like(orig_bgr[masks["main"] > 128], [255, 100, 50]), 0.4, 0)
    tinted[masks["accent"] > 128] = cv2.addWeighted(orig_bgr[masks["accent"] > 128], 0.6, np.full_like(orig_bgr[masks["accent"] > 128], [50, 180, 255]), 0.4, 0)
    tinted[masks["trim_gate"] > 128] = cv2.addWeighted(orig_bgr[masks["trim_gate"] > 128], 0.6, np.full_like(orig_bgr[masks["trim_gate"] > 128], [80, 220, 80]), 0.4, 0)
    cv2.imwrite(str(ARTIFACT_DIR / "zones_overlay.jpg"), tinted, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 4. Prepare multi-layer context with 90th percentile normalization
    print("\n[Step 3] Precomputing Shading Contexts (90th percentile normalization)...")
    layer_contexts = recolor_svc.prepare_multi_layer_context(orig_bgr, masks, surface_type="exterior")
    l_ref = layer_contexts["main"]["l_ref"]
    print(f"  - Combined Facade L_ref (90th percentile of lit pixels): {l_ref:.2f}")

    # 5. Load DB paints and select 4 schemes
    db = SessionLocal()
    all_paints = {p.id: p for p in db.query(Paint).all()}
    scheme_svc = SchemeService()

    room_analysis = RoomAnalysis(
        room_type="exterior facade",
        architectural_style="Modern Contemporary",
        lighting="bright",
        lighting_condition="direct sunlight with cast shadows",
        dominant_colors=["off-white", "concrete grey", "dark iron"],
        current_wall_color_hex="#E8E2D7",
        mood="Clean, contemporary, premium residential look",
        scene_type="exterior",
        building_type="Two-Storey Residential House"
    )

    paints_list = list(all_paints.values())
    selected_schemes = await scheme_svc.select_schemes(db, room_analysis, paints_list[:15])
    db.close()

    print(f"\n[Step 4] Selected {len(selected_schemes)} Exterior Schemes:")
    scheme_outputs = []
    photometric_report = []

    for idx, s in enumerate(selected_schemes, start=1):
        mp = all_paints[s.main_id]
        ap = all_paints[s.accent_id]
        tp = all_paints[s.trim_gate_id]

        scheme_hexes = {"main": mp.hex, "accent": ap.hex, "trim_gate": tp.hex}
        comp_bgr = recolor_svc.recolor_multi_layer(orig_bgr, layer_contexts, scheme_hexes)
        out_path = ARTIFACT_DIR / f"exterior_scheme_{idx}.jpg"
        cv2.imwrite(str(out_path), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        scheme_outputs.append(comp_bgr)

        # Photometric deltaE measurements
        de_main, lab_main = measure_top_quartile_delta_e(comp_bgr, orig_bgr, masks["main"], mp.hex)
        de_accent, lab_accent = measure_top_quartile_delta_e(comp_bgr, orig_bgr, masks["accent"], ap.hex)
        de_trim, lab_trim = measure_top_quartile_delta_e(comp_bgr, orig_bgr, masks["trim_gate"], tp.hex)

        print(f"\n  Scheme #{idx}:")
        print(f"    Main Wall   : {mp.name} ({mp.shade_code}, {mp.hex}) | Top-Quartile deltaE = {de_main:.2f}")
        print(f"    Accent Slabs: {ap.name} ({ap.shade_code}, {ap.hex}) | Top-Quartile deltaE = {de_accent:.2f}")
        print(f"    Gate & Trims: {tp.name} ({tp.shade_code}, {tp.hex}) | Top-Quartile deltaE = {de_trim:.2f}")
        print(f"    Rationale   : {s.reason}")

        photometric_report.append({
            "scheme_index": idx,
            "main": {"name": mp.name, "code": mp.shade_code, "hex": mp.hex, "delta_e": round(de_main, 2)},
            "accent": {"name": ap.name, "code": ap.shade_code, "hex": ap.hex, "delta_e": round(de_accent, 2)},
            "trim": {"name": tp.name, "code": tp.shade_code, "hex": tp.hex, "delta_e": round(de_trim, 2)},
            "reason": s.reason
        })

    # 6. Build 2x2 Comparison Collage
    grid_top = np.hstack([scheme_outputs[0], scheme_outputs[1]])
    grid_bot = np.hstack([scheme_outputs[2], scheme_outputs[3]])
    collage = np.vstack([grid_top, grid_bot])
    # Resize collage for preview
    ch, cw = collage.shape[:2]
    preview_collage = cv2.resize(collage, (cw // 2, ch // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ARTIFACT_DIR / "exterior_schemes_2x2_comparison.jpg"), preview_collage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print("\n[Step 5] Saved all artifacts to:")
    print(f"  {ARTIFACT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
