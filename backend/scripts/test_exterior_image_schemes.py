import os
import sys
import time
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.paint import Paint
from app.schemas.vision import RoomAnalysis
from app.services.image_utils import detect_and_crop_camera_bar
from app.services.segmentation_service import SegmentationService
from app.services.zone_mask_service import ZoneMaskService
from app.services.shortlist_service import ShortlistService
from app.services.scheme_service import SchemeService
from app.services.image_gen.service import ImageSchemeService
from app.services.image_gen.cost_and_cache import cost_tracker, scheme_cache

async def main():
    img_path = Path("test_photos/exterior_house.jpg")
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        sys.exit(1)

    print(f"=== TESTING EXTERIOR IMAGE SCHEME RENDERING ON {img_path} ===")
    api_key = settings.IMAGE_API_KEY or settings.GEMINI_API_KEY
    print(f"API Key Present      : {bool(api_key)} ({api_key[:8] + '...' if api_key else 'None'})")
    print(f"IMAGE_ALLOW_MOCK     : {settings.IMAGE_ALLOW_MOCK}")
    print(f"Target Model         : {settings.IMAGE_MODEL}")
    if not api_key:
        print("[!] NOTICE: No IMAGE_API_KEY is configured. System will demonstrate production fallback or mock banner.")
    img_raw = cv2.imread(str(img_path))
    h_orig, w_orig = img_raw.shape[:2]
    orig_aspect = w_orig / h_orig
    print(f"1. Input Image: {w_orig}x{h_orig} (aspect ratio: {orig_aspect:.4f})")

    # Step 1: Camera Info Bar Auto-Crop
    t0 = time.perf_counter()
    img_cropped, cropped_rows = detect_and_crop_camera_bar(img_raw)
    t_crop = (time.perf_counter() - t0) * 1000
    h_crop, w_crop = img_cropped.shape[:2]
    crop_aspect = w_crop / h_crop
    print(f"2. Auto-crop Camera Info Bar: cropped {cropped_rows} rows in {t_crop:.2f} ms")
    print(f"   Cropped resolution: {w_crop}x{h_crop} (aspect ratio: {crop_aspect:.4f})")

    # Step 2: SegFormer Wall Segmentation
    _, enc_cropped = cv2.imencode(".jpg", img_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cropped_bytes = enc_cropped.tobytes()

    seg = SegmentationService()
    t0 = time.perf_counter()
    wall_mask, _ = seg.detect_walls(cropped_bytes, surface_type="exterior")
    t_seg = (time.perf_counter() - t0) * 1000
    print(f"3. SegFormer Wall Detection completed in {t_seg:.1f} ms")

    # Step 3: Zone Mask Initialization (3 layers)
    zone_svc = ZoneMaskService()
    temp_dir = Path("uploads/temp_test_session")
    temp_dir.mkdir(parents=True, exist_ok=True)
    zone_masks = zone_svc.init_session_masks(temp_dir, wall_mask)

    # Add realistic accent and gate layers for test
    accent_mask = np.zeros_like(wall_mask)
    accent_mask[int(h_crop*0.37):int(h_crop*0.42), int(w_crop*0.25):int(w_crop*0.75)] = 255
    accent_mask[int(h_crop*0.58):int(h_crop*0.62), int(w_crop*0.25):int(w_crop*0.75)] = 255
    accent_mask = accent_mask & wall_mask
    zone_masks = zone_svc.update_layer(zone_masks, "accent", accent_mask)

    gate_mask = np.zeros_like(wall_mask)
    gate_mask[int(h_crop*0.77):int(h_crop*0.96), int(w_crop*0.28):int(w_crop*0.65)] = 255
    zone_masks = zone_svc.update_layer(zone_masks, "trim_gate", gate_mask)

    coverage = zone_svc.get_coverage_pct(zone_masks)
    print(f"4. Zone Masks: main={coverage['main']}%, accent={coverage['accent']}%, trim_gate={coverage['trim_gate']}%")

    # Step 4: Database Schemes via SchemeService
    db = SessionLocal()
    room_analysis = RoomAnalysis(
        scene_type="exterior",
        lighting="bright",
        current_wall_color_hex="#D0C8B8",
        style="Modern Contemporary",
        mood="Welcoming & Sophisticated",
        building_type="Independent House / Villa",
        light_conditions="sunny"
    )

    shortlist_svc = ShortlistService()
    shortlist = shortlist_svc.shortlist_shades(db, room_analysis, min_count=20, max_count=40)
    scheme_svc = SchemeService()
    selected_schemes = await scheme_svc.select_schemes(db, room_analysis, shortlist)
    all_paints = {p.id: p for p in db.query(Paint).all()}
    db.close()

    print(f"5. Schemes from SchemeService (Count: {len(selected_schemes)}):")
    for idx, s in enumerate(selected_schemes, 1):
        mp = all_paints[s.main_id]
        ap = all_paints[s.accent_id]
        tp = all_paints[s.trim_gate_id]
        print(f"   Scheme {idx}: Main='{mp.name}' ({mp.shade_code}, {mp.hex}) | Accent='{ap.name}' ({ap.shade_code}, {ap.hex}) | Trim='{tp.name}' ({tp.shade_code}, {tp.hex})")

    # Step 5: Render Schemes with ImageSchemeService
    img_scheme_svc = ImageSchemeService()
    out_dir = Path("/Users/spidey./.gemini/antigravity-ide/brain/c909709f-8cbb-4cab-bbfa-8a619c166b78")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results, mode, fallback_msg = await img_scheme_svc.render_schemes_parallel(
        original_bgr=img_cropped,
        cropped_image_bytes=cropped_bytes,
        zone_masks=zone_masks,
        selected_schemes=selected_schemes,
        all_paints=all_paints,
        client_ip="127.0.0.1",
        session_dir=temp_dir
    )
    total_render_ms = (time.perf_counter() - t0) * 1000
    print(f"6. Parallel Rendering completed in {total_render_ms:.1f} ms (mode={mode}, fallback={fallback_msg})")

    # Step 6: Post-Process Verification Measurements
    building_mask = zone_svc.get_combined_mask(zone_masks)
    outside_mask = (building_mask == 0)

    rendered_images = []
    print("\n=== MEASURED VERIFICATION RESULTS ===")
    for idx, res in enumerate(results, 1):
        saved_img_path = temp_dir / f"scheme_{idx}.jpg"
        comp_bgr = cv2.imread(str(saved_img_path))
        rendered_images.append(comp_bgr)

        # In-memory exact compositing check:
        # Re-run compositing in memory to verify 0.00 pixel difference before lossy JPEG encoding
        comp_in_mem, mean_d, pct_d = img_scheme_svc.composite_with_mask(
            img_cropped, comp_bgr, building_mask, idx
        )
        diff_in_mem = np.abs(comp_in_mem[outside_mask].astype(int) - img_cropped[outside_mask].astype(int))
        max_in_mem_diff = np.max(diff_in_mem)
        exact_outside_match = (max_in_mem_diff == 0)

        # Measurement B: Color check (deltaE)
        mp = all_paints[selected_schemes[idx - 1].main_id]
        if res.mock:
            delta_e_str = "N/A (Mock - No API Key)"
            drift_str = "N/A (Mock - No API Key)"
        else:
            delta_e_str = f"deltaE = {res.delta_e} vs requested {mp.hex}" if res.delta_e is not None else "N/A"
            drift_str = f"{res.drift_outside_pct}% pixels changed before compositing" if res.drift_outside_pct is not None else "N/A"

        print(f"   [Scheme {idx}] Status: {res.status} | Mode: {res.mode} | Mock: {res.mock}")
        print(f"      - In-Memory Outside Building Exact Match: {exact_outside_match} (max diff = {max_in_mem_diff})")
        print(f"      - Raw AI Image Drift Outside Mask: {drift_str}")
        print(f"      - Main Wall Repainted Color Check: {delta_e_str}")
        print(f"      - Dimensions: {comp_bgr.shape[1]}x{comp_bgr.shape[0]} (aspect ratio: {comp_bgr.shape[1]/comp_bgr.shape[0]:.4f})")

        # Save artifact image
        out_scheme_path = out_dir / f"ai_scheme_{idx}.jpg"
        cv2.imwrite(str(out_scheme_path), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Create 2x2 comparison
    h_r, w_r = rendered_images[0].shape[:2]
    scale = 0.5
    h_s, w_s = int(h_r * scale), int(w_r * scale)

    panel1 = cv2.resize(rendered_images[0], (w_s, h_s))
    panel2 = cv2.resize(rendered_images[1], (w_s, h_s))
    panel3 = cv2.resize(rendered_images[2], (w_s, h_s))
    panel4 = cv2.resize(rendered_images[3], (w_s, h_s))

    top_row = np.hstack([panel1, panel2])
    bot_row = np.hstack([panel3, panel4])
    grid_2x2 = np.vstack([top_row, bot_row])

    # Overlay labels
    def add_label(img, text, pos):
        cv2.rectangle(img, (pos[0]-5, pos[1]-20), (pos[0]+280, pos[1]+8), (20, 20, 20), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    add_label(grid_2x2, f"Scheme 1: {all_paints[selected_schemes[0].main_id].name}", (15, 30))
    add_label(grid_2x2, f"Scheme 2: {all_paints[selected_schemes[1].main_id].name}", (w_s + 15, 30))
    add_label(grid_2x2, f"Scheme 3: {all_paints[selected_schemes[2].main_id].name}", (15, h_s + 30))
    add_label(grid_2x2, f"Scheme 4: {all_paints[selected_schemes[3].main_id].name}", (w_s + 15, h_s + 30))

    grid_path = out_dir / "ai_schemes_2x2_comparison.jpg"
    cv2.imwrite(str(grid_path), grid_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"\nSaved 2x2 AI composite comparison to: {grid_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
