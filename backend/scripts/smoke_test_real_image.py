import os
import sys
import time
from pathlib import Path

# Add backend to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.core.config import settings
from app.services.image_utils import detect_and_crop_camera_bar
from app.services.segmentation_service import SegmentationService
from app.services.zone_mask_service import ZoneMaskService
from app.services.image_gen.gemini_client import GeminiImageClient
from app.services.image_gen.service import ImageSchemeService, calculate_delta_e_rgb

async def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST: REAL IMAGE MODEL GENERATION")
    print("=" * 70)

    model_name = settings.IMAGE_MODEL or "gemini-2.5-flash-image"
    api_key = settings.IMAGE_API_KEY or settings.GEMINI_API_KEY

    print(f"Configured Model : {model_name}")
    print(f"API Key Present  : {bool(api_key)} ({api_key[:8] + '...' if api_key else 'None'})")

    if not api_key:
        print("\n[-] SKIPPED REAL CALL: IMAGE_API_KEY is not configured in .env.")
        print("    To run against the live API, set:")
        print("    IMAGE_API_KEY=<your_gemini_api_key>")
        print("    IMAGE_MODEL=gemini-2.5-flash-image (or imagen-3.0-generate-002)")
        print("=" * 70)
        return

    img_path = Path("test_photos/exterior_house.jpg")
    if not img_path.exists():
        print(f"[-] Error: {img_path} not found.")
        return

    raw_bgr = cv2.imread(str(img_path))
    h_orig, w_orig = raw_bgr.shape[:2]
    print(f"[+] Loaded test photo: {w_orig}x{h_orig}")

    # Crop camera info bar
    cropped_bgr, cropped_rows = detect_and_crop_camera_bar(raw_bgr)
    h_crop, w_crop = cropped_bgr.shape[:2]
    print(f"[+] Auto-cropped camera info bar: removed {cropped_rows} rows -> {w_crop}x{h_crop} (aspect: {w_crop/h_crop:.4f})")

    _, enc_cropped = cv2.imencode(".jpg", cropped_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cropped_bytes = enc_cropped.tobytes()

    # SegFormer Wall Detection for compositing
    print("[+] Running SegFormer wall detection...")
    seg = SegmentationService()
    wall_mask, _ = seg.detect_walls(cropped_bytes, surface_type="exterior")
    zone_svc = ZoneMaskService()
    temp_dir = Path("uploads/smoke_test")
    temp_dir.mkdir(parents=True, exist_ok=True)
    zone_masks = zone_svc.init_session_masks(temp_dir, wall_mask)

    # Accent and gate zones
    accent_mask = np.zeros_like(wall_mask)
    accent_mask[int(h_crop*0.37):int(h_crop*0.42), int(w_crop*0.25):int(w_crop*0.75)] = 255
    accent_mask = accent_mask & wall_mask
    zone_masks = zone_svc.update_layer(zone_masks, "accent", accent_mask)

    gate_mask = np.zeros_like(wall_mask)
    gate_mask[int(h_crop*0.77):int(h_crop*0.96), int(w_crop*0.28):int(w_crop*0.65)] = 255
    zone_masks = zone_svc.update_layer(zone_masks, "trim_gate", gate_mask)
    building_mask = zone_svc.get_combined_mask(zone_masks)

    # Target Scheme (Scheme 1: Magnolia White + Golden Ochre + Midnight Navy)
    main_name, main_hex = "Magnolia White", "#FAF6EE"
    accent_name, accent_hex = "Golden Ochre", "#EDB858"
    trim_name, trim_hex = "Midnight Navy", "#1E3A52"

    prompt = (
        f"Repaint this building's exterior. "
        f"Main walls: {main_name} ({main_hex}). "
        f"Projecting slabs, window surrounds, side panels and tower: {accent_name} ({accent_hex}). "
        f"Gate: {trim_name} ({trim_hex}). "
        f"Keep the structure, sky, road, ground, vegetation and text unchanged. "
        f"Do not add lights or objects. Photorealistic."
    )
    print(f"\n[+] Built Prompt:\n    {prompt}\n")

    client = GeminiImageClient(api_key=api_key, model=model_name)
    print(f"[+] Sending request to real API model '{model_name}'...")

    t0 = time.perf_counter()
    raw_gen_bytes = None
    api_error = None
    try:
        raw_gen_bytes = await client.generate_scheme_image(
            image_bytes=cropped_bytes,
            mime_type="image/jpeg",
            prompt=prompt
        )
    except Exception as e:
        api_error = str(e)
    latency_ms = (time.perf_counter() - t0) * 1000

    if api_error:
        print(f"\n[-] API ERROR OCCURRED (Latency: {latency_ms:.1f} ms):")
        print(f"    Error Text: {api_error}")
        print("=" * 70)
        return

    print(f"\n[+] API CALL SUCCEEDED! Latency: {latency_ms:.1f} ms")
    print(f"    Received {len(raw_gen_bytes)} raw image bytes.")

    # Estimated cost
    unit_cost = 0.02 if "flash" in model_name.lower() else 0.04
    print(f"    Estimated Request Cost: ${unit_cost:.4f} USD")

    # Decode raw output
    gen_nparr = np.frombuffer(raw_gen_bytes, np.uint8)
    gen_bgr = cv2.imdecode(gen_nparr, cv2.IMREAD_COLOR)

    # 1. Save RAW model output (before compositing)
    raw_out_path = Path("smoke_raw_model_output.jpg")
    cv2.imwrite(str(raw_out_path), gen_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[+] Saved RAW model output (before compositing): {raw_out_path}")

    # 2. Post-process: composite with SegFormer mask
    img_scheme_svc = ImageSchemeService()
    comp_bgr, mean_delta, pct_drift = img_scheme_svc.composite_with_mask(
        original_bgr=cropped_bgr,
        generated_bgr=gen_bgr,
        building_mask=building_mask,
        s_idx=1
    )

    comp_out_path = Path("smoke_composited_result.jpg")
    cv2.imwrite(str(comp_out_path), comp_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[+] Saved COMPOSITED result (with SegFormer mask): {comp_out_path}")
    print(f"    Raw outside-mask drift before compositing: {pct_drift:.2f}% pixels changed (mean delta: {mean_delta:.2f})")

    # 3. Sample main wall color and compute deltaE
    main_m = zone_masks["main"]
    delta_e, sampled_hex = img_scheme_svc.check_main_wall_color(comp_bgr, main_m, main_hex, s_idx=1)
    print(f"[+] Main Wall Color Fidelity Check:")
    print(f"    Requested: {main_hex} | Sampled Median: {sampled_hex} | deltaE: {delta_e:.2f}")

    print("=" * 70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_smoke_test())
