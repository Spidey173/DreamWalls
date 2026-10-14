import os
import sys
import json
import time
import base64
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.services.image_utils import detect_and_crop_camera_bar
from app.services.recolor_service import RecolorService
from app.core.config import settings


def main():
    root_dir = BACKEND_DIR.parent
    test_photo_path = root_dir / "test_photos" / "exterior_house.jpg"
    results_dir = root_dir / "test_photos" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path("/Users/spidey./.gemini/antigravity-ide/brain/c909709f-8cbb-4cab-bbfa-8a619c166b78")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    assert test_photo_path.exists(), f"Image not found at {test_photo_path}"

    raw_bgr = cv2.imread(str(test_photo_path))
    h_orig, w_orig = raw_bgr.shape[:2]

    # 1. Camera bar crop verification
    cropped_bgr, cropped_rows = detect_and_crop_camera_bar(raw_bgr)
    h_crop, w_crop = cropped_bgr.shape[:2]
    total_pixels = h_crop * w_crop

    timing_log = {}

    with TestClient(app) as client:
        # 2. Step 1: POST /api/v1/analyze
        print("Executing Step 1: POST /api/v1/analyze ...")
        t0 = time.time()
        with open(test_photo_path, "rb") as f:
            resp_analyze = client.post(
                "/api/v1/analyze",
                files={"file": ("exterior_house.jpg", f, "image/jpeg")}
            )
        t_analyze_ms = (time.time() - t0) * 1000.0
        timing_log["analyze_ms"] = round(t_analyze_ms, 2)
        assert resp_analyze.status_code == 200, f"Analyze failed: {resp_analyze.text}"
        data_analyze = resp_analyze.json()
        session_id = data_analyze["session_id"]
        print(f"  Session created: {session_id} in {t_analyze_ms:.1f}ms")

        # 3. Step 2: POST /api/v1/select-zone for MAIN wall (magic_wand tap at facade)
        print("Executing Step 2: POST /api/v1/select-zone (tool=magic_wand, layer=main) ...")
        t0 = time.time()
        resp_main = client.post(
            "/api/v1/select-zone",
            json={
                "session_id": session_id,
                "layer": "main",
                "tool": "magic_wand",
                "point": {"x": 0.50, "y": 0.30},
                "tolerance": 12.0,
                "operation": "replace"
            }
        )
        t_main_ms = (time.time() - t0) * 1000.0
        timing_log["select_main_magic_wand_ms"] = round(t_main_ms, 2)
        assert resp_main.status_code == 200, f"Select main failed: {resp_main.text}"
        data_main = resp_main.json()
        print(f"  Main zone updated: {data_main.get('coverage_by_layer')} in {t_main_ms:.1f}ms")

        # 4. Step 3: POST /api/v1/select-zone for ACCENT (sam tap at projecting slab)
        print("Executing Step 3: POST /api/v1/select-zone (tool=sam, layer=accent) ...")
        t0 = time.time()
        resp_accent = client.post(
            "/api/v1/select-zone",
            json={
                "session_id": session_id,
                "layer": "accent",
                "tool": "sam",
                "point": {"x": 0.25, "y": 0.23, "label": 1.0},
                "operation": "replace"
            }
        )
        t_accent_ms = (time.time() - t0) * 1000.0
        timing_log["select_accent_sam_ms"] = round(t_accent_ms, 2)
        assert resp_accent.status_code == 200, f"Select accent failed: {resp_accent.text}"
        data_accent = resp_accent.json()
        print(f"  Accent zone updated: {data_accent.get('coverage_by_layer')} in {t_accent_ms:.1f}ms")

        # 5. Step 4: POST /api/v1/select-zone for TRIM_GATE (sam tap at side architectural feature/trim)
        # Includes mid-tone visible custom shades override:
        # Main: Raw Jute (#E3D5C0), Accent: Terracotta Glow (#C96849), Trim: Dark Espresso Brown (#3B2314)
        print("Executing Step 4: POST /api/v1/select-zone (tool=sam, layer=trim_gate + custom mid-tone shades) ...")
        t0 = time.time()
        resp_trim = client.post(
            "/api/v1/select-zone",
            json={
                "session_id": session_id,
                "layer": "trim_gate",
                "tool": "sam",
                "point": {"x": 0.15, "y": 0.35, "label": 1.0},
                "operation": "replace",
                "custom_shades": {
                    "main": "#E3D5C0",       # Mid-tone Warm Beige / Raw Jute
                    "accent": "#C96849",     # Terracotta Glow
                    "trim_gate": "#3B2314"   # Dark Espresso Brown
                }
            }
        )
        t_trim_ms = (time.time() - t0) * 1000.0
        timing_log["select_trim_sam_and_render_ms"] = round(t_trim_ms, 2)
        assert resp_trim.status_code == 200, f"Select trim failed: {resp_trim.text}"
        final_data = resp_trim.json()
        print(f"  Trim/gate zone updated and rendered: {final_data.get('coverage_by_layer')} in {t_trim_ms:.1f}ms")

    # 6. Load session directory assets & masks
    session_dir = settings.UPLOAD_DIR / session_id

    main_mask = cv2.imread(str(session_dir / "mask_main.png"), cv2.IMREAD_GRAYSCALE)
    accent_mask = cv2.imread(str(session_dir / "mask_accent.png"), cv2.IMREAD_GRAYSCALE)
    trim_mask = cv2.imread(str(session_dir / "mask_trim_gate.png"), cv2.IMREAD_GRAYSCALE)
    prot_mask = cv2.imread(str(session_dir / "protected_mask.png"), cv2.IMREAD_GRAYSCALE)
    sky_mask = cv2.imread(str(session_dir / "sky_mask.png"), cv2.IMREAD_GRAYSCALE)
    hatch_rgba = cv2.imread(str(session_dir / "protected_hatched.png"), cv2.IMREAD_UNCHANGED)

    orig_cropped = cv2.imread(str(session_dir / "original.jpg"))
    h_act, w_act = orig_cropped.shape[:2]
    assert (h_act, w_act) == (h_crop, w_crop), "Crop dimensions do not match API saved original"

    # 7. Protection overlay
    comp_overlay = orig_cropped.copy()
    if hatch_rgba is not None and hatch_rgba.shape[-1] == 4:
        hatch_alpha = hatch_rgba[:, :, 3].astype(np.float32) / 255.0
        for c in range(3):
            comp_overlay[:, :, c] = np.clip(
                comp_overlay[:, :, c] * (1.0 - hatch_alpha) + hatch_rgba[:, :, c] * hatch_alpha,
                0,
                255
            ).astype(np.uint8)

    # 8. 3-Zone Tinted Layer Overlay
    # Main: Soft Blue [235, 180, 80] (BGR)
    # Accent: Coral/Orange [50, 120, 240] (BGR)
    # Trim: Emerald Green [80, 200, 80] (BGR)
    zones_overlay = orig_cropped.copy().astype(np.float32)
    m_active = main_mask > 0
    a_active = accent_mask > 0
    t_active = trim_mask > 0

    zones_overlay[m_active] = 0.55 * zones_overlay[m_active] + 0.45 * np.array([235, 180, 80], dtype=np.float32)
    zones_overlay[a_active] = 0.55 * zones_overlay[a_active] + 0.45 * np.array([50, 120, 240], dtype=np.float32)
    zones_overlay[t_active] = 0.55 * zones_overlay[t_active] + 0.45 * np.array([80, 200, 80], dtype=np.float32)
    zones_overlay = np.clip(zones_overlay, 0, 255).astype(np.uint8)

    # 9. Load recolored schemes
    recolored_schemes = []
    combined_mask = (main_mask > 0) | (accent_mask > 0) | (trim_mask > 0)
    unselected_mask = ~combined_mask
    unselected_pixel_count = int(np.sum(unselected_mask))

    for idx in range(1, 5):
        scheme_path = session_dir / f"scheme_{idx}.jpg"
        if not scheme_path.exists():
            scheme_path = session_dir / f"recolored_scheme_{idx}.jpg"
        assert scheme_path.exists(), f"Scheme image {scheme_path} missing!"
        out_bgr = cv2.imread(str(scheme_path))
        recolored_schemes.append(out_bgr)

    # 10. Byte identity verification
    # A) In-memory algorithmic verification: run recolor_multi_layer in memory and test diff
    recolor_svc = RecolorService()
    masks_dict = {"main": main_mask, "accent": accent_mask, "trim_gate": trim_mask}
    contexts = recolor_svc.prepare_multi_layer_context(orig_cropped, masks_dict, surface_type="exterior")
    custom_scheme_hexes = {"main": "#E3D5C0", "accent": "#C96849", "trim_gate": "#3B2314"}
    in_memory_recolored = recolor_svc.recolor_multi_layer(
        orig_cropped, contexts, custom_scheme_hexes, facade_mask=combined_mask.astype(np.uint8) * 255
    )

    unsel_orig_raw = orig_cropped[unselected_mask]
    unsel_mem_raw = in_memory_recolored[unselected_mask]
    mem_diff = np.abs(unsel_mem_raw.astype(np.int32) - unsel_orig_raw.astype(np.int32))
    mem_max_diff = int(np.max(mem_diff))
    mem_identical_count = int(np.sum(np.all(mem_diff == 0, axis=-1)))

    byte_identity_results = [
        {
            "verification_type": "in_memory_unselected_pixels",
            "unselected_pixels_total": unselected_pixel_count,
            "byte_identical_pixels": mem_identical_count,
            "max_difference": mem_max_diff,
            "is_100_percent_identical": bool(mem_max_diff == 0 and mem_identical_count == unselected_pixel_count)
        }
    ]

    for idx, out_bgr in enumerate(recolored_schemes, start=1):
        disk_diff = np.abs(out_bgr[unselected_mask].astype(np.int32) - unsel_orig_raw.astype(np.int32))
        byte_identity_results.append({
            "verification_type": f"scheme_{idx}_disk_jpeg_lossy",
            "unselected_pixels_total": unselected_pixel_count,
            "byte_identical_pixels": int(np.sum(np.all(disk_diff == 0, axis=-1))),
            "max_difference": int(np.max(disk_diff)),
            "is_100_percent_identical": bool(np.max(disk_diff) == 0)
        })

    # 11. 2x2 comparison grid
    grid_top = np.hstack([recolored_schemes[0], recolored_schemes[1]])
    grid_bot = np.hstack([recolored_schemes[2], recolored_schemes[3]])
    grid_2x2 = np.vstack([grid_top, grid_bot])

    # 12. Calculate zone metrics and overlaps
    main_px = int(np.sum(main_mask > 0))
    accent_px = int(np.sum(accent_mask > 0))
    trim_px = int(np.sum(trim_mask > 0))

    sky_in_main = int(np.sum((main_mask > 0) & (sky_mask > 0))) if sky_mask is not None else 0
    sky_in_accent = int(np.sum((accent_mask > 0) & (sky_mask > 0))) if sky_mask is not None else 0
    sky_in_trim = int(np.sum((trim_mask > 0) & (sky_mask > 0))) if sky_mask is not None else 0

    frac_sky_main = (sky_in_main / main_px) if main_px > 0 else 0.0
    frac_sky_accent = (sky_in_accent / accent_px) if accent_px > 0 else 0.0
    frac_sky_trim = (sky_in_trim / trim_px) if trim_px > 0 else 0.0

    overlap_main_accent = int(np.sum((main_mask > 0) & (accent_mask > 0)))
    overlap_main_trim = int(np.sum((main_mask > 0) & (trim_mask > 0)))
    overlap_accent_trim = int(np.sum((accent_mask > 0) & (trim_mask > 0)))
    total_overlap = overlap_main_accent + overlap_main_trim + overlap_accent_trim

    sky_px_total = int(np.sum(sky_mask > 0)) if sky_mask is not None else 0
    prot_px_total = int(np.sum(prot_mask > 0)) if prot_mask is not None else 0

    # 13. Save outputs to test_photos/results/ and artifact directory
    files_to_save = {
        "real_api_protected_overlay.jpg": comp_overlay,
        "real_api_zones_overlay.jpg": zones_overlay,
        "real_api_scheme_1.jpg": recolored_schemes[0],
        "real_api_scheme_2.jpg": recolored_schemes[1],
        "real_api_scheme_3.jpg": recolored_schemes[2],
        "real_api_scheme_4.jpg": recolored_schemes[3],
        "real_api_2x2_comparison.jpg": grid_2x2,
    }

    for fname, img in files_to_save.items():
        dst1 = results_dir / fname
        dst2 = artifact_dir / fname
        cv2.imwrite(str(dst1), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(dst2), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    metrics_report = {
        "image": {
            "source_path": str(test_photo_path),
            "raw_dimensions": {"width": w_orig, "height": h_orig},
            "cropped_dimensions": {"width": w_crop, "height": h_crop},
            "cropped_rows": cropped_rows,
            "total_pixels": total_pixels
        },
        "api_session_id": session_id,
        "latencies_ms": timing_log,
        "protection": {
            "sky_pixels": sky_px_total,
            "sky_pct": round((sky_px_total / total_pixels) * 100, 2),
            "protected_dilated_pixels": prot_px_total,
            "protected_dilated_pct": round((prot_px_total / total_pixels) * 100, 2)
        },
        "zones": {
            "main": {
                "tool": "magic_wand (tap=(0.50, 0.30), tol=12.0)",
                "pixels": main_px,
                "coverage_pct": round((main_px / total_pixels) * 100, 2),
                "sky_pixels": sky_in_main,
                "sky_fraction": frac_sky_main,
                "sky_fraction_pct": f"{frac_sky_main * 100:.4f}%",
                "meets_threshold_less_than_1pct": bool(frac_sky_main < 0.01)
            },
            "accent": {
                "tool": "sam (tap=(0.25, 0.23))",
                "pixels": accent_px,
                "coverage_pct": round((accent_px / total_pixels) * 100, 2),
                "sky_pixels": sky_in_accent,
                "sky_fraction": frac_sky_accent,
                "sky_fraction_pct": f"{frac_sky_accent * 100:.4f}%",
                "meets_threshold_less_than_1pct": bool(frac_sky_accent < 0.01)
            },
            "trim_gate": {
                "tool": "sam (tap=(0.15, 0.35))",
                "pixels": trim_px,
                "coverage_pct": round((trim_px / total_pixels) * 100, 2),
                "sky_pixels": sky_in_trim,
                "sky_fraction": frac_sky_trim,
                "sky_fraction_pct": f"{frac_sky_trim * 100:.4f}%",
                "meets_threshold_less_than_1pct": bool(frac_sky_trim < 0.01)
            }
        },
        "overlaps": {
            "overlap_main_accent_pixels": overlap_main_accent,
            "overlap_main_trim_pixels": overlap_main_trim,
            "overlap_accent_trim_pixels": overlap_accent_trim,
            "total_overlap_pixels": total_overlap,
            "zero_overlap_verified": bool(total_overlap == 0)
        },
        "mid_tone_schemes_override": {
            "main": "#E3D5C0 (Raw Jute)",
            "accent": "#C96849 (Terracotta Glow)",
            "trim_gate": "#3B2314 (Dark Espresso Brown)"
        },
        "byte_identity_unselected_pixels": {
            "unselected_pixels_total": unselected_pixel_count,
            "unselected_pixels_pct": round((unselected_pixel_count / total_pixels) * 100, 2),
            "verifications": byte_identity_results
        },
        "saved_files": {
            "test_photos_results": [str(results_dir / f) for f in files_to_save.keys()],
            "artifact_directory": [str(artifact_dir / f) for f in files_to_save.keys()]
        }
    }

    metrics_filename = "real_api_metrics.json"
    with open(results_dir / metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    with open(artifact_dir / metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    print("\n=== REAL API REVIEW SCRIPT EXECUTION REPORT ===")
    print(json.dumps(metrics_report, indent=2))


if __name__ == "__main__":
    main()
