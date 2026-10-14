import os
import sys
import json
import time
import tracemalloc
import resource
from pathlib import Path
import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.image_utils import detect_and_crop_camera_bar
from app.services.protection_service import ProtectionService
from app.services.auto_regions_service import AutoRegionsService, CandidateRegion


def get_rss_mb() -> float:
    """Returns current process RSS memory in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS ru_maxrss is in bytes; Linux in KB
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024.0 * 1024.0)
    return rusage.ru_maxrss / 1024.0


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

    # 1. Camera bar crop
    img_bgr, cropped_rows = detect_and_crop_camera_bar(raw_bgr)
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Start memory tracking
    tracemalloc.start()
    rss_before_mb = get_rss_mb()

    # 2. Protection masks
    t_prot_start = time.time()
    prot_svc = ProtectionService()
    prot_mask, sky_mask, _ = prot_svc.compute_protection_masks(img_bgr)
    t_prot_ms = (time.time() - t_prot_start) * 1000.0

    # 3. Auto-Regions computation with breakdown
    ar_svc = AutoRegionsService()

    t_emb_start = time.time()
    embedding, _ = ar_svc.sam_service.get_or_compute_embedding("benchmark_session", img_bgr)
    t_emb_ms = (time.time() - t_emb_start) * 1000.0

    t_comp_start = time.time()
    candidates = ar_svc.compute_candidate_regions(
        image_bgr=img_bgr,
        cache_key="benchmark_session",
        grid_size=16,
        protected_mask=prot_mask
    )
    t_comp_ms = (time.time() - t_comp_start) * 1000.0
    total_auto_region_time_ms = t_emb_ms + t_comp_ms

    # Memory measurement
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_mb = get_rss_mb()
    rss_delta_mb = max(0.0, rss_after_mb - rss_before_mb)
    peak_traced_mb = peak_mem / (1024.0 * 1024.0)

    # 4. Region statistics
    areas = [c.pixel_count for c in candidates]
    min_area = int(np.min(areas)) if areas else 0
    max_area = int(np.max(areas)) if areas else 0
    median_area = float(np.median(areas)) if areas else 0.0

    min_area_pct = round((min_area / total_pixels) * 100, 3)
    max_area_pct = round((max_area / total_pixels) * 100, 3)
    median_area_pct = round((median_area / total_pixels) * 100, 3)

    # 5. Save candidate regions to session
    session_dir = results_dir / "benchmark_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ar_svc.save_regions(session_dir, candidates)

    # 6. Generate distinct tinted overlay visualization
    overlay = img_bgr.copy().astype(np.float32)
    # Generate distinct colors for each region
    np.random.seed(42)
    colors = [
        (220, 50, 50),    # Blue/cyan
        (50, 200, 50),    # Green
        (50, 80, 230),    # Red/orange
        (200, 180, 50),   # Cyan
        (200, 50, 200),   # Magenta
        (50, 200, 220),   # Yellow
        (180, 100, 30),   # Deep blue
        (80, 180, 120),   # Seafoam
        (130, 60, 200),   # Rose
        (60, 120, 180),   # Gold
    ]

    for idx, cand in enumerate(candidates):
        c_bgr = np.array(colors[idx % len(colors)], dtype=np.float32)
        m_bool = cand.mask > 0
        overlay[m_bool] = 0.60 * overlay[m_bool] + 0.40 * c_bgr

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw region labels at centroids
    for idx, cand in enumerate(candidates):
        cx = int(round(cand.centroid[0] * w))
        cy = int(round(cand.centroid[1] * h))
        label = f"R{cand.region_id}:{cand.pixel_count}px"
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Test "Tap a region" at 5 representative sample points
    test_tap_points = [
        {"name": "Center Facade Wall", "point": {"x": 0.50, "y": 0.30}},
        {"name": "Balcony / Slabs", "point": {"x": 0.40, "y": 0.42}},
        {"name": "Gate / Entrance", "point": {"x": 0.52, "y": 0.52}},
        {"name": "Side Pillar / Band", "point": {"x": 0.15, "y": 0.35}},
        {"name": "Lower Detail / Trim", "point": {"x": 0.27, "y": 0.54}},
    ]

    tap_test_results = []
    for test_tap in test_tap_points:
        pt = test_tap["point"]
        picked = ar_svc.find_region_at_tap(candidates, pt["x"], pt["y"], (h, w))
        if picked is not None:
            tap_test_results.append({
                "tap_name": test_tap["name"],
                "tap_coord": pt,
                "found_region": True,
                "region_id": picked.region_id,
                "pixel_count": picked.pixel_count,
                "pct_of_image": round((picked.pixel_count / total_pixels) * 100, 2),
                "bbox": list(picked.bbox),
                "centroid": list(picked.centroid)
            })
        else:
            tap_test_results.append({
                "tap_name": test_tap["name"],
                "tap_coord": pt,
                "found_region": False,
                "region_id": None,
                "pixel_count": 0,
                "pct_of_image": 0.0
            })

    # 8. Save artifacts
    overlay_filename = "auto_regions_overlay.jpg"
    cv2.imwrite(str(results_dir / overlay_filename), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(artifact_dir / overlay_filename), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    metrics = {
        "image": {
            "path": str(test_photo_path),
            "raw_dimensions": {"width": w_orig, "height": h_orig},
            "cropped_dimensions": {"width": w, "height": h},
            "cropped_rows": cropped_rows,
            "total_pixels": total_pixels
        },
        "performance": {
            "embedding_time_ms": round(t_emb_ms, 2),
            "protection_masks_time_ms": round(t_prot_ms, 2),
            "candidate_regions_computation_time_ms": round(t_comp_ms, 2),
            "total_time_ms": round(total_auto_region_time_ms, 2),
            "rss_before_mb": round(rss_before_mb, 2),
            "rss_after_mb": round(rss_after_mb, 2),
            "rss_delta_mb": round(rss_delta_mb, 2),
            "traced_peak_mb": round(peak_traced_mb, 2)
        },
        "regions": {
            "candidate_count": len(candidates),
            "min_area_pixels": min_area,
            "min_area_pct": min_area_pct,
            "max_area_pixels": max_area,
            "max_area_pct": max_area_pct,
            "median_area_pixels": median_area,
            "median_area_pct": median_area_pct,
            "region_summaries": [
                {
                    "region_id": c.region_id,
                    "pixels": c.pixel_count,
                    "pct_of_image": round((c.pixel_count / total_pixels) * 100, 2),
                    "bbox": list(c.bbox),
                    "centroid": [round(c.centroid[0], 3), round(c.centroid[1], 3)]
                }
                for c in candidates
            ]
        },
        "tap_to_select_tests": tap_test_results,
        "saved_files": [
            str(results_dir / overlay_filename),
            str(artifact_dir / overlay_filename)
        ]
    }

    metrics_filename = "auto_regions_metrics.json"
    with open(results_dir / metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(artifact_dir / metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
