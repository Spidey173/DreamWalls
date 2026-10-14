import os
import sys
import asyncio
import cv2
import numpy as np
from pathlib import Path
import logging

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.paint import Paint
from app.services.vision_service import VisionService
from app.services.shortlist_service import ShortlistService
from app.services.selection_service import SelectionService
from app.services.segmentation_service import SegmentationService
from app.services.recolor_service import RecolorService
from scripts.seed_db import seed_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_photos_pipeline")

def draw_text_banner(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Draws a clean modern dark banner on top of the image panel."""
    h, w = img.shape[:2]
    out = img.copy()
    banner_h = 55
    overlay = out[0:banner_h, 0:w].copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.85, out[0:banner_h, 0:w], 0.15, 0, out[0:banner_h, 0:w])
    
    cv2.putText(out, title, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 205, 220), 1, cv2.LINE_AA)
    return out

async def process_photo(photo_path: Path, output_dir: Path, db):
    logger.info(f"\n=======================================================")
    logger.info(f"Processing: {photo_path.name}")
    logger.info(f"=======================================================")

    with open(photo_path, "rb") as f:
        image_bytes = f.read()

    img_bgr = cv2.imread(str(photo_path))
    if img_bgr is None:
        logger.error(f"Failed to load image at {photo_path}")
        return

    orig_h, orig_w = img_bgr.shape[:2]
    stem = photo_path.stem
    photo_out_dir = output_dir / stem
    photo_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vision & LLM Analysis
    vision_service = VisionService()
    try:
        room_analysis = await vision_service.analyze_room(image_bytes, mime_type="image/jpeg")
        logger.info(f"Room: {room_analysis.room_type} | Lighting: {room_analysis.lighting} | Style: {room_analysis.style}")
    except Exception as e:
        logger.warning(f"Vision analysis fallback due to: {e}")
        from app.schemas.vision import RoomAnalysisResult
        room_analysis = RoomAnalysisResult(
            room_type="living room",
            lighting="bright",
            current_wall_color_hex="#E0DDD7",
            floor_color_hex="#C29B72",
            dominant_furniture_colors=["#333333", "#EAEAEA"],
            style="contemporary",
            mood="fresh"
        )

    # 2. Shortlist & Shade Selection
    shortlist_service = ShortlistService()
    shortlist = shortlist_service.shortlist_shades(db, room_analysis, min_count=20, max_count=40)
    selection_service = SelectionService()
    selected_shades = await selection_service.select_shades(db, room_analysis, shortlist)

    # 3. Segmentation (Bilinear logit upsampling + Guided Filter edge refinement + SegFormer B2)
    seg_service = SegmentationService()
    wall_mask, mask_cache_path = seg_service.detect_walls(image_bytes, force_recompute=True)
    wall_pct = (np.mean(wall_mask > 0)) * 100
    logger.info(f"Wall segmentation completed. Wall coverage: {wall_pct:.1f}%")

    # Save wall mask
    cv2.imwrite(str(photo_out_dir / "wall_mask.png"), wall_mask)

    # 4. Recolor for all 4 shades
    recolor_service = RecolorService()
    repainted_panels = []

    for idx, (paint, reason) in enumerate(selected_shades, 1):
        repainted_bgr = recolor_service.recolor(
            image_bgr=img_bgr,
            wall_mask=wall_mask,
            target_hex=paint.hex,
            gamma=0.7,
            filter_type="guided",
            highlight_knee=210.0,
            highlight_max=250.0,
            feather_radius=3
        )

        shade_fn = f"shade_{idx}_{paint.shade_code}.jpg"
        cv2.imwrite(str(photo_out_dir / shade_fn), repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Panel with label banner
        panel = draw_text_banner(
            repainted_bgr,
            f"#{idx} {paint.name} ({paint.shade_code})",
            f"Hex: {paint.hex} | {paint.finish} | {paint.color_family}"
        )
        repainted_panels.append(panel)
        logger.info(f"  ✓ Repainted Shade #{idx}: {paint.name} ({paint.hex})")

    # 5. Create Mask Overlay panel
    mask_overlay = img_bgr.copy()
    green_tint = np.zeros_like(img_bgr)
    green_tint[:] = [40, 200, 40]  # Green tint in BGR
    mask_float = (wall_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    blended_overlay = (img_bgr * (1.0 - 0.35 * mask_float) + green_tint * (0.35 * mask_float)).astype(np.uint8)
    mask_panel = draw_text_banner(
        blended_overlay,
        "SegFormer B2 Guided Mask",
        f"Wall area: {wall_pct:.1f}% | Bilinear logits + edge refinement"
    )

    # 6. Original panel
    orig_panel = draw_text_banner(
        img_bgr,
        f"Original: {photo_path.name}",
        f"{orig_w}x{orig_h} | Base: {room_analysis.current_wall_color_hex} | Lighting: {room_analysis.lighting}"
    )

    # 7. Assemble 2x3 Side-by-Side Comparison Sheet
    # Standardize panel size for clean grid sheet
    grid_w = 720
    grid_h = int(orig_h * (grid_w / orig_w))

    resized_orig = cv2.resize(orig_panel, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_panel, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    r_panels = [cv2.resize(p, (grid_w, grid_h), interpolation=cv2.INTER_AREA) for p in repainted_panels]

    # Row 1: Original | Mask Overlay | Shade #1
    row1 = np.hstack([resized_orig, resized_mask, r_panels[0]])
    # Row 2: Shade #2 | Shade #3 | Shade #4
    row2 = np.hstack([r_panels[1], r_panels[2], r_panels[3]])

    # Combined Sheet
    sheet = np.vstack([row1, row2])
    sheet_path = output_dir / f"{stem}_comparison_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    logger.info(f"★ Saved Before/After Sheet: {sheet_path}")

    return {
        "photo": photo_path.name,
        "wall_pct": wall_pct,
        "shades": [(p.name, p.shade_code, p.hex) for p, _ in selected_shades],
        "sheet_path": str(sheet_path)
    }

async def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    test_photos_dir = root_dir / "test_photos"
    results_dir = test_photos_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure database is seeded
    seed_database()
    db = SessionLocal()

    photo_files = sorted(list(test_photos_dir.glob("*.jpg")) + list(test_photos_dir.glob("*.png")))
    if not photo_files:
        logger.error(f"No photos found in {test_photos_dir}!")
        return

    logger.info(f"Found {len(photo_files)} test photos in {test_photos_dir}.")

    reports = []
    for photo in photo_files:
        if photo.name.endswith("_comparison_sheet.jpg"):
            continue
        report = await process_photo(photo, results_dir, db)
        if report:
            reports.append(report)

    db.close()

    print("\n" + "=" * 80)
    print("                    TEST PHOTOS PIPELINE SUMMARY")
    print("=" * 80)
    for r in reports:
        print(f"\nPhoto: {r['photo']} (Wall Area: {r['wall_pct']:.1f}%)")
        print(f"Comparison Sheet: {r['sheet_path']}")
        print(f"Shades Applied:")
        for name, code, hex_val in r['shades']:
            print(f"   • {name} ({code}) - {hex_val}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
