import os
import sys
import argparse
import asyncio
from pathlib import Path
import cv2
import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import SessionLocal, Base, engine
from app.core.config import settings
from app.models.paint import Paint
from app.services.vision_service import VisionService
from app.services.shortlist_service import ShortlistService
from app.services.selection_service import SelectionService
from app.services.segmentation_service import SegmentationService
from app.services.recolor_service import RecolorService
from scripts.seed_db import seed_database

async def run_pipeline(image_path: Path, output_dir: Path):
    print("=" * 70)
    print("  PaintMyWall - Backend Pipeline End-to-End Test")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure database is seeded
    db = SessionLocal()
    paint_count = db.query(Paint).count()
    if paint_count == 0:
        print("Database is empty. Running initial seeder...")
        seed_database()
        paint_count = db.query(Paint).count()
    print(f"Connected to database with {paint_count} Asian Paints catalog entries.")

    # 1. Read input image
    if not image_path.exists():
        fallback_user = settings.UPLOAD_DIR / "user_room.png"
        if fallback_user.exists():
            image_path = fallback_user
        else:
            raise FileNotFoundError(f"Input image not found at {image_path}. Please supply an image.")

    print(f"\n[1/5] Loaded Input Image: {image_path} ({image_path.stat().st_size} bytes)")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Save original copy in output_dir
    orig_copy_path = output_dir / "original_room.jpg"
    with open(orig_copy_path, "wb") as f:
        f.write(image_bytes)

    # 2. Vision Service
    print(f"\n[2/5] Running Vision Analysis (Provider: {settings.LLM_PROVIDER})...")
    vision_service = VisionService()
    room_analysis = await vision_service.analyze_room(image_bytes, mime_type="image/jpeg")

    print(f"  • Room Type:       {room_analysis.room_type}")
    print(f"  • Lighting:        {room_analysis.lighting}")
    print(f"  • Current Wall:    {room_analysis.current_wall_color_hex}")
    print(f"  • Flooring:        {room_analysis.floor_color_hex}")
    print(f"  • Dominant Colors: {', '.join(room_analysis.dominant_furniture_colors)}")
    print(f"  • Style:           {room_analysis.style}")
    print(f"  • Mood:            {room_analysis.mood}")

    # 3. Shortlist Service (Done in pure code with LAB/HSL color harmony & lighting)
    print("\n[3/5] Computing Color Harmony Shortlist in LAB/HSL space...")
    shortlist_service = ShortlistService()
    shortlist = shortlist_service.shortlist_shades(db, room_analysis, min_count=20, max_count=40)
    print(f"  Filtered catalog down to {len(shortlist)} harmony candidates:")
    for idx, c in enumerate(shortlist[:5], 1):
        print(f"    {idx}. {c.name} ({c.shade_code}) - {c.color_family} [{c.harmony_category}] score: {c.score}")
    if len(shortlist) > 5:
        print(f"    ... and {len(shortlist) - 5} more shortlisted shades.")

    # 4. Selection Service (AI picks best 4, strictly verified against DB)
    print("\n[4/5] AI Selection of Best 4 Shades (with Catalog ID Verification)...")
    selection_service = SelectionService()
    top_4_selections = await selection_service.select_shades(db, room_analysis, shortlist)
    print(f"  Selected & Verified {len(top_4_selections)} Asian Paints Shades:")
    for idx, (paint, reason) in enumerate(top_4_selections, 1):
        print(f"    #{idx} [{paint.shade_code}] {paint.name} ({paint.hex}, {paint.finish})")
        print(f"       Reason: \"{reason}\"")

    # 5. Wall Segmentation (SegFormer ADE20K ONNX)
    print("\n[5/5] Detecting Wall Mask using SegFormer (ADE20K) ONNX...")
    segmentation_service = SegmentationService()
    wall_mask, mask_path = segmentation_service.detect_walls(image_bytes)
    wall_pixel_pct = (np.sum(wall_mask > 0) / wall_mask.size) * 100
    print(f"  Wall mask detected! Wall area: {wall_pixel_pct:.1f}% of image.")

    # Save mask to output_dir
    mask_out_path = output_dir / "wall_mask.png"
    cv2.imwrite(str(mask_out_path), wall_mask)
    print(f"  Saved wall mask to: {mask_out_path}")

    # 6. Recolor Service for each of the 4 shades
    print("\n[Repainting] Repainting walls in LAB color space for the 4 shades...")
    recolor_service = RecolorService()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    saved_outputs = []
    for idx, (paint, reason) in enumerate(top_4_selections, 1):
        repainted_bgr = recolor_service.recolor(
            image_bgr=img_bgr,
            wall_mask=wall_mask,
            target_hex=paint.hex,
            feather_radius=5
        )
        safe_name = paint.name.lower().replace(" ", "_").replace("/", "-")
        out_filename = f"shade_{idx}_{paint.shade_code}_{safe_name}.jpg"
        out_filepath = output_dir / out_filename
        cv2.imwrite(str(out_filepath), repainted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_outputs.append((paint, reason, out_filepath))
        print(f"  ✓ Saved shade #{idx} repainted image: {out_filepath}")

    db.close()

    print("\n" + "=" * 70)
    print("  SUMMARY OF REPAINTED OUTPUTS:")
    print("=" * 70)
    for idx, (paint, reason, path) in enumerate(saved_outputs, 1):
        print(f"\nSHADE #{idx}: {paint.name}")
        print(f"  • Brand:        {paint.brand}")
        print(f"  • Shade Code:   {paint.shade_code}")
        print(f"  • Hex Color:    {paint.hex}")
        print(f"  • Finish:       {paint.finish}")
        print(f"  • Color Family: {paint.color_family}")
        print(f"  • Design Note:  {reason}")
        print(f"  • Saved File:   {path}")
    print("\n" + "=" * 70)
    print(f"All 4 output images successfully generated and saved to:\n  {output_dir}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="PaintMyWall Backend CLI Pipeline Tester")
    parser.add_argument("--image", type=str, default=str(settings.UPLOAD_DIR / "user_room.png"),
                        help="Path to room image (JPG/PNG)")
    parser.add_argument("--output-dir", type=str, default=str(settings.DATA_DIR.parent.parent / "output_samples"),
                        help="Directory to save the 4 repainted images and masks")
    args = parser.parse_args()

    asyncio.run(run_pipeline(Path(args.image), Path(args.output_dir)))

if __name__ == "__main__":
    main()
