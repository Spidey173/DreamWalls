import pytest
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.database import Base, engine, SessionLocal
from scripts.seed_db import seed_database
from app.models.paint import Paint
from app.services.protection_service import ProtectionService
from app.services.selection_service import SelectionService
from app.services.zone_mask_service import ZoneMaskService
from app.services.recolor_service import RecolorService


TEST_PHOTOS_DIR = Path(__file__).resolve().parent.parent.parent / "test_photos"
EXTERIOR_HOUSE = TEST_PHOTOS_DIR / "exterior_house.jpg"


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    seed_database()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def exterior_image_bgr():
    assert EXTERIOR_HOUSE.exists(), f"Test photo not found at {EXTERIOR_HOUSE}"
    img = cv2.imread(str(EXTERIOR_HOUSE))
    assert img is not None
    return img


# ==============================================================================
# 1. SegFormer Auto-Protect Tests
# ==============================================================================

def test_segformer_auto_protect_exterior_house(exterior_image_bgr):
    """
    Verify SegFormer auto-protect isolates sky, ground, veg, and vehicles,
    dilating protected pixels by ~2px (5x5 kernel), and producing a hatched overlay.
    """
    prot_svc = ProtectionService()
    h, w = exterior_image_bgr.shape[:2]
    total_pixels = h * w

    prot_mask, sky_mask, hatched_overlay = prot_svc.compute_protection_masks(exterior_image_bgr)

    assert prot_mask.shape == (h, w)
    assert sky_mask.shape == (h, w)
    assert hatched_overlay.shape == (h, w, 4)

    sky_pixels = int(np.sum(sky_mask > 0))
    prot_pixels = int(np.sum(prot_mask > 0))

    sky_pct = (sky_pixels / total_pixels) * 100.0
    prot_pct = (prot_pixels / total_pixels) * 100.0

    # Overcast sky on exterior_house.jpg is ~18% of the image
    assert sky_pct > 10.0, f"Expected sky > 10%, got {sky_pct:.2f}%"
    # Ground + sky + veg is ~58% of the image
    assert prot_pct > 40.0, f"Expected protected > 40%, got {prot_pct:.2f}%"

    # All sky pixels must be included in dilated protected mask
    sky_in_protected = np.sum((sky_mask > 0) & (prot_mask > 0))
    assert sky_in_protected == sky_pixels, "All sky pixels must be within the protected mask!"

    # Hatched overlay must have non-zero alpha pixels where protected
    assert np.any(hatched_overlay[:, :, 3] > 0)


# ==============================================================================
# 2. Selection Tools (Magic Wand, Polygon, Rectangle, Brush, SAM) Tests
# ==============================================================================

def test_magic_wand_selection_no_sky_leakage(exterior_image_bgr):
    """
    Verify Magic Wand flood-fill from tap on white building under overcast sky:
    - Fills continuous facade
    - Respects dilated protected boundary
    - Strictly 0 sky pixels inside zone (< 1%)
    """
    prot_svc = ProtectionService()
    sel_svc = SelectionService()

    prot_mask, sky_mask, _ = prot_svc.compute_protection_masks(exterior_image_bgr)

    # Tap on the center-upper white wall: (x=0.5, y=0.3)
    main_selection = sel_svc.magic_wand(
        image_bgr=exterior_image_bgr,
        seed_point={"x": 0.5, "y": 0.3},
        tolerance=12.0,
        protected_mask=prot_mask,
        active_mask=None,
        operation="replace"
    )

    h, w = exterior_image_bgr.shape[:2]
    total_pixels = h * w
    zone_pixels = int(np.sum(main_selection > 0))

    assert zone_pixels > 0, "Magic wand should select wall area"
    cov_pct = (zone_pixels / total_pixels) * 100.0
    assert cov_pct > 20.0, f"Expected coverage > 20%, got {cov_pct:.2f}%"

    # CRITICAL CHECK: Sky fraction must be < 1% (strictly 0.0%)
    sky_in_zone = int(np.sum((main_selection > 0) & (sky_mask > 0)))
    sky_fraction = prot_svc.calculate_sky_fraction(main_selection, sky_mask)

    assert sky_in_zone == 0, f"Sky pixels leaked into main zone: {sky_in_zone}"
    assert sky_fraction < 0.01
    assert sky_fraction == 0.0


def test_polygon_rectangle_and_brush_selection(exterior_image_bgr):
    """
    Verify Polygon, Rectangle, and Brush selection tools:
    - Clip to image boundary
    - Auto-subtract protected mask
    - Clean holes and small blobs
    - Sky fraction is strictly 0.0%
    """
    prot_svc = ProtectionService()
    sel_svc = SelectionService()
    h, w = exterior_image_bgr.shape[:2]

    prot_mask, sky_mask, _ = prot_svc.compute_protection_masks(exterior_image_bgr)

    # 1. Polygon: Accent slab/panel
    poly_pts = [
        {"x": 0.30, "y": 0.40},
        {"x": 0.70, "y": 0.40},
        {"x": 0.70, "y": 0.48},
        {"x": 0.30, "y": 0.48},
    ]
    poly_mask = sel_svc.polygon((h, w), poly_pts, protected_mask=prot_mask, operation="replace")
    assert np.any(poly_mask > 0)
    assert prot_svc.calculate_sky_fraction(poly_mask, sky_mask) == 0.0

    # 2. Rectangle: Front gate/trim on lower building
    rect_box = {"x1": 0.45, "y1": 0.48, "x2": 0.65, "y2": 0.58}
    rect_mask = sel_svc.rectangle((h, w), rect_box, protected_mask=prot_mask, operation="replace")
    assert np.any(rect_mask > 0)
    assert prot_svc.calculate_sky_fraction(rect_mask, sky_mask) == 0.0

    # 2b. Rectangle over ground/road: MUST be completely rejected by auto-protect
    ground_box = {"x1": 0.42, "y1": 0.65, "x2": 0.60, "y2": 0.85}
    ground_mask = sel_svc.rectangle((h, w), ground_box, protected_mask=prot_mask, operation="replace")
    assert int(np.sum(ground_mask > 0)) == 0, "Ground/road must be 100% blocked by auto-protect"

    # 3. Brush: Add/subtract strokes
    stroke_pts = [{"x": 0.45, "y": 0.42}, {"x": 0.55, "y": 0.42}]
    brush_mask = sel_svc.brush((h, w), stroke_pts, radius=20, protected_mask=prot_mask, operation="add")
    assert np.any(brush_mask > 0)
    assert prot_svc.calculate_sky_fraction(brush_mask, sky_mask) == 0.0


# ==============================================================================
# 3. Multi-Layer Integrity: Last-Edited Layer Wins & Zero Overlap
# ==============================================================================

def test_last_edited_layer_wins_and_zero_overlap(exterior_image_bgr):
    """
    Verify:
    1. Last-edited layer wins: active layer steals overlapping pixels from others.
    2. Overlap between any two zones is strictly 0 pixels.
    3. Sky fraction across all zones is < 1% (strictly 0.0%).
    """
    prot_svc = ProtectionService()
    sel_svc = SelectionService()
    zone_svc = ZoneMaskService()
    h, w = exterior_image_bgr.shape[:2]

    prot_mask, sky_mask, _ = prot_svc.compute_protection_masks(exterior_image_bgr)

    # Layer 1: Main (Magic Wand)
    main_sel = sel_svc.magic_wand(
        exterior_image_bgr, {"x": 0.5, "y": 0.3}, tolerance=12.0, protected_mask=prot_mask, operation="replace"
    )
    masks = {"main": main_sel, "accent": np.zeros((h, w), dtype=np.uint8), "trim_gate": np.zeros((h, w), dtype=np.uint8)}

    # Layer 2: Accent (Polygon crossing main)
    poly_pts = [
        {"x": 0.30, "y": 0.35},
        {"x": 0.70, "y": 0.35},
        {"x": 0.70, "y": 0.45},
        {"x": 0.30, "y": 0.45},
    ]
    accent_sel = sel_svc.polygon((h, w), poly_pts, protected_mask=prot_mask, operation="replace")
    masks = zone_svc.update_layer(masks, "accent", accent_sel, protected_mask=prot_mask)

    # Layer 3: Trim Gate (Rectangle crossing accent/main)
    rect_box = {"x1": 0.40, "y1": 0.40, "x2": 0.60, "y2": 0.60}
    trim_sel = sel_svc.rectangle((h, w), rect_box, protected_mask=prot_mask, operation="replace")
    masks = zone_svc.update_layer(masks, "trim_gate", trim_sel, protected_mask=prot_mask)

    # Programmatic verification of NO OVERLAP
    overlap_main_accent = int(np.sum((masks["main"] > 0) & (masks["accent"] > 0)))
    overlap_main_trim = int(np.sum((masks["main"] > 0) & (masks["trim_gate"] > 0)))
    overlap_accent_trim = int(np.sum((masks["accent"] > 0) & (masks["trim_gate"] > 0)))

    assert overlap_main_accent == 0, f"Overlap main-accent: {overlap_main_accent} px"
    assert overlap_main_trim == 0, f"Overlap main-trim: {overlap_main_trim} px"
    assert overlap_accent_trim == 0, f"Overlap accent-trim: {overlap_accent_trim} px"

    # Sky fractions across all layers
    sky_fractions = zone_svc.get_sky_fractions(masks, sky_mask)
    for layer, frac in sky_fractions.items():
        assert frac < 0.01, f"Layer {layer} sky fraction {frac} >= 1%"
        assert frac == 0.0


# ==============================================================================
# 4. Byte-Identity of Unselected Pixels Outside Zones
# ==============================================================================

def test_byte_identical_unselected_pixels(exterior_image_bgr):
    """
    Verify that when recoloring with multi-layer zones, all unselected pixels
    outside the facade/zones are 100% BYTE-IDENTICAL to the original photo.
    """
    prot_svc = ProtectionService()
    sel_svc = SelectionService()
    zone_svc = ZoneMaskService()
    recolor_svc = RecolorService()
    h, w = exterior_image_bgr.shape[:2]

    prot_mask, _, _ = prot_svc.compute_protection_masks(exterior_image_bgr)

    # Create 3 non-overlapping zones
    main_sel = sel_svc.magic_wand(
        exterior_image_bgr, {"x": 0.5, "y": 0.3}, tolerance=12.0, protected_mask=prot_mask, operation="replace"
    )
    accent_sel = sel_svc.polygon(
        (h, w),
        [{"x": 0.3, "y": 0.4}, {"x": 0.7, "y": 0.4}, {"x": 0.7, "y": 0.48}, {"x": 0.3, "y": 0.48}],
        protected_mask=prot_mask,
        operation="replace"
    )
    trim_sel = sel_svc.rectangle(
        (h, w),
        {"x1": 0.45, "y1": 0.48, "x2": 0.65, "y2": 0.58},
        protected_mask=prot_mask,
        operation="replace"
    )

    masks = {"main": main_sel, "accent": accent_sel, "trim_gate": trim_sel}
    masks = zone_svc.update_layer(masks, "main", main_sel, protected_mask=prot_mask)
    masks = zone_svc.update_layer(masks, "accent", accent_sel, protected_mask=prot_mask)
    masks = zone_svc.update_layer(masks, "trim_gate", trim_sel, protected_mask=prot_mask)

    combined_mask = zone_svc.get_combined_mask(masks)
    unselected_mask = (combined_mask == 0)
    unselected_count = int(np.sum(unselected_mask))

    assert unselected_count > 0, "Should have unselected pixels (sky, ground, etc.)"

    # Multi-layer recolor
    contexts = recolor_svc.prepare_multi_layer_context(exterior_image_bgr, masks, surface_type="exterior")
    scheme_hexes = {"main": "#FAF6EE", "accent": "#C0583A", "trim_gate": "#303437"}

    output_bgr = recolor_svc.recolor_multi_layer(
        exterior_image_bgr, contexts, scheme_hexes, facade_mask=combined_mask
    )

    # Byte-by-byte comparison on all unselected pixels
    unselected_orig = exterior_image_bgr[unselected_mask]
    unselected_out = output_bgr[unselected_mask]

    diff = np.abs(unselected_out.astype(np.int32) - unselected_orig.astype(np.int32))
    max_diff = int(np.max(diff))
    byte_identical_count = int(np.sum(np.all(diff == 0, axis=-1)))

    assert max_diff == 0, f"Unselected pixels differed! Max diff was {max_diff}"
    assert byte_identical_count == unselected_count, (
        f"Only {byte_identical_count}/{unselected_count} unselected pixels were byte-identical"
    )
    assert np.array_equal(unselected_orig, unselected_out)


# ==============================================================================
# 5. Full API Endpoints Integration Test
# ==============================================================================

def test_select_zone_api_endpoints_workflow(client, exterior_image_bgr):
    """
    Test full API workflow:
    1. Initialize session with exterior_house.jpg and meta.json.
    2. POST /api/v1/select-zone with magic_wand.
    3. POST /api/v1/select-zone with polygon.
    4. POST /api/v1/select-zone with rectangle.
    5. Verify sky fractions < 1% in API response.
    6. Verify zero layer overlap in returned masks.
    """
    import uuid
    import json
    session_id = f"test_sel_{uuid.uuid4().hex[:8]}"
    session_dir = settings.UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(session_dir / "original.jpg"), exterior_image_bgr)

    # Query 3 valid paints from DB
    db = SessionLocal()
    paints = db.query(Paint).limit(3).all()
    p_main, p_accent, p_trim = paints[0], paints[1], paints[2]
    db.close()

    meta_content = {
        "room_analysis": {
            "scene_type": "exterior",
            "building_type": "Independent House",
            "style": "Modern",
            "lighting": "bright",
            "current_wall_color_hex": "#F0EAE1"
        },
        "schemes": [
            {
                "scheme_index": 1,
                "main_id": p_main.id,
                "accent_id": p_accent.id,
                "trim_gate_id": p_trim.id,
                "reason": "Test scheme"
            }
        ]
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_content, f)

    # 1. Magic Wand on Main
    mw_resp = client.post("/api/v1/select-zone", json={
        "session_id": session_id,
        "layer": "main",
        "tool": "magic_wand",
        "point": {"x": 0.5, "y": 0.3},
        "tolerance": 12.0,
        "operation": "replace"
    })
    assert mw_resp.status_code == 200
    mw_data = mw_resp.json()
    assert mw_data["active_layer"] == "main"
    assert "sky_fraction_by_layer" in mw_data
    assert mw_data["sky_fraction_by_layer"]["main"] < 0.01
    assert "protected_mask_url" in mw_data
    assert "protected_hatched_url" in mw_data

    # 2. Polygon on Accent
    poly_resp = client.post("/api/v1/select-zone", json={
        "session_id": session_id,
        "layer": "accent",
        "tool": "polygon",
        "points": [
            {"x": 0.30, "y": 0.38},
            {"x": 0.70, "y": 0.38},
            {"x": 0.70, "y": 0.46},
            {"x": 0.30, "y": 0.46}
        ],
        "operation": "replace"
    })
    assert poly_resp.status_code == 200
    poly_data = poly_resp.json()
    assert poly_data["active_layer"] == "accent"
    assert poly_data["sky_fraction_by_layer"]["accent"] < 0.01

    # 3. Rectangle on Trim Gate
    rect_resp = client.post("/api/v1/select-zone", json={
        "session_id": session_id,
        "layer": "trim_gate",
        "tool": "rectangle",
        "box": {"x1": 0.45, "y1": 0.48, "x2": 0.65, "y2": 0.58},
        "operation": "replace"
    })
    assert rect_resp.status_code == 200
    rect_data = rect_resp.json()
    assert rect_data["active_layer"] == "trim_gate"
    assert rect_data["sky_fraction_by_layer"]["trim_gate"] < 0.01

    # Verify zero overlap between saved masks on disk
    mask_main = cv2.imread(str(session_dir / "mask_main.png"), cv2.IMREAD_GRAYSCALE)
    mask_accent = cv2.imread(str(session_dir / "mask_accent.png"), cv2.IMREAD_GRAYSCALE)
    mask_trim = cv2.imread(str(session_dir / "mask_trim_gate.png"), cv2.IMREAD_GRAYSCALE)

    assert int(np.sum((mask_main > 0) & (mask_accent > 0))) == 0
    assert int(np.sum((mask_main > 0) & (mask_trim > 0))) == 0
    assert int(np.sum((mask_accent > 0) & (mask_trim > 0))) == 0
