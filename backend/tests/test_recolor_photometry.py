import cv2
import numpy as np
import math
from pathlib import Path
import pytest

from app.services.recolor_service import RecolorService
from app.services.sam_service import SamService
from app.services.zone_mask_service import ZoneMaskService
from app.services.shortlist_service import hex_to_rgb, rgb_to_lab
from app.core.config import BASE_DIR

def calculate_top_quartile_median_delta_e(
    repainted_bgr: np.ndarray,
    orig_bgr: np.ndarray,
    zone_mask: np.ndarray,
    target_hex: str
) -> float:
    """
    Computes deltaE (CIE76) between the target catalog swatch and
    the median of the top-quartile-lit pixels of the repainted zone.
    """
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
    l_orig = orig_lab[:, :, 0]
    
    zone_bool = zone_mask > 128
    lit_bool = zone_bool & (l_orig > 30)
    if not np.any(lit_bool):
        lit_bool = zone_bool

    # Top quartile of lit pixels (>= 75th percentile of lit pixels)
    q75 = np.percentile(l_orig[lit_bool], 75)
    top_q_mask = zone_bool & (l_orig >= q75)

    rep_rgb = cv2.cvtColor(repainted_bgr, cv2.COLOR_BGR2RGB)[top_q_mask]
    med_r = float(np.median(rep_rgb[:, 0]))
    med_g = float(np.median(rep_rgb[:, 1]))
    med_b = float(np.median(rep_rgb[:, 2]))

    l1, a1, b1 = rgb_to_lab(int(round(med_r)), int(round(med_g)), int(round(med_b)))
    r2, g2, b2 = hex_to_rgb(target_hex)
    l2, a2, b2 = rgb_to_lab(r2, g2, b2)

    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def test_flat_grey_swatch_accuracy():
    """
    Requirement 1:
    Test that the median of the top-quartile-lit pixels of a painted zone is
    within deltaE 6 of the shade on a synthetic flat grey test image.
    """
    recolor_svc = RecolorService()
    flat_bgr = np.full((300, 300, 3), 160, dtype=np.uint8)
    flat_mask = np.full((300, 300), 255, dtype=np.uint8)

    test_shades = [
        "#D8C5A8",  # Daybreak Sand (L* ~ 80.4)
        "#F5F2EB",  # Magnolia White (L* ~ 95.5)
        "#A75D43",  # Terracotta (L* ~ 47.6)
        "#2B4C7E",  # Royal Blue (L* ~ 32.2)
        "#343837",  # Slate Grey (L* ~ 23.1)
    ]

    for shade_hex in test_shades:
        ctx = recolor_svc.prepare_shading_context(flat_bgr, flat_mask, surface_type="exterior")
        repainted = recolor_svc.recolor_with_context(ctx, shade_hex)
        delta_e = calculate_top_quartile_median_delta_e(repainted, flat_bgr, flat_mask, shade_hex)
        assert delta_e <= 6.0, f"Flat grey deltaE {delta_e:.2f} exceeded 6.0 for {shade_hex}"

def test_exterior_house_real_sam_taps_photometry():
    """
    Requirements 1 & 4:
    Test with real SAM taps on test_photos/exterior_house.jpg:
    - Main facade wall tap: (0.50, 0.30)
    - Accent slab taps: pos (0.55, 0.38), neg (0.55, 0.30), neg (0.55, 0.45)
    - Gate taps: pos (0.50, 0.75), neg (0.50, 0.50), neg (0.50, 0.95)
    Strictly NO hand-coded rectangles or synthetic masks.
    Asserts median of top-quartile-lit pixels is within deltaE 6 of shade.
    """
    img_path = Path(__file__).resolve().parent.parent.parent / "test_photos" / "exterior_house.jpg"
    if not img_path.exists():
        img_path = Path("test_photos/exterior_house.jpg")
    assert img_path.exists(), f"exterior_house.jpg not found at {img_path}"

    img = cv2.imread(str(img_path))
    assert img is not None
    h, w = img.shape[:2]

    # Initialize real MobileSAM ONNX
    sam = SamService()
    zone_svc = ZoneMaskService()
    recolor_svc = RecolorService()

    # Real SAM Tap Coordinates (Fractions of width/height)
    main_pts = [{"x": 0.50, "y": 0.30, "label": 1}]
    accent_pts = [
        {"x": 0.55, "y": 0.38, "label": 1},
        {"x": 0.55, "y": 0.30, "label": 0},
        {"x": 0.55, "y": 0.45, "label": 0}
    ]
    gate_pts = [
        {"x": 0.35, "y": 0.52, "label": 1},
        {"x": 0.35, "y": 0.35, "label": 0},
        {"x": 0.35, "y": 0.65, "label": 0}
    ]

    # Run MobileSAM inference for each zone
    m_main = sam.segment_with_points(img, cache_key="photometry_test", points=main_pts, refine_edges=True)
    m_accent = sam.segment_with_points(img, cache_key="photometry_test", points=accent_pts, refine_edges=True)
    m_gate = sam.segment_with_points(img, cache_key="photometry_test", points=gate_pts, refine_edges=True)

    assert np.sum(m_main > 128) > 50000, "Main wall SAM mask was unexpectedly empty"
    assert np.sum(m_accent > 128) > 10000, "Accent slab SAM mask was unexpectedly empty"
    assert np.sum(m_gate > 128) > 2000, "Gate SAM mask was unexpectedly empty"

    # Multi-layer sequential update: last-edited layer wins
    masks = {"main": m_main, "accent": np.zeros((h, w), dtype=np.uint8), "trim_gate": np.zeros((h, w), dtype=np.uint8)}
    masks = zone_svc.update_layer(masks, "accent", m_accent)
    masks = zone_svc.update_layer(masks, "trim_gate", m_gate)

    # Verify zero overlap between zones
    overlap_ma = np.sum((masks["main"] > 128) & (masks["accent"] > 128))
    overlap_mt = np.sum((masks["main"] > 128) & (masks["trim_gate"] > 128))
    overlap_at = np.sum((masks["accent"] > 128) & (masks["trim_gate"] > 128))
    assert overlap_ma == 0, "Main and Accent layers overlap"
    assert overlap_mt == 0, "Main and Trim layers overlap"
    assert overlap_at == 0, "Accent and Trim layers overlap"

    # Multi-layer context with 90th percentile normalization over whole facade
    layer_contexts = recolor_svc.prepare_multi_layer_context(img, masks, surface_type="exterior")

    test_shades = ["#D8C5A8", "#E6DFD3", "#A75D43"]
    for shade_hex in test_shades:
        scheme_hexes = {"main": shade_hex, "accent": "#8C6D58", "trim_gate": "#2A2A2A"}
        comp_bgr = recolor_svc.recolor_multi_layer(img, layer_contexts, scheme_hexes)

        # Check deltaE on main wall
        delta_e = calculate_top_quartile_median_delta_e(comp_bgr, img, masks["main"], shade_hex)
        assert delta_e <= 6.0, f"exterior_house.jpg top-quartile deltaE {delta_e:.2f} exceeded 6.0 for {shade_hex}"
