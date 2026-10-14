import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch

from app.models.paint import Paint
from app.schemas.selection import SchemeSelectionItem
from app.services.image_gen.cost_and_cache import ImageCostTracker, ImageSchemeCache
from app.services.image_gen.service import ImageSchemeService, calculate_delta_e_rgb
from app.services.image_gen.mock_client import MockImageClient
from app.services.image_gen.factory import get_image_client

def test_cost_tracker_limits_and_rates():
    tracker = ImageCostTracker()
    assert tracker.get_model_cost_per_image("gemini-2.5-flash-image") == 0.02
    assert tracker.get_model_cost_per_image("imagen-3.0-generate-002") == 0.04
    assert tracker.get_model_cost_per_image("mock") == 0.0

    client_ip = "192.168.1.100"
    allowed, msg = tracker.check_limits(client_ip)
    assert allowed is True

    tracker.record_generation(client_ip, "gemini-2.5-flash-image", count=5)
    stats = tracker.get_stats()
    assert stats["global_count"] == 5

    with patch("app.core.config.settings.IMAGE_PER_IP_DAILY_LIMIT", 5):
        allowed, msg = tracker.check_limits(client_ip)
        assert allowed is False
        assert "Per-IP daily limit" in msg

    with patch("app.core.config.settings.IMAGE_GLOBAL_DAILY_CAP", 5):
        allowed, msg = tracker.check_limits("another.ip")
        assert allowed is False
        assert "Global daily cap" in msg

def test_scheme_cache(tmp_path):
    cache = ImageSchemeCache(cache_dir=tmp_path)
    dummy_bytes = b"fake_jpeg_bytes"
    key = cache.compute_cache_key(dummy_bytes, 1, 2, 3, "gemini-2.5-flash-image")

    assert cache.get(key) is None

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[:] = [100, 150, 200]
    meta = {"delta_e": 4.5, "drift_outside_pct": 1.2}
    cache.set(key, dummy_img, meta)

    hit = cache.get(key)
    assert hit is not None
    loaded_img, loaded_meta = hit
    assert loaded_img.shape == (100, 100, 3)
    assert loaded_meta["delta_e"] == 4.5

def test_factory_production_vs_mock_modes():
    # Production mode: no API key and allow_mock=False -> must raise ValueError
    with pytest.raises(ValueError) as excinfo:
        get_image_client(model="gemini-2.5-flash-image", api_key="", allow_mock=False)
    assert "IMAGE_API_KEY is not configured and IMAGE_ALLOW_MOCK is false" in str(excinfo.value)

    # Demo mode: allow_mock=True -> returns MockImageClient with is_mock=True
    client = get_image_client(model="gemini-2.5-flash-image", api_key="", allow_mock=True)
    assert isinstance(client, MockImageClient)
    assert client.is_mock is True

def test_mock_image_client_returns_original_with_banner():
    client = MockImageClient()
    assert client.is_mock is True

    # Create dummy original photo
    orig_img = np.full((300, 400, 3), 120, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", orig_img)

    prompt = "Repaint this building's exterior..."
    import asyncio
    out_bytes = asyncio.run(client.generate_scheme_image(enc.tobytes(), "image/jpeg", prompt))
    assert len(out_bytes) > 0

    decoded = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (300, 400, 3)
    # The banner modifies the center region
    center_y = int(300 * 0.45)
    assert not np.array_equal(decoded[center_y, 200], orig_img[center_y, 200])
    # The corner outside the banner is the original photo
    assert np.all(np.abs(decoded[10, 10].astype(int) - orig_img[10, 10].astype(int)) <= 5)

def test_prompt_building():
    svc = ImageSchemeService()
    p_main = Paint(id=1, name="Magnolia White", shade_code="0314", hex="#F2EBD9")
    p_accent = Paint(id=2, name="Terracotta Glow", shade_code="0524", hex="#C15C3D")
    p_trim = Paint(id=3, name="Midnight Navy", shade_code="9210", hex="#1B2838")

    prompt = svc.build_prompt(p_main, p_accent, p_trim)
    assert "Repaint this building's exterior." in prompt
    assert "Main walls: Magnolia White (#F2EBD9)." in prompt
    assert "Projecting slabs, window surrounds, side panels and tower: Terracotta Glow (#C15C3D)." in prompt
    assert "Gate: Midnight Navy (#1B2838)." in prompt
    assert "Keep the structure, sky, road, ground, vegetation and text unchanged." in prompt
    assert "Photorealistic." in prompt

def test_composite_with_mask_preserves_background():
    svc = ImageSchemeService()
    h, w = 200, 200
    orig = np.full((h, w, 3), 50, dtype=np.uint8)
    gen = np.full((h, w, 3), 200, dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    composite, mean_delta, pct_changed = svc.composite_with_mask(orig, gen, mask, s_idx=1)
    assert np.all(composite[mask == 0] == 50)
    assert np.all(composite[mask > 0] == 200)
    assert mean_delta == 150.0
    assert pct_changed == 100.0

@pytest.mark.asyncio
async def test_production_fallback_when_no_api_key(tmp_path):
    svc = ImageSchemeService()
    h, w = 150, 150
    orig_bgr = np.full((h, w, 3), 120, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", orig_bgr)
    orig_bytes = enc.tobytes()

    zone_masks = {
        "main": np.zeros((h, w), dtype=np.uint8),
        "accent": np.zeros((h, w), dtype=np.uint8),
        "trim_gate": np.zeros((h, w), dtype=np.uint8),
    }
    zone_masks["main"][30:120, 30:120] = 255

    p1 = Paint(id=1, name="Magnolia White", shade_code="0314", hex="#F2EBD9", finish="Matt", color_family="White")
    p2 = Paint(id=2, name="Terracotta Glow", shade_code="0524", hex="#C15C3D", finish="Matt", color_family="Orange")
    p3 = Paint(id=3, name="Midnight Navy", shade_code="9210", hex="#1B2838", finish="Gloss", color_family="Blue")
    all_paints = {1: p1, 2: p2, 3: p3}

    schemes = [
        SchemeSelectionItem(main_id=1, accent_id=2, trim_gate_id=3, reason="Test scheme 1"),
    ]

    # In production mode (IMAGE_ALLOW_MOCK=False, no IMAGE_API_KEY)
    with patch("app.core.config.settings.IMAGE_ALLOW_MOCK", False), \
         patch("app.core.config.settings.IMAGE_API_KEY", ""):
        results, mode, fallback_reason = await svc.render_schemes_parallel(
            original_bgr=orig_bgr,
            cropped_image_bytes=orig_bytes,
            zone_masks=zone_masks,
            selected_schemes=schemes,
            all_paints=all_paints,
            client_ip="127.0.0.1",
            session_dir=tmp_path
        )
        # Must fall back to precise mode
        assert mode == "precise"
        assert fallback_reason is not None
        assert results[0].status == "fallback"
        assert results[0].mode == "precise"
        assert results[0].mock is False

@pytest.mark.asyncio
async def test_demo_mock_mode_skips_metrics(tmp_path):
    svc = ImageSchemeService()
    h, w = 150, 150
    orig_bgr = np.full((h, w, 3), 120, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", orig_bgr)
    orig_bytes = enc.tobytes()

    zone_masks = {
        "main": np.zeros((h, w), dtype=np.uint8),
        "accent": np.zeros((h, w), dtype=np.uint8),
        "trim_gate": np.zeros((h, w), dtype=np.uint8),
    }
    zone_masks["main"][30:120, 30:120] = 255

    p1 = Paint(id=1, name="Magnolia White", shade_code="0314", hex="#F2EBD9", finish="Matt", color_family="White")
    p2 = Paint(id=2, name="Terracotta Glow", shade_code="0524", hex="#C15C3D", finish="Matt", color_family="Orange")
    p3 = Paint(id=3, name="Midnight Navy", shade_code="9210", hex="#1B2838", finish="Gloss", color_family="Blue")
    all_paints = {1: p1, 2: p2, 3: p3}

    schemes = [
        SchemeSelectionItem(main_id=1, accent_id=2, trim_gate_id=3, reason="Test scheme 1"),
    ]

    # In demo mode (IMAGE_ALLOW_MOCK=True and no key)
    with patch("app.core.config.settings.IMAGE_ALLOW_MOCK", True), patch("app.core.config.settings.IMAGE_API_KEY", ""):
        results, mode, fallback_reason = await svc.render_schemes_parallel(
            original_bgr=orig_bgr,
            cropped_image_bytes=orig_bytes,
            zone_masks=zone_masks,
            selected_schemes=schemes,
            all_paints=all_paints,
            client_ip="127.0.0.1",
            session_dir=tmp_path
        )
        assert results[0].mock is True
        # deltaE and drift metrics must be skipped (None)
        assert results[0].delta_e is None
        assert results[0].drift_outside_pct is None
