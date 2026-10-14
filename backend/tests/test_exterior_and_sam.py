import io
import json
import base64
import logging
import pytest
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.database import SessionLocal, Base, engine
from scripts.seed_db import seed_database
from app.services.segmentation_service import SegmentationService
from app.services.sam_service import SamService


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    seed_database()
    with TestClient(app) as c:
        yield c


# ==========================================
# 1. Exterior Class-List Logic Tests
# ==========================================

def test_resolve_class_ids_standard_and_case_insensitive():
    seg_service = SegmentationService()
    
    # Check that standard classes resolve
    ids = seg_service._resolve_class_ids(["wall", "building", "sky"])
    assert len(ids) == 3
    assert seg_service.label2id["wall"] in ids
    assert seg_service.label2id["building"] in ids
    assert seg_service.label2id["sky"] in ids

    # Check case-insensitivity and whitespace trimming
    ids_case = seg_service._resolve_class_ids(["  WALL  ", "SkY "])
    assert len(ids_case) == 2
    assert ids_case[0] == seg_service.label2id["wall"]
    assert ids_case[1] == seg_service.label2id["sky"]


def test_resolve_class_ids_skips_missing_without_crashing(caplog):
    seg_service = SegmentationService()

    with caplog.at_level(logging.WARNING):
        resolved = seg_service._resolve_class_ids([
            "wall",
            "completely_nonexistent_class_xyz",
            "gate",
            "building"
        ])

    # Should only return valid IDs for wall and building
    assert len(resolved) == 2
    assert seg_service.label2id["wall"] in resolved
    assert seg_service.label2id["building"] in resolved

    # Should log warnings for missing classes
    assert any("completely_nonexistent_class_xyz" in record.message for record in caplog.records)
    assert any("gate" in record.message for record in caplog.records)


def test_exterior_segmentation_subtracts_protected_classes():
    """
    Synthesize an image containing sky-like blue at top, building-like structure in middle,
    and road-like ground at bottom. Verify segment_wall handles surface_type='exterior'.
    """
    seg_service = SegmentationService()
    
    h, w = 300, 300
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky region
    test_img[:100, :] = [235, 206, 135]  # Sky blue in BGR
    # Building region
    test_img[100:220, :] = [180, 180, 180]  # Light grey
    # Road region
    test_img[220:, :] = [50, 50, 50]  # Dark grey asphalt

    _, encoded = cv2.imencode(".jpg", test_img)
    mask, _ = seg_service.detect_walls(encoded.tobytes(), surface_type="exterior", force_recompute=True)

    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert mask.min() >= 0 and mask.max() <= 255
    # The mask should have valid pixel values in [0, 255]
    assert np.any(mask > 0)


# ==========================================
# 2. SAM Service & Tap-Segment Endpoint Tests
# ==========================================

def test_sam_service_points_segmentation_and_caching():
    from app.services.sam_service import _embedding_cache
    sam = SamService()
    
    # 200x200 image with a white square in the center
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = 255

    points = [
        {"x": 0.5, "y": 0.5, "label": 1},  # Positive point on the square
        {"x": 0.1, "y": 0.1, "label": 0},  # Negative point outside
    ]

    cache_key = "test_sam_cache_session"
    mask1 = sam.segment_with_points(img, cache_key=cache_key, points=points, refine_edges=False)
    
    assert mask1.shape == (200, 200)
    assert mask1[100, 100] == 255  # Selected point is inside mask
    assert mask1[10, 10] == 0      # Background point is excluded

    # Verify embedding is cached
    assert cache_key in _embedding_cache

    # Second call uses cached embedding
    mask2 = sam.segment_with_points(img, cache_key=cache_key, points=points, refine_edges=False)
    assert np.array_equal(mask1, mask2)


def test_tap_segment_endpoint_success_and_multi_point(client):
    # Setup a mock session with original.jpg
    import uuid
    session_id = f"test_tap_{uuid.uuid4().hex[:8]}"
    session_dir = settings.UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:180, :] = [210, 215, 220]  # Light wall
    img[180:, :] = [70, 90, 120]    # Dark floor
    cv2.imwrite(str(session_dir / "original.jpg"), img)

    # 1. Single positive point
    payload_single = {
        "session_id": session_id,
        "points": [{"x": 0.5, "y": 0.3, "label": 1}]
    }
    resp = client.post("/api/v1/tap-segment", json=payload_single)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert "mask_url" in data
    assert data["mask_url"].startswith("data:image/png;base64,")
    assert "coverage_pct" in data
    assert 0 <= data["coverage_pct"] <= 100

    # Verify mask.png was saved to disk
    saved_mask = cv2.imread(str(session_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert saved_mask is not None
    assert saved_mask.shape == (300, 300)

    # 2. Multi-point with positive and negative points
    payload_multi = {
        "session_id": session_id,
        "points": [
            {"x": 0.5, "y": 0.3, "label": 1},
            {"x": 0.5, "y": 0.8, "label": 0}
        ]
    }
    resp_multi = client.post("/api/v1/tap-segment", json=payload_multi)
    assert resp_multi.status_code == 200
    data_multi = resp_multi.json()
    assert "mask_url" in data_multi


def test_tap_segment_endpoint_not_found(client):
    resp = client.post("/api/v1/tap-segment", json={
        "session_id": "nonexistent_session_99999",
        "points": [{"x": 0.5, "y": 0.5, "label": 1}]
    })
    assert resp.status_code == 404
