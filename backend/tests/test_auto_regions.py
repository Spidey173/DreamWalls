import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.auto_regions_service import AutoRegionsService, CandidateRegion
from app.services.protection_service import ProtectionService
from app.core.config import settings


@pytest.fixture
def test_image_bgr():
    root_dir = Path(__file__).resolve().parent.parent.parent
    path = root_dir / "test_photos" / "exterior_house.jpg"
    assert path.exists()
    img = cv2.imread(str(path))
    assert img is not None
    return img


def test_auto_regions_service_unit():
    # Test CandidateRegion and find_region_at_tap
    cand1 = CandidateRegion(
        region_id=1,
        mask=np.zeros((100, 100), dtype=np.uint8),
        pixel_count=50,
        bbox=(20, 20, 30, 30),
        centroid=(0.25, 0.25)
    )
    cand1.mask[20:30, 20:30] = 255

    cand2 = CandidateRegion(
        region_id=2,
        mask=np.zeros((100, 100), dtype=np.uint8),
        pixel_count=2000,
        bbox=(10, 10, 80, 80),
        centroid=(0.45, 0.45)
    )
    cand2.mask[10:80, 10:80] = 255

    svc = AutoRegionsService()
    candidates = [cand1, cand2]  # sorted ascending by area

    # Tap at (0.25, 0.25) is inside both cand1 and cand2 -> should pick cand1 (smallest)
    picked = svc.find_region_at_tap(candidates, 0.25, 0.25, (100, 100))
    assert picked is not None
    assert picked.region_id == 1

    # Tap at (0.50, 0.50) is only inside cand2 -> should pick cand2
    picked2 = svc.find_region_at_tap(candidates, 0.50, 0.50, (100, 100))
    assert picked2 is not None
    assert picked2.region_id == 2

    # Tap outside both
    picked3 = svc.find_region_at_tap(candidates, 0.02, 0.02, (100, 100))
    assert picked3 is None


def test_auto_regions_save_and_load(tmp_path):
    cand = CandidateRegion(
        region_id=42,
        mask=np.zeros((50, 50), dtype=np.uint8),
        pixel_count=100,
        bbox=(5, 5, 15, 15),
        centroid=(0.2, 0.2)
    )
    cand.mask[5:15, 5:15] = 255

    svc = AutoRegionsService()
    svc.save_regions(tmp_path, [cand])

    loaded = svc.load_regions(tmp_path)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].region_id == 42
    assert loaded[0].pixel_count == 100
    assert np.array_equal(loaded[0].mask, cand.mask)


def test_auto_region_select_zone_api(test_image_bgr):
    with TestClient(app) as client:
        # 1. Analyze
        root_dir = Path(__file__).resolve().parent.parent.parent
        img_path = root_dir / "test_photos" / "exterior_house.jpg"
        with open(img_path, "rb") as f:
            resp = client.post("/api/v1/analyze", files={"file": ("house.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        # 2. Select zone using tool: auto_region
        sel_resp = client.post(
            "/api/v1/select-zone",
            json={
                "session_id": sid,
                "layer": "accent",
                "tool": "auto_region",
                "point": {"x": 0.15, "y": 0.35},
                "operation": "replace"
            }
        )
        assert sel_resp.status_code == 200
        data = sel_resp.json()
        assert "auto_region_count" in data
        assert data["auto_region_count"] > 0
        assert "selected_region_id" in data
        assert data["coverage_by_layer"]["accent"] > 0
