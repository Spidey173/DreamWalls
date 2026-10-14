import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.database import SessionLocal, Base, engine
from scripts.seed_db import seed_database

@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    seed_database()
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_provider" in data
    assert data["model_variant"] == "b2"

def test_cors_origins_configuration():
    # Test wildcard
    settings.CORS_ORIGINS = "*"
    assert settings.get_cors_origins() == ["*"]

    # Test comma-separated origins
    settings.CORS_ORIGINS = "http://localhost:3000, https://paintmywall.vercel.app , https://myshop.com"
    origins = settings.get_cors_origins()
    assert origins == ["http://localhost:3000", "https://paintmywall.vercel.app", "https://myshop.com"]

    # Reset
    settings.CORS_ORIGINS = "*"

def test_ensure_models_downloaded():
    from app.core.model_manager import ensure_models_downloaded
    # Calling this must not crash and should ensure all files are verified
    ensure_models_downloaded()
    assert settings.get_segformer_model_path().exists()
    assert (settings.MODEL_DIR / "mobile_sam_encoder.onnx").exists()
    assert (settings.MODEL_DIR / "mobile_sam_decoder.onnx").exists()

def test_admin_auth(client):
    # Missing password
    resp = client.get("/api/v1/admin/paints")
    assert resp.status_code == 401

    # Wrong password
    resp = client.get("/api/v1/admin/paints", headers={"X-Admin-Password": "wrong"})
    assert resp.status_code == 401

    # Correct password
    resp = client.get("/api/v1/admin/paints", headers={"X-Admin-Password": settings.ADMIN_PASSWORD})
    assert resp.status_code == 200
    paints = resp.json()
    assert len(paints) > 0

def test_admin_verify(client):
    resp = client.post("/api/v1/admin/verify", json={"password": "wrong"})
    assert resp.status_code == 401

    resp = client.post("/api/v1/admin/verify", json={"password": settings.ADMIN_PASSWORD})
    assert resp.status_code == 200
    assert resp.json()["valid"] is True

def test_analyze_invalid_mime(client):
    fake_file = io.BytesIO(b"not an image")
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert resp.status_code == 400

def test_analyze_and_recolor_flow(client):
    # Create small test room image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:180, :] = [230, 220, 210]  # wall
    img[180:, :] = [60, 80, 120]    # floor
    _, encoded = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("room.jpg", file_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "session_id" in data
    assert "room_analysis" in data
    assert "results" in data
    assert len(data["results"]) == 4

    session_id = data["session_id"]
    first_result = data["results"][0]
    assert "shade_name" in first_result
    assert "shade_code" in first_result
    assert "image_url" in first_result

    # Now test mask correction endpoint /api/v1/recolor-mask
    # Create edited mask (all white)
    import base64
    mask = np.full((300, 300), 255, dtype=np.uint8)
    _, mask_enc = cv2.imencode(".png", mask)
    b64_mask = base64.b64encode(mask_enc.tobytes()).decode("utf-8")

    recolor_resp = client.post(
        "/api/v1/recolor-mask",
        json={"session_id": session_id, "custom_mask_base64": b64_mask}
    )
    assert recolor_resp.status_code == 200
    recolor_data = recolor_resp.json()
    assert len(recolor_data["results"]) == 4
