import pytest
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.paint import Paint
from app.schemas.vision import RoomAnalysis
from app.schemas.paint import PaintCandidate
from app.services.selection_service import SelectionService
from app.services.llm.base import BaseLLMClient

class MockCustomLLMClient(BaseLLMClient):
    def __init__(self, output_json: str):
        self.output_json = output_json

    async def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        return "{}"

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.output_json

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()

    paints = [
        Paint(id=1, brand="Asian Paints", name="Morning Glory", shade_code="0942", hex="#F4EBD9", finish="Matte", color_family="Warm Neutral"),
        Paint(id=2, brand="Asian Paints", name="Canvas Cream", shade_code="L102", hex="#F8F2E4", finish="Eggshell", color_family="Warm Neutral"),
        Paint(id=3, brand="Asian Paints", name="Misty Grey", shade_code="8297", hex="#E0E2E1", finish="Matte", color_family="Cool Neutral"),
        Paint(id=4, brand="Asian Paints", name="Sea Breeze", shade_code="7348", hex="#D0E8EB", finish="Sheen", color_family="Cool Blue"),
        Paint(id=5, brand="Asian Paints", name="Sage Whisper", shade_code="7729", hex="#C8D2BF", finish="Sheen", color_family="Earthy Green"),
    ]
    db.add_all(paints)
    db.commit()
    yield db
    db.close()

@pytest.fixture
def sample_candidates():
    return [
        PaintCandidate(id=1, name="Morning Glory", shade_code="0942", hex="#F4EBD9", finish="Matte", color_family="Warm Neutral", harmony_category="Neutral", score=90.0),
        PaintCandidate(id=2, name="Canvas Cream", shade_code="L102", hex="#F8F2E4", finish="Eggshell", color_family="Warm Neutral", harmony_category="Neutral", score=88.0),
        PaintCandidate(id=3, name="Misty Grey", shade_code="8297", hex="#E0E2E1", finish="Matte", color_family="Cool Neutral", harmony_category="Neutral", score=85.0),
        PaintCandidate(id=4, name="Sea Breeze", shade_code="7348", hex="#D0E8EB", finish="Sheen", color_family="Cool Blue", harmony_category="Analogous", score=80.0),
        PaintCandidate(id=5, name="Sage Whisper", shade_code="7729", hex="#C8D2BF", finish="Sheen", color_family="Earthy Green", harmony_category="Complementary", score=78.0),
    ]

@pytest.fixture
def sample_room():
    return RoomAnalysis(
        room_type="living room",
        lighting="bright",
        current_wall_color_hex="#F4EBD9",
        floor_color_hex="#8A5A36",
        dominant_furniture_colors=["#1E3A52"],
        style="Contemporary",
        mood="Inviting"
    )

@pytest.mark.asyncio
async def test_valid_ids_accepted(test_db, sample_candidates, sample_room):
    llm_output = json.dumps({
        "selections": [
            {"id": 1, "reason": "Warm and neutral tone."},
            {"id": 2, "reason": "Brings daylight in."},
            {"id": 4, "reason": "Fresh seaside tone."},
            {"id": 5, "reason": "Subtle green accent."}
        ]
    })
    client = MockCustomLLMClient(llm_output)
    service = SelectionService(llm_client=client)

    results = await service.select_shades(test_db, sample_room, sample_candidates)
    assert len(results) == 4
    result_ids = [p.id for p, r in results]
    assert result_ids == [1, 2, 4, 5]

@pytest.mark.asyncio
async def test_hallucinated_ids_dropped_and_replaced(test_db, sample_candidates, sample_room):
    # LLM hallucinates non-existent IDs 999 and -5
    llm_output = json.dumps({
        "selections": [
            {"id": 1, "reason": "Valid shade 1."},
            {"id": 999, "reason": "Hallucinated shade!"},
            {"id": -5, "reason": "Negative hallucinated shade!"},
            {"id": 3, "reason": "Valid shade 3."}
        ]
    })
    client = MockCustomLLMClient(llm_output)
    service = SelectionService(llm_client=client)

    results = await service.select_shades(test_db, sample_room, sample_candidates)
    assert len(results) == 4
    result_ids = [p.id for p, r in results]

    # 999 and -5 must NOT be in the results
    assert 999 not in result_ids
    assert -5 not in result_ids

    # All 4 returned shades must exist in test_db
    for p, reason in results:
        assert p.id in {1, 2, 3, 4, 5}
        assert p.name is not None
        assert p.shade_code is not None

@pytest.mark.asyncio
async def test_duplicate_ids_handled(test_db, sample_candidates, sample_room):
    llm_output = json.dumps({
        "selections": [
            {"id": 1, "reason": "Reason A"},
            {"id": 1, "reason": "Reason B (duplicate)"},
            {"id": 2, "reason": "Reason C"},
            {"id": 3, "reason": "Reason D"}
        ]
    })
    client = MockCustomLLMClient(llm_output)
    service = SelectionService(llm_client=client)

    results = await service.select_shades(test_db, sample_room, sample_candidates)
    assert len(results) == 4
    result_ids = [p.id for p, r in results]
    # No duplicate IDs
    assert len(set(result_ids)) == 4
