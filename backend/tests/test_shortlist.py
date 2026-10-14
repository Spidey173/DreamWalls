import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.paint import Paint
from app.schemas.vision import RoomAnalysis
from app.services.shortlist_service import (
    ShortlistService,
    hex_to_rgb,
    rgb_to_hsl,
    rgb_to_lab,
    circular_hue_diff,
)

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()

    # Seed varied paints for testing
    sample_paints = [
        # Neutrals
        Paint(brand="Asian Paints", name="Morning Glory", shade_code="0942", hex="#F4EBD9", finish="Matte", color_family="Warm Neutral"),
        Paint(brand="Asian Paints", name="Canvas Cream", shade_code="L102", hex="#F8F2E4", finish="Eggshell", color_family="Warm Neutral"),
        Paint(brand="Asian Paints", name="Misty Grey", shade_code="8297", hex="#E0E2E1", finish="Matte", color_family="Cool Neutral"),
        # Blues
        Paint(brand="Asian Paints", name="Sea Breeze", shade_code="7348", hex="#D0E8EB", finish="Sheen", color_family="Cool Blue"),
        Paint(brand="Asian Paints", name="Midnight Navy", shade_code="7370", hex="#1E3A52", finish="Matte", color_family="Cool Blue"),
        # Greens
        Paint(brand="Asian Paints", name="Sage Whisper", shade_code="7729", hex="#C8D2BF", finish="Sheen", color_family="Earthy Green"),
        Paint(brand="Asian Paints", name="Pine Needle", shade_code="7698", hex="#4B5E43", finish="Matte", color_family="Earthy Green"),
        # Yellows / Terracottas
        Paint(brand="Asian Paints", name="Golden Ochre", shade_code="7895", hex="#EDB858", finish="Matte", color_family="Warm Yellow"),
        Paint(brand="Asian Paints", name="Terracotta Glow", shade_code="0512", hex="#C96849", finish="Matte", color_family="Terracotta & Red"),
    ]
    db.add_all(sample_paints)
    db.commit()
    yield db
    db.close()

def test_color_space_conversions():
    rgb = hex_to_rgb("#F4EBD9")
    assert rgb == (244, 235, 217)

    h, s, l = rgb_to_hsl(*rgb)
    assert 0 <= h <= 360
    assert 0 <= s <= 1.0
    assert 0 <= l <= 1.0

    l_star, a_star, b_star = rgb_to_lab(*rgb)
    assert 0 <= l_star <= 100

def test_circular_hue_diff():
    assert circular_hue_diff(10, 20) == 10
    assert circular_hue_diff(10, 350) == 20
    assert circular_hue_diff(0, 180) == 180

def test_dim_lighting_filters_dark_shades(test_db):
    service = ShortlistService()
    dim_room = RoomAnalysis(
        room_type="bedroom",
        lighting="dim",
        current_wall_color_hex="#EAE6DF",
        floor_color_hex="#8A5A36",
        dominant_furniture_colors=["#1E3A52"],
        style="Modern",
        mood="Cozy"
    )

    candidates = service.shortlist_shades(test_db, dim_room, min_count=2, max_count=10)
    # Midnight Navy (#1E3A52, L* ~ 25) must NOT be shortlisted in a dim room
    candidate_codes = [c.shade_code for c in candidates]
    assert "7370" not in candidate_codes
    # Lighter shades should be present
    assert "0942" in candidate_codes or "L102" in candidate_codes

def test_shortlist_harmony_diversity(test_db):
    service = ShortlistService()
    bright_room = RoomAnalysis(
        room_type="living room",
        lighting="bright",
        current_wall_color_hex="#F0EDE5",
        floor_color_hex="#5C4033",
        dominant_furniture_colors=["#D0E8EB"],
        style="Contemporary",
        mood="Airy"
    )

    candidates = service.shortlist_shades(test_db, bright_room, min_count=4, max_count=10)
    assert len(candidates) >= 4
    categories = {c.harmony_category for c in candidates}
    # Should include diverse harmony groups
    assert "Neutral" in categories
