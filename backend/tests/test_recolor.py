import numpy as np
import cv2
import pytest
from app.services.recolor_service import RecolorService

@pytest.fixture
def recolor_service():
    return RecolorService()

@pytest.fixture
def synthetic_room_image():
    # 200x200 test image with a gradient wall (top 120 rows) and floor (bottom 80 rows)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Wall gradient (shading from L=140 to L=180)
    for y in range(120):
        val = int(140 + (y / 120.0) * 40)
        img[y, :] = [val, val, val]
    # Floor (dark brown)
    img[120:, :] = [40, 60, 90]
    return img

@pytest.fixture
def synthetic_wall_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Mask corresponds to the wall (top 120 rows)
    mask[:120, :] = 255
    return mask

def test_recolor_output_shape_and_type(recolor_service, synthetic_room_image, synthetic_wall_mask):
    repainted = recolor_service.recolor(synthetic_room_image, synthetic_wall_mask, "#7348", feather_radius=3)
    assert repainted.shape == synthetic_room_image.shape
    assert repainted.dtype == np.uint8

def test_unmasked_pixels_preserved(recolor_service, synthetic_room_image, synthetic_wall_mask):
    # Target shade: Sea Breeze #D0E8EB
    repainted = recolor_service.recolor(synthetic_room_image, synthetic_wall_mask, "#D0E8EB", feather_radius=3)
    # Floor area (bottom 50 rows) is far from the mask boundary and should be unchanged
    floor_orig = synthetic_room_image[150:, :]
    floor_repainted = repainted[150:, :]
    np.testing.assert_array_equal(floor_orig, floor_repainted)

def test_masked_wall_colors_shifted(recolor_service, synthetic_room_image, synthetic_wall_mask):
    # Repaint with a warm terracotta #C96849
    repainted = recolor_service.recolor(synthetic_room_image, synthetic_wall_mask, "#C96849", feather_radius=3)
    wall_orig = synthetic_room_image[20:100, 20:180]
    wall_repainted = repainted[20:100, 20:180]

    # Masked wall must change significantly
    assert not np.array_equal(wall_orig, wall_repainted)

    # In BGR, terracotta has higher Red than Blue/Green
    mean_bgr = np.mean(wall_repainted, axis=(0, 1))
    # B, G, R
    assert mean_bgr[2] > mean_bgr[0]  # Red channel should be higher than Blue channel

def test_shadow_and_texture_variation_preserved(recolor_service, synthetic_room_image, synthetic_wall_mask):
    # Wall in synthetic_room_image has a gradient across Y (standard deviation > 0)
    repainted = recolor_service.recolor(synthetic_room_image, synthetic_wall_mask, "#D0E8EB", feather_radius=3)
    repainted_lab = cv2.cvtColor(repainted, cv2.COLOR_BGR2LAB)
    wall_l = repainted_lab[10:100, 10:190, 0]

    # Texture/shading variance must be preserved (not flattened to a uniform solid slab)
    l_std = np.std(wall_l)
    assert l_std > 3.0, "Lightness variance (shadows/textures) was flattened!"

def test_mask_feathering_is_smooth(recolor_service, synthetic_wall_mask):
    feathered = recolor_service.feather_mask(synthetic_wall_mask, blur_radius=7)
    assert feathered.dtype == np.float32
    assert 0.0 <= np.min(feathered) <= 0.05
    assert 0.95 <= np.max(feathered) <= 1.0
    # Values around y=120 (boundary) should be between 0.1 and 0.9
    boundary_values = feathered[118:122, 100]
    assert np.any((boundary_values > 0.05) & (boundary_values < 0.95))
