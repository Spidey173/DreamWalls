import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
import logging
from sqlalchemy.orm import Session

from app.models.paint import Paint
from app.schemas.vision import RoomAnalysis
from app.schemas.paint import PaintCandidate
from app.schemas.selection import ShadeSelectionResponse, ShadeSelectionItem
from app.services.llm.base import BaseLLMClient
from app.services.llm.factory import get_llm_client
from app.services.vision_service import extract_json
from app.services.protection_service import ProtectionService
from app.services.sam_service import SamService

logger = logging.getLogger("paintmywall.selection")

SELECTION_SYSTEM_PROMPT = """You are an expert interior designer specializing in Asian Paints shade palettes.
Your goal is to select the 4 finest paint shades for the user's room from ONLY the provided candidate list.
Provide exactly 4 recommendations offering distinct decorative moods (e.g. 1 crisp/serene neutral, 1 harmonious warm/earthy shade, 1 expressive accent or complementary tone, 1 contemporary balanced tone).

CRITICAL CONSTRAINTS:
1. You MUST ONLY pick 'id' numbers that exist in the provided candidate list.
2. Output STRICT JSON only:
{
  "selections": [
    {"id": <integer_id>, "reason": "<one line explaining why this shade suits the room and lighting>"},
    ...
  ]
}
3. Do NOT output markdown or commentary outside the JSON.
"""

class SelectionService:
    """
    Unified Selection Tools Engine for Architectural Exterior Recolor & Interior Shade Selection:
    1. Magic Wand (Default tool): LAB flood-fill with tolerance slider, restricted to non-protected pixels.
    2. Polygon: Click corners, close shape.
    3. Rectangle: Box selection.
    4. Brush: Add / subtract circular brush path.
    5. Intelligent Scissors: Edge-snapped contour via cv2.segmentation.IntelligentScissorsMB.
    6. MobileSAM (Optional tool): AI point-prompt segmentation.
    7. select_shades: LLM shade recommendation from verified candidates (interior pipeline).

    All mask tools:
    - Restrict selections to non-protected pixels (sky, ground, veg, vehicles auto-subtracted).
    - Run post-selection cleanup: fill holes, remove blobs < 0.1% area, smooth edges lightly.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm_client = llm_client
        self.protection_service = ProtectionService()
        self.sam_service = SamService()

    @property
    def llm_client(self) -> BaseLLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def select_shades(
        self,
        db: Session,
        room: RoomAnalysis,
        candidates: List[PaintCandidate]
    ) -> List[Tuple[Paint, str]]:
        """
        Sends only shortlist candidates to LLM for final 4 selection.
        Verifies every returned ID against the database, dropping/replacing any invalid IDs.
        NEVER lets the AI invent shade names or codes.
        """
        if not candidates:
            return []

        candidate_payload = [
            {
                "id": c.id,
                "name": c.name,
                "code": c.shade_code,
                "hex": c.hex,
                "family": c.color_family,
                "category": c.harmony_category
            }
            for c in candidates
        ]

        if room.scene_type == "exterior":
            user_prompt = f"""Exterior Architectural Analysis:
- Scene Type: Exterior
- Building Type: {room.building_type or 'House'}
- Natural Lighting: {room.light_conditions or room.lighting}
- Current Facade Color: {room.current_wall_color_hex}
- Surroundings: {room.surroundings or 'Street / garden environment'}
- Trim & Accent Tones: {', '.join(room.dominant_furniture_colors) or 'Neutral'}
- Style: {room.style}
- Desired Mood: {room.mood}

Available Shortlisted Asian Paints Candidates:
{json.dumps(candidate_payload, indent=2)}

Select the 4 best exterior paint shades offering distinct architectural moods.
Remember: ONLY return 'id' values from the list above with a 1-line reason each.
"""
        else:
            user_prompt = f"""Room Analysis:
- Scene Type: Interior
- Room Type: {room.room_type}
- Lighting: {room.lighting}
- Current Wall: {room.current_wall_color_hex}
- Flooring: {room.floor_color_hex}
- Dominant Furniture Accents: {', '.join(room.dominant_furniture_colors)}
- Style: {room.style}
- Desired Mood: {room.mood}

Available Shortlisted Asian Paints Candidates:
{json.dumps(candidate_payload, indent=2)}

Select the 4 best shades with a 1-line reason each.
Remember: ONLY return 'id' values from the list above.
"""

        raw_output = await self.llm_client.generate_text(
            prompt=user_prompt,
            system_prompt=SELECTION_SYSTEM_PROMPT
        )

        parsed_items: List[ShadeSelectionItem] = []
        try:
            cleaned = extract_json(raw_output)
            data = json.loads(cleaned)
            parsed_response = ShadeSelectionResponse.model_validate(data)
            parsed_items = parsed_response.selections
        except Exception as e:
            logger.warning(f"Failed to parse LLM shade selection response: {e}. Falling back to deterministic selection.")

        candidate_dict: Dict[int, PaintCandidate] = {c.id: c for c in candidates}
        shortlist_ids = set(candidate_dict.keys())

        db_paints = {
            p.id: p for p in db.query(Paint).filter(Paint.id.in_(shortlist_ids)).all()
        }

        verified_results: List[Tuple[Paint, str]] = []
        seen_ids = set()

        for item in parsed_items:
            if item.id in db_paints and item.id not in seen_ids:
                seen_ids.add(item.id)
                paint = db_paints[item.id]
                if room.scene_type == "exterior":
                    clean_reason = item.reason.strip() if item.reason else f"Enhances the {room.building_type or 'building'}'s {room.style} facade."
                else:
                    clean_reason = item.reason.strip() if item.reason else f"Complements the {room.room_type}'s {room.style} decor."
                verified_results.append((paint, clean_reason))
                if len(verified_results) == 4:
                    break
            else:
                logger.warning(f"Dropped invalid/hallucinated shade ID from LLM response: {item.id}")

        if len(verified_results) < 4:
            for c in candidates:
                if c.id in db_paints and c.id not in seen_ids:
                    seen_ids.add(c.id)
                    paint = db_paints[c.id]
                    fallback_reason = f"Harmonizes naturally with {room.lighting} lighting and existing {paint.color_family.lower()} accents."
                    verified_results.append((paint, fallback_reason))
                    if len(verified_results) == 4:
                        break

        return verified_results[:4]

    def magic_wand(
        self,
        image_bgr: np.ndarray,
        seed_point: Dict[str, float],  # {"x": 0.5, "y": 0.3} in normalized [0, 1]
        tolerance: float = 12.0,       # Slider range ~2 to 40
        protected_mask: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        operation: Literal["add", "subtract", "replace"] = "add",
        use_edge_barrier: bool = False
    ) -> np.ndarray:
        """
        Magic-wand flood fill on lightly smoothed image in LAB:
        - restricted to non-protected pixels
        - connected to tap
        - smooths result, fills holes, removes small blobs (< 0.1%)
        """
        h, w = image_bgr.shape[:2]
        seed_x = int(np.clip(round(seed_point["x"] * w), 0, w - 1))
        seed_y = int(np.clip(round(seed_point["y"] * h), 0, h - 1))

        # Check if seed falls on protected pixel; find closest non-protected pixel within 5px
        if protected_mask is not None and protected_mask[seed_y, seed_x] > 0:
            found = False
            for r in range(1, 8):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = seed_y + dy, seed_x + dx
                        if 0 <= ny < h and 0 <= nx < w and protected_mask[ny, nx] == 0:
                            seed_x, seed_y = nx, ny
                            found = True
                            break
                    if found:
                        break
            if not found:
                logger.warning("Magic wand tap is entirely within protected zone.")
                return active_mask if active_mask is not None else np.zeros((h, w), dtype=np.uint8)

        # 1. Lightly smooth image
        smoothed = cv2.GaussianBlur(image_bgr, (5, 5), 1.0)
        lab_8u = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

        # 2. Setup floodFill mask (size H+2, W+2)
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Non-zero pixels in ff_mask act as barriers
        if protected_mask is not None:
            ff_mask[1:-1, 1:-1] = (protected_mask > 0).astype(np.uint8) * 1

        if use_edge_barrier:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 100)
            ff_mask[1:-1, 1:-1] |= (edges > 0).astype(np.uint8) * 1

        # 3. OpenCV floodFill with gradient neighbor comparison
        tol_val = int(np.clip(tolerance, 1, 60))
        lo_diff = (tol_val, int(tol_val * 0.8), int(tol_val * 0.8))
        up_diff = (tol_val, int(tol_val * 0.8), int(tol_val * 0.8))

        # Flags: 4-connectivity, fill value 255 in mask only
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(lab_8u, ff_mask, (seed_x, seed_y), 255, lo_diff, up_diff, flags)

        flooded = (ff_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255

        # 4. Post-selection cleanup
        cleaned_selection = self.protection_service.clean_selection_mask(
            flooded, protected_mask=protected_mask
        )

        # 5. Apply operation onto active_mask
        return self._apply_operation(active_mask, cleaned_selection, operation, (h, w))

    def rectangle(
        self,
        image_shape: Tuple[int, int],
        box: Dict[str, float],  # {"x1": float, "y1": float, "x2": float, "y2": float}
        protected_mask: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        operation: Literal["add", "subtract", "replace"] = "add"
    ) -> np.ndarray:
        """
        Rectangle selection tool:
        Draws bounding box defined by corners, cleans, and applies operation.
        """
        h, w = image_shape[:2]
        x1 = int(np.clip(round(box["x1"] * w), 0, w - 1))
        y1 = int(np.clip(round(box["y1"] * h), 0, h - 1))
        x2 = int(np.clip(round(box["x2"] * w), 0, w - 1))
        y2 = int(np.clip(round(box["y2"] * h), 0, h - 1))

        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return active_mask if active_mask is not None else np.zeros((h, w), dtype=np.uint8)

        rect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(rect_mask, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), 255, -1)

        cleaned_selection = self.protection_service.clean_selection_mask(
            rect_mask, protected_mask=protected_mask
        )
        return self._apply_operation(active_mask, cleaned_selection, operation, (h, w))

    def polygon(
        self,
        image_shape: Tuple[int, int],
        points: List[Dict[str, float]],  # [{"x": 0.1, "y": 0.2}, ...]
        protected_mask: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        operation: Literal["add", "subtract", "replace"] = "add"
    ) -> np.ndarray:
        """
        Polygon selection tool:
        User clicks corners, closes shape. Rasterizes polygon, cleans, applies operation.
        """
        h, w = image_shape[:2]
        if len(points) < 3:
            return active_mask if active_mask is not None else np.zeros((h, w), dtype=np.uint8)

        pts = np.array(
            [[int(np.clip(round(p["x"] * w), 0, w - 1)), int(np.clip(round(p["y"] * h), 0, h - 1))] for p in points],
            dtype=np.int32
        )

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)

        cleaned_selection = self.protection_service.clean_selection_mask(
            poly_mask, protected_mask=protected_mask
        )
        return self._apply_operation(active_mask, cleaned_selection, operation, (h, w))

    def brush(
        self,
        image_shape: Tuple[int, int],
        points: List[Dict[str, float]],  # stroke path points
        radius: int = 25,
        protected_mask: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        operation: Literal["add", "subtract"] = "add"
    ) -> np.ndarray:
        """
        Brush add/subtract tool:
        Strokes with specified pixel radius, cleans, and unions or subtracts.
        """
        h, w = image_shape[:2]
        stroke_mask = np.zeros((h, w), dtype=np.uint8)

        for i, pt in enumerate(points):
            px = int(np.clip(round(pt["x"] * w), 0, w - 1))
            py = int(np.clip(round(pt["y"] * h), 0, h - 1))
            cv2.circle(stroke_mask, (px, py), radius, 255, -1)
            if i > 0:
                prev_x = int(np.clip(round(points[i - 1]["x"] * w), 0, w - 1))
                prev_y = int(np.clip(round(points[i - 1]["y"] * h), 0, h - 1))
                cv2.line(stroke_mask, (prev_x, prev_y), (px, py), 255, thickness=radius * 2)

        cleaned_selection = self.protection_service.clean_selection_mask(
            stroke_mask, protected_mask=protected_mask
        )
        return self._apply_operation(active_mask, cleaned_selection, operation, (h, w))

    def intelligent_scissors_snapped_contour(
        self,
        image_bgr: np.ndarray,
        point1: Dict[str, float],
        point2: Dict[str, float]
    ) -> List[Dict[str, float]]:
        """
        Computes edge-snapped shortest path contour between two points using OpenCV IntelligentScissorsMB.
        """
        h, w = image_bgr.shape[:2]
        p1_x = int(np.clip(round(point1["x"] * w), 0, w - 1))
        p1_y = int(np.clip(round(point1["y"] * h), 0, h - 1))
        p2_x = int(np.clip(round(point2["x"] * w), 0, w - 1))
        p2_y = int(np.clip(round(point2["y"] * h), 0, h - 1))

        if not hasattr(cv2, "segmentation") or not hasattr(cv2.segmentation, "IntelligentScissorsMB"):
            # Linear fallback
            return [point1, point2]

        tool = cv2.segmentation.IntelligentScissorsMB()
        tool.applyImage(image_bgr)
        tool.buildMap((p1_x, p1_y))
        contour = tool.getContour((p2_x, p2_y))

        if contour is None or len(contour) == 0:
            return [point1, point2]

        snapped_pts = []
        for pt in contour:
            px, py = pt[0]
            snapped_pts.append({"x": float(px / w), "y": float(py / h)})
        return snapped_pts

    def sam_selection(
        self,
        image_bgr: np.ndarray,
        cache_key: str,
        points: List[Dict[str, float]],
        protected_mask: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        operation: Literal["add", "subtract", "replace"] = "replace"
    ) -> np.ndarray:
        """
        Optional SAM tap-to-select tool (now optional, not default):
        Segments with MobileSAM ONNX, applies post-selection cleaning and auto-protection.
        """
        h, w = image_bgr.shape[:2]
        raw_mask = self.sam_service.segment_with_points(
            image_bgr=image_bgr,
            cache_key=cache_key,
            points=points,
            refine_edges=True
        )

        cleaned_selection = self.protection_service.clean_selection_mask(
            raw_mask, protected_mask=protected_mask
        )
        return self._apply_operation(active_mask, cleaned_selection, operation, (h, w))

    def _apply_operation(
        self,
        active_mask: Optional[np.ndarray],
        selection: np.ndarray,
        operation: Literal["add", "subtract", "replace"],
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Combines current active layer mask with new selection based on operation.
        """
        base = active_mask if active_mask is not None else np.zeros(shape, dtype=np.uint8)
        if base.shape[:2] != shape:
            base = cv2.resize(base, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        base_bin = (base > 127).astype(np.uint8) * 255
        sel_bin = (selection > 127).astype(np.uint8) * 255

        if operation == "add":
            result = cv2.bitwise_or(base_bin, sel_bin)
        elif operation == "subtract":
            result = cv2.bitwise_and(base_bin, cv2.bitwise_not(sel_bin))
        else:  # replace
            result = sel_bin

        return result
