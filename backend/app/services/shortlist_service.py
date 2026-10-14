import math
from typing import List, Tuple, Dict, Any
from sqlalchemy.orm import Session
from app.models.paint import Paint
from app.schemas.vision import RoomAnalysis
from app.schemas.paint import PaintCandidate

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (200, 200, 200)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    c_max = max(r_norm, g_norm, b_norm)
    c_min = min(r_norm, g_norm, b_norm)
    delta = c_max - c_min

    # Lightness
    l = (c_max + c_min) / 2.0

    # Saturation
    if delta == 0:
        s = 0.0
        h = 0.0
    else:
        s = delta / (1.0 - abs(2.0 * l - 1.0))
        # Hue
        if c_max == r_norm:
            h = ((g_norm - b_norm) / delta) % 6.0
        elif c_max == g_norm:
            h = ((b_norm - r_norm) / delta) + 2.0
        else:
            h = ((r_norm - g_norm) / delta) + 4.0
        h = h * 60.0
        if h < 0:
            h += 360.0

    return (h, s, l)

def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to CIE-L*a*b* using standard D65 illuminant."""
    # Linearize sRGB
    def pivot_rgb(c: float) -> float:
        c = c / 255.0
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r_lin = pivot_rgb(r)
    g_lin = pivot_rgb(g)
    b_lin = pivot_rgb(b)

    # Convert to XYZ
    x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) * 100
    y = (r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750) * 100
    z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) * 100

    # D65 standard reference white
    xn, yn, zn = 95.047, 100.000, 108.883

    def pivot_xyz(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx = pivot_xyz(x / xn)
    fy = pivot_xyz(y / yn)
    fz = pivot_xyz(z / zn)

    l_star = (116.0 * fy) - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return (l_star, a_star, b_star)

def circular_hue_diff(h1: float, h2: float) -> float:
    diff = abs(h1 - h2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff

class ShortlistService:
    def shortlist_shades(
        self,
        db: Session,
        room: RoomAnalysis,
        min_count: int = 30,
        max_count: int = 40
    ) -> List[PaintCandidate]:
        """
        Filter paints to 30-40 candidates using color harmony (neutral, analogous, complementary)
        in LAB/HSL space, with lighting-aware filtering.
        """
        all_paints: List[Paint] = db.query(Paint).all()
        if not all_paints:
            return []

        # Analyze reference colors (safely handle optional floor_color_hex for exteriors)
        wall_rgb = hex_to_rgb(room.current_wall_color_hex)
        wall_hsl = rgb_to_hsl(*wall_rgb)

        reference_hues = [wall_hsl[0]]
        if room.floor_color_hex:
            floor_rgb = hex_to_rgb(room.floor_color_hex)
            floor_hsl = rgb_to_hsl(*floor_rgb)
            reference_hues.append(floor_hsl[0])

        accents = (room.accent_colors if room.scene_type == "exterior" and room.accent_colors else room.dominant_furniture_colors) or []
        furniture_hsls = [rgb_to_hsl(*hex_to_rgb(c)) for c in accents]
        reference_hues += [f[0] for f in furniture_hsls]

        candidates: List[PaintCandidate] = []

        for p in all_paints:
            rgb = hex_to_rgb(p.hex)
            h, s, l = rgb_to_hsl(*rgb)
            l_star, a_star, b_star = rgb_to_lab(*rgb)
            chroma = math.sqrt(a_star**2 + b_star**2)

            # 1. Lighting factor:
            # Dim rooms benefit strongly from lighter, light-reflecting shades
            if room.lighting == "dim":
                if l_star < 68.0:
                    continue  # Filter out dark/heavy shades in dim rooms
                lighting_bonus = (l_star - 68.0) * 0.4
            elif room.lighting == "bright":
                # Bright rooms can accommodate both airy and rich, moody tones
                lighting_bonus = 5.0
            else:  # mixed
                lighting_bonus = 3.0

            # 2. Categorize Harmony
            category = "General"
            harmony_score = 0.0

            # A. Neutral / Off-white (Chroma low or saturation low, high lightness)
            if chroma < 16.0 or s < 0.25:
                category = "Neutral"
                harmony_score = 80.0 + (l_star * 0.2) + lighting_bonus

            # B. Analogous (Close in hue to room reference colors: 10° to 35°)
            else:
                min_diff = min(circular_hue_diff(h, ref_h) for ref_h in reference_hues)
                if min_diff <= 35.0:
                    category = "Analogous"
                    harmony_score = 75.0 + (35.0 - min_diff) + lighting_bonus

                # C. Complementary (Opposite in hue to furniture or floor: ~180° ± 35°)
                elif any(abs(circular_hue_diff(h, ref_h) - 180.0) <= 35.0 for ref_h in reference_hues):
                    category = "Complementary"
                    harmony_score = 70.0 + (35.0 - min(abs(circular_hue_diff(h, ref_h) - 180.0) for ref_h in reference_hues)) + lighting_bonus
                else:
                    category = "Accent"
                    harmony_score = 50.0 + lighting_bonus

            candidates.append(PaintCandidate(
                id=p.id,
                name=p.name,
                shade_code=p.shade_code,
                hex=p.hex,
                finish=p.finish,
                color_family=p.color_family,
                harmony_category=category,
                score=round(harmony_score, 2)
            ))

        # If lighting filter was too strict (e.g. fewer than min_count found), relax filter
        if len(candidates) < min_count and len(all_paints) >= min_count:
            # Fallback to scoring all paints without hard filter
            candidates = []
            for p in all_paints:
                rgb = hex_to_rgb(p.hex)
                h, s, l = rgb_to_hsl(*rgb)
                l_star, a_star, b_star = rgb_to_lab(*rgb)
                chroma = math.sqrt(a_star**2 + b_star**2)
                score = (l_star * 0.5) + (30.0 if (chroma < 18 or s < 0.25) else 20.0)
                category = "Neutral" if (chroma < 18 or s < 0.25) else "Accent"
                candidates.append(PaintCandidate(
                    id=p.id,
                    name=p.name,
                    shade_code=p.shade_code,
                    hex=p.hex,
                    finish=p.finish,
                    color_family=p.color_family,
                    harmony_category=category,
                    score=round(score, 2)
                ))

        # Sort by score descending and ensure category diversity
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Diverse balancing: take best neutrals, analogous, and complementary
        neutrals = [c for c in candidates if c.harmony_category == "Neutral"]
        analogous = [c for c in candidates if c.harmony_category == "Analogous"]
        complementary = [c for c in candidates if c.harmony_category == "Complementary"]
        others = [c for c in candidates if c.harmony_category not in ("Neutral", "Analogous", "Complementary")]

        selected: List[PaintCandidate] = []
        # Target balanced distribution
        while len(selected) < min(max_count, len(candidates)):
            added = False
            for group in [neutrals, analogous, complementary, others]:
                if group and len(selected) < min(max_count, len(candidates)):
                    selected.append(group.pop(0))
                    added = True
            if not added:
                break

        return selected[:max_count]
