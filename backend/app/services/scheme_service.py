import math
import logging
from typing import List, Dict, Tuple, Optional, Any
from sqlalchemy.orm import Session
import json

from app.models.paint import Paint
from app.schemas.paint import PaintCandidate
from app.schemas.vision import RoomAnalysis
from app.schemas.selection import SchemeSelectionItem, SchemeSelectionResponse
from app.services.shortlist_service import hex_to_rgb, rgb_to_hsl, rgb_to_lab, circular_hue_diff
from app.services.llm.factory import get_llm_client

logger = logging.getLogger("paintmywall.scheme_service")

def delta_e_lab(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """CIE76 Delta E color difference between two LAB colors."""
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )

class SchemeService:
    """
    Selects 4 diverse, architecturally sound 3-colour exterior schemes:
    - main: large facade body
    - accent: borders, slabs, architectural trims
    - trim_gate: exterior gates, metal grills, doors
    """

    @staticmethod
    def score_roles(paint: Paint) -> Dict[str, float]:
        """
        User Change 5:
        Soft scoring for roles instead of hard filters:
        - trim_gate: must be dark (L* <= ~50), rich tones welcome (not low-chroma).
        - main: light, neutral facade base (L* >= 60, moderate chroma).
        - accent: vibrant or distinct architectural elements.
        """
        rgb = hex_to_rgb(paint.hex)
        l_star, a_star, b_star = rgb_to_lab(*rgb)
        chroma = math.sqrt(a_star**2 + b_star**2)

        # 1. Main wall score
        # Ideal: L* in [65, 92], chroma < 25
        main_score = 60.0
        if l_star >= 60.0:
            main_score += (l_star - 60.0) * 1.2
        else:
            main_score -= (60.0 - l_star) * 3.5  # Soft penalty for dark walls

        if chroma < 22.0:
            main_score += (22.0 - chroma) * 1.0  # Bonus for elegant neutrals
        elif chroma > 30.0:
            main_score -= (chroma - 30.0) * 1.8  # Penalty for loud saturated main walls

        # 2. Trim & Gate score
        # User Change 5: trim_gate must be dark (L* <= ~50), not low-chroma.
        trim_score = 60.0
        if l_star <= 50.0:
            trim_score += (50.0 - l_star) * 2.0  # Bonus for deeper darker tones
            # Rich chroma bonus (navy, dark forest, charcoal bronze are great for gates)
            if chroma > 10.0:
                trim_score += min(chroma, 35.0) * 0.5
        else:
            trim_score -= (l_star - 50.0) * 3.5  # Strong soft penalty for light gates

        # 3. Accent score
        # Distinct presence, good chroma or contrasting tone
        accent_score = 55.0
        if chroma >= 15.0:
            accent_score += min(chroma, 40.0) * 1.0
        if 40.0 <= l_star <= 80.0:
            accent_score += 10.0

        return {
            "main": max(0.0, main_score),
            "accent": max(0.0, accent_score),
            "trim_gate": max(0.0, trim_score),
            "lab": (l_star, a_star, b_star),
            "chroma": chroma
        }

    def generate_scored_combos(
        self,
        candidates: List[PaintCandidate],
        paint_map: Dict[int, Paint],
        min_delta_e_main_accent: float = 15.0,  # User Change 5: >= 15
        min_delta_e_main_trim: float = 20.0,
        min_delta_e_accent_trim: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Builds and scores 3-colour combos in code:
        - Filters candidates into candidate pools by role scores
        - Enforces minimum Delta E requirements (User Change 5)
        - Computes composite harmony score
        """
        scored_paints = {}
        for c in candidates:
            p = paint_map.get(c.id)
            if p:
                scored_paints[c.id] = (p, self.score_roles(p))

        # Top candidates per role
        main_pool = sorted(
            [cid for cid, (_, sc) in scored_paints.items() if sc["main"] >= 50.0],
            key=lambda cid: scored_paints[cid][1]["main"],
            reverse=True
        )[:15]

        accent_pool = sorted(
            [cid for cid, (_, sc) in scored_paints.items() if sc["accent"] >= 50.0],
            key=lambda cid: scored_paints[cid][1]["accent"],
            reverse=True
        )[:15]

        trim_pool = sorted(
            [cid for cid, (_, sc) in scored_paints.items() if sc["trim_gate"] >= 45.0],
            key=lambda cid: scored_paints[cid][1]["trim_gate"],
            reverse=True
        )[:15]

        # Fallback if pools are small
        if not main_pool:
            main_pool = sorted(scored_paints.keys(), key=lambda cid: scored_paints[cid][1]["main"], reverse=True)[:10]
        if not accent_pool:
            accent_pool = sorted(scored_paints.keys(), key=lambda cid: scored_paints[cid][1]["accent"], reverse=True)[:10]
        if not trim_pool:
            trim_pool = sorted(scored_paints.keys(), key=lambda cid: scored_paints[cid][1]["trim_gate"], reverse=True)[:10]

        combos = []
        for m_id in main_pool:
            m_paint, m_sc = scored_paints[m_id]
            m_lab = m_sc["lab"]

            for a_id in accent_pool:
                if a_id == m_id:
                    continue
                a_paint, a_sc = scored_paints[a_id]
                a_lab = a_sc["lab"]

                de_ma = delta_e_lab(m_lab, a_lab)
                # User Change 5: Require deltaE >= 15 between main and accent
                if de_ma < min_delta_e_main_accent:
                    continue

                for t_id in trim_pool:
                    if t_id == m_id or t_id == a_id:
                        continue
                    t_paint, t_sc = scored_paints[t_id]
                    t_lab = t_sc["lab"]

                    de_mt = delta_e_lab(m_lab, t_lab)
                    if de_mt < min_delta_e_main_trim:
                        continue

                    de_at = delta_e_lab(a_lab, t_lab)
                    if de_at < min_delta_e_accent_trim:
                        continue

                    # Composite score
                    total_score = (
                        m_sc["main"] * 0.35 +
                        a_sc["accent"] * 0.30 +
                        t_sc["trim_gate"] * 0.35 +
                        min(de_ma, 40.0) * 0.4 +
                        min(de_mt, 50.0) * 0.3
                    )

                    combos.append({
                        "main_id": m_id,
                        "accent_id": a_id,
                        "trim_gate_id": t_id,
                        "main_paint": m_paint,
                        "accent_paint": a_paint,
                        "trim_paint": t_paint,
                        "score": round(total_score, 2),
                        "de_ma": round(de_ma, 1),
                        "de_mt": round(de_mt, 1),
                        "main_family": m_paint.color_family
                    })

        return combos

    def select_top_diverse_combos(
        self,
        combos: List[Dict[str, Any]],
        target_count: int = 24
    ) -> List[Dict[str, Any]]:
        """
        User Change 6:
        Score combos in code, keep the top 20-30 with diversity across main colour families
        and varied accent and trim combinations.
        """
        if not combos:
            return []

        # Sort all by score descending
        combos.sort(key=lambda c: c["score"], reverse=True)

        # Group by main paint color family
        family_groups: Dict[str, List[Dict[str, Any]]] = {}
        for c in combos:
            fam = c["main_family"]
            if fam not in family_groups:
                family_groups[fam] = []
            family_groups[fam].append(c)

        diverse_combos: List[Dict[str, Any]] = []
        used_accent_ids = set()
        used_trim_ids = set()

        # Pass 1: Select best combo per family trying to introduce fresh accent/trim
        for fam, group in family_groups.items():
            if len(diverse_combos) >= target_count:
                break
            # Find candidate in this family with fresh accent/trim if possible
            best_fresh = None
            for cand in group:
                if cand["accent_id"] not in used_accent_ids and cand["trim_gate_id"] not in used_trim_ids:
                    best_fresh = cand
                    break
            chosen = best_fresh if best_fresh is not None else group[0]
            diverse_combos.append(chosen)
            used_accent_ids.add(chosen["accent_id"])
            used_trim_ids.add(chosen["trim_gate_id"])
            group.remove(chosen)

        # Pass 2: Round-robin fill remaining slots
        while len(diverse_combos) < min(target_count, len(combos)):
            added = False
            for fam, group in list(family_groups.items()):
                if group and len(diverse_combos) < target_count:
                    # Pick highest scoring remaining
                    chosen = group.pop(0)
                    diverse_combos.append(chosen)
                    added = True
            if not added:
                break

        return diverse_combos

    def build_llm_prompt(
        self,
        room: RoomAnalysis,
        diverse_combos: List[Dict[str, Any]]
    ) -> str:
        """
        Creates prompt sending ONLY the top 20-30 diverse combos to the LLM.
        """
        combo_lines = []
        for idx, c in enumerate(diverse_combos, 1):
            mp = c["main_paint"]
            ap = c["accent_paint"]
            tp = c["trim_paint"]
            combo_lines.append(
                f"Combo {idx}: "
                f"main_id={mp.id} (\"{mp.name}\", {mp.shade_code}, {mp.hex}, {mp.color_family}) | "
                f"accent_id={ap.id} (\"{ap.name}\", {ap.shade_code}, {ap.hex}, {ap.color_family}) | "
                f"trim_gate_id={tp.id} (\"{tp.name}\", {tp.shade_code}, {tp.hex}, {tp.color_family})"
            )

        prompt = f"""You are a senior architectural colour consultant specializing in exterior residential facades.
Building Analysis:
- Scene: {room.scene_type}
- Current Exterior Wall: {room.current_wall_color_hex}
- Architectural Style: {room.building_type or room.style}
- Surroundings: {room.surroundings or "Residential exterior"}
- Design Mood: {room.mood}
- Exterior Lighting: {room.lighting}

Select the 4 BEST and most aesthetically diverse colour SCHEMES from ONLY the shortlisted candidate combos below.
Rules:
1. You MUST choose ONLY from the listed combo options. Do not make up any shade IDs.
2. Select 4 schemes with distinct design moods (e.g. Classic Elegance, Contemporary Warmth, Earthy Natural, Bold Modern Contrast).
3. For each scheme, provide a 1-line client-facing professional reason explaining how the main wall, accent border, and trim/gate harmonize.

Shortlisted Candidate Combos:
{chr(10).join(combo_lines)}

Output STRICT JSON matching this schema:
{{
  "schemes": [
    {{
      "main_id": <int>,
      "accent_id": <int>,
      "trim_gate_id": <int>,
      "reason": "<1-line reason>"
    }}
  ]
}}
"""
        return prompt

    async def select_schemes(
        self,
        db: Session,
        room: RoomAnalysis,
        candidates: List[PaintCandidate]
    ) -> List[SchemeSelectionItem]:
        """
        End-to-end scheme selection pipeline:
        1. Code evaluates and scores combos, enforcing deltaE >= 15 between main & accent.
        2. Code keeps top 20-30 with color family diversity.
        3. LLM picks 4 schemes with rationale.
        4. Code verifies validity of all IDs and applies fallbacks if necessary.
        """
        all_paints = {p.id: p for p in db.query(Paint).all()}

        # Step 1: Score combos in code
        combos = self.generate_scored_combos(candidates, all_paints)
        if not combos:
            # Fallback with relaxed deltaE if needed
            combos = self.generate_scored_combos(candidates, all_paints, min_delta_e_main_accent=10.0, min_delta_e_main_trim=15.0)

        # Step 2: Keep top 20-30 diverse combos
        top_combos = self.select_top_diverse_combos(combos, target_count=24)
        logger.info(f"Generated {len(combos)} total exterior combos; filtered to {len(top_combos)} diverse candidates for LLM")

        # Step 3: Call LLM
        prompt = self.build_llm_prompt(room, top_combos)
        client = get_llm_client()

        llm_schemes: List[SchemeSelectionItem] = []
        try:
            raw_response = await client.generate_text(
                prompt=prompt,
                system_prompt="You are an expert exterior colour consultant. Output strictly valid JSON with no markdown wrapping."
            )
            # Clean JSON
            clean_text = raw_response.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed = json.loads(clean_text)
            for item in parsed.get("schemes", []):
                m_id = int(item.get("main_id", 0))
                a_id = int(item.get("accent_id", 0))
                t_id = int(item.get("trim_gate_id", 0))
                reason = str(item.get("reason", "Harmonious multi-colour exterior scheme")).strip()
                if m_id in all_paints and a_id in all_paints and t_id in all_paints:
                    llm_schemes.append(SchemeSelectionItem(
                        main_id=m_id,
                        accent_id=a_id,
                        trim_gate_id=t_id,
                        reason=reason
                    ))
        except Exception as e:
            logger.warning(f"LLM scheme selection failed or invalid format: {e}; falling back to code-scored combos")

        # Step 4: Verification & Fallbacks
        final_schemes: List[SchemeSelectionItem] = []
        seen_combos = set()

        for s in llm_schemes:
            combo_key = (s.main_id, s.accent_id, s.trim_gate_id)
            if combo_key not in seen_combos and s.main_id != s.accent_id and s.main_id != s.trim_gate_id:
                seen_combos.add(combo_key)
                final_schemes.append(s)
            if len(final_schemes) >= 4:
                break

        # Fill remaining slots with top-scored diverse combos
        for c in top_combos:
            if len(final_schemes) >= 4:
                break
            combo_key = (c["main_id"], c["accent_id"], c["trim_gate_id"])
            if combo_key not in seen_combos:
                seen_combos.add(combo_key)
                m = c["main_paint"]
                a = c["accent_paint"]
                t = c["trim_paint"]
                reason = f"Sophisticated palette pairing {m.name} facade with {a.name} architectural borders and {t.name} trims."
                final_schemes.append(SchemeSelectionItem(
                    main_id=c["main_id"],
                    accent_id=c["accent_id"],
                    trim_gate_id=c["trim_gate_id"],
                    reason=reason
                ))

        return final_schemes[:4]
