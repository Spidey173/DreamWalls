from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re

class RoomAnalysis(BaseModel):
    scene_type: Literal["interior", "exterior"] = Field(
        default="interior",
        description="Type of scene: 'interior' or 'exterior'"
    )
    lighting: Literal["bright", "dim", "mixed"] = Field(
        default="bright",
        description="Lighting condition: bright, dim, or mixed"
    )
    current_wall_color_hex: str = Field(
        default="#EFE8DA",
        description="Estimated current wall hex color code, e.g. #EFE8DA"
    )
    style: str = Field(
        default="Modern",
        description="e.g. Modern, Traditional, Scandinavian, Minimalist, Industrial, Colonial, Contemporary"
    )
    mood: str = Field(
        default="Warm and welcoming",
        description="e.g. Warm and cozy, Airy and energetic, Serene and calm, Bold and elegant"
    )

    # Interior-only fields
    room_type: Optional[str] = Field(
        default="living room",
        description="e.g. living room, bedroom, dining room, kitchen, home office"
    )
    floor_color_hex: Optional[str] = Field(
        default="#7E5233",
        description="Estimated flooring hex color code, e.g. #7E5233"
    )
    dominant_furniture_colors: List[str] = Field(
        default_factory=list,
        description="List of 2-4 dominant furniture hex color codes for interior, e.g. ['#334D5C', '#E27D60']"
    )

    # Exterior-only fields
    building_type: Optional[str] = Field(
        default=None,
        description="e.g. Single-family house, Villa, Bungalow, Apartment complex, Commercial facade"
    )
    light_conditions: Optional[Literal["sunny", "overcast", "golden_hour", "mixed"]] = Field(
        default=None,
        description="Natural lighting conditions for exterior: sunny, overcast, golden_hour, mixed"
    )
    surroundings: Optional[str] = Field(
        default=None,
        description="e.g. Suburban street, Lush green garden, Urban concrete with road, Mountain backdrop"
    )
    accent_colors: Optional[List[str]] = Field(
        default=None,
        description="List of 2-4 accent/trim/gate/roof hex color codes for exterior, e.g. ['#334D5C', '#E27D60']"
    )

    @field_validator("current_wall_color_hex", mode="before")
    @classmethod
    def normalize_wall_hex(cls, v: str) -> str:
        if not isinstance(v, str):
            return "#EFE8DA"
        v = v.strip()
        if not v.startswith("#"):
            v = f"#{v}"
        if re.match(r"^#[0-9a-fA-F]{6}$", v):
            return v.upper()
        if re.match(r"^#[0-9a-fA-F]{3}$", v):
            return f"#{v[1]*2}{v[2]*2}{v[3]*2}".upper()
        return "#EFE8DA"

    @field_validator("floor_color_hex", mode="before")
    @classmethod
    def normalize_floor_hex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            return None
        v = v.strip()
        if not v.startswith("#"):
            v = f"#{v}"
        if re.match(r"^#[0-9a-fA-F]{6}$", v):
            return v.upper()
        if re.match(r"^#[0-9a-fA-F]{3}$", v):
            return f"#{v[1]*2}{v[2]*2}{v[3]*2}".upper()
        return "#CCCCCC"

    @field_validator("dominant_furniture_colors", mode="before")
    @classmethod
    def normalize_furniture_hexes(cls, v: list) -> list[str]:
        if not isinstance(v, list):
            return ["#4A3728"]
        cleaned = []
        for item in v:
            if isinstance(item, str):
                item = item.strip()
                if not item.startswith("#"):
                    item = f"#{item}"
                if re.match(r"^#[0-9a-fA-F]{6}$", item):
                    cleaned.append(item.upper())
                elif re.match(r"^#[0-9a-fA-F]{3}$", item):
                    cleaned.append(f"#{item[1]*2}{item[2]*2}{item[3]*2}".upper())
        return cleaned or ["#4A3728"]

    @field_validator("accent_colors", mode="before")
    @classmethod
    def normalize_accent_hexes(cls, v: Optional[list]) -> Optional[list[str]]:
        if v is None:
            return None
        if not isinstance(v, list):
            return ["#4A3728"]
        cleaned = []
        for item in v:
            if isinstance(item, str):
                item = item.strip()
                if not item.startswith("#"):
                    item = f"#{item}"
                if re.match(r"^#[0-9a-fA-F]{6}$", item):
                    cleaned.append(item.upper())
                elif re.match(r"^#[0-9a-fA-F]{3}$", item):
                    cleaned.append(f"#{item[1]*2}{item[2]*2}{item[3]*2}".upper())
        return cleaned or ["#4A3728"]

    @model_validator(mode="after")
    def sync_accent_and_furniture_colors(self):
        if self.scene_type == "exterior":
            if self.accent_colors and not self.dominant_furniture_colors:
                self.dominant_furniture_colors = list(self.accent_colors)
            elif self.dominant_furniture_colors and not self.accent_colors:
                self.accent_colors = list(self.dominant_furniture_colors)
        return self

