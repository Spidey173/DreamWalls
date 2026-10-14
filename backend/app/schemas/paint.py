from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class PaintBase(BaseModel):
    brand: str = "Asian Paints"
    name: str
    shade_code: str
    hex: str
    finish: str = "Matte"
    color_family: str
    price: Optional[float] = None

class PaintCreate(PaintBase):
    pass

class PaintUpdate(BaseModel):
    brand: Optional[str] = None
    name: Optional[str] = None
    shade_code: Optional[str] = None
    hex: Optional[str] = None
    finish: Optional[str] = None
    color_family: Optional[str] = None
    price: Optional[float] = None

class PaintRead(PaintBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class PaintCandidate(BaseModel):
    id: int
    name: str
    shade_code: str
    hex: str
    finish: str
    color_family: str
    harmony_category: str
    score: float
