from typing import List, Optional, Dict
from pydantic import BaseModel
from app.schemas.vision import RoomAnalysis

class RepaintedShadeResult(BaseModel):
    shade_id: int
    shade_name: str
    shade_code: str
    hex: str
    finish: str
    color_family: str
    reason: str
    image_url: str
    price: Optional[float] = None

class SchemeShadeInfo(BaseModel):
    role: str  # "main" | "accent" | "trim_gate"
    shade_id: int
    shade_name: str
    shade_code: str
    hex: str
    finish: str
    color_family: str

class SchemeResult(BaseModel):
    scheme_index: int  # 1-4
    reason: str
    shades: List[SchemeShadeInfo]  # exactly 3 (main, accent, trim_gate)
    image_url: str  # composite recolored image
    status: Optional[str] = "success"  # "success" | "failed" | "fallback"
    mode: Optional[str] = "ai"  # "ai" | "precise"
    mock: Optional[bool] = False  # True if generated via MockImageClient
    delta_e: Optional[float] = None
    drift_outside_pct: Optional[float] = None
    error_message: Optional[str] = None

class AnalysisAndRecolorResponse(BaseModel):
    session_id: str
    original_image_url: str
    mask_url: str
    masks: Optional[Dict[str, str]] = None  # {"main": url, "accent": url, "trim_gate": url}
    coverage_pct: Optional[Dict[str, float]] = None
    coverage_by_layer: Optional[Dict[str, float]] = None
    sky_fraction_by_layer: Optional[Dict[str, float]] = None  # {"main": 0.0, "accent": 0.0, "trim_gate": 0.0}
    protected_mask_url: Optional[str] = None
    protected_hatched_url: Optional[str] = None
    room_analysis: RoomAnalysis
    results: List[RepaintedShadeResult]
    schemes: Optional[List[SchemeResult]] = None
    mode: Optional[str] = "ai"  # "ai" | "precise"
    mock: Optional[bool] = False  # True if using mock client
    fallback_reason: Optional[str] = None

class RegenerateSchemePayload(BaseModel):
    session_id: str
    scheme_index: int  # 1-4

class RecolorAdjustmentRequest(BaseModel):
    session_id: str
    custom_mask_base64: Optional[str] = None
