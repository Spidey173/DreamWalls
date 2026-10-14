from typing import List
from pydantic import BaseModel, Field

class ShadeSelectionItem(BaseModel):
    id: int = Field(description="Must match an exact id from the provided candidate list")
    reason: str = Field(description="One concise line explaining why this shade suits the room, lighting, and mood")

class ShadeSelectionResponse(BaseModel):
    selections: List[ShadeSelectionItem] = Field(
        min_length=1,
        max_length=6,
        description="List of selected shade items with valid catalog IDs and reasons"
    )

class SchemeSelectionItem(BaseModel):
    main_id: int = Field(description="Must match an exact candidate id for the main wall")
    accent_id: int = Field(description="Must match an exact candidate id for borders/accents")
    trim_gate_id: int = Field(description="Must match an exact candidate id for gate/trim")
    reason: str = Field(description="One concise line explaining why this 3-colour scheme works harmoniously together")

class SchemeSelectionResponse(BaseModel):
    schemes: List[SchemeSelectionItem] = Field(
        min_length=1,
        max_length=6,
        description="List of selected 3-colour schemes"
    )
