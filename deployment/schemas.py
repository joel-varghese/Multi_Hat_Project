from pydantic import BaseModel, Field
from typing import Optional, List

class PredictRequest(BaseModel):
    title:       str
    description: Optional[str] = ""
    top_k:       Optional[int] = Field(10,  ge=1, le=50)
    threshold:   Optional[float] = Field(None, ge=0.0, le=1.0)

class SkillOut(BaseModel):
    skill_id:   str
    skill_name: str
    confidence: float

class PredictResponse(BaseModel):
    title:   str
    skills:  List[SkillOut]
    total:   int