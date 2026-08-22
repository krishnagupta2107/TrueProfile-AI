from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ProfileCreate(BaseModel):
    username: Optional[str] = None
    account_age_days: int = 0
    followers: int = 0
    following: int = 0
    posts_per_day: float = 0.0
    profile_image_url: Optional[str] = None

class ProfileResponse(BaseModel):
    id: int
    username: Optional[str]
    
    # Raw features
    account_age_days: int
    followers: int
    following: int
    posts_per_day: float
    profile_image_url: Optional[str]
    
    # ML Scores
    face_score: Optional[float]
    deepfake_score: Optional[float]
    behavior_score: Optional[float]
    metadata_score: Optional[float]
    network_score: Optional[float]
    
    # Final outcome
    risk_score: Optional[float]
    risk_level: Optional[str]
    evidence: List[str]
    
    # Audit
    model_version: str
    created_at: datetime
    analyzed_at: Optional[datetime]
    
    class Config:
        from_attributes = True
