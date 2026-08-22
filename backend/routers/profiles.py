from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models.profile import Profile
from backend.schemas.profile import ProfileCreate, ProfileResponse
from backend.services.profile_analyzer import ProfileAnalyzerService

router = APIRouter(prefix="/profiles", tags=["profiles"])
analyzer_service = ProfileAnalyzerService()

@router.get("/", response_model=List[ProfileResponse])
def get_profiles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    profiles = db.query(Profile).offset(skip).limit(limit).all()
    return profiles

@router.get("/{profile_id}", response_model=ProfileResponse)
def get_profile(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@router.post("/analyze", response_model=ProfileResponse)
def analyze_and_store_profile(profile_data: ProfileCreate, db: Session = Depends(get_db)):
    # 1. Analyze profile
    analysis_result = analyzer_service.analyze_profile(profile_data.model_dump())
    
    # 2. Create DB record
    db_profile = Profile(
        username=profile_data.username,
        account_age_days=profile_data.account_age_days,
        followers=profile_data.followers,
        following=profile_data.following,
        posts_per_day=profile_data.posts_per_day,
        profile_image_url=profile_data.profile_image_url,
        
        face_score=analysis_result["scores"]["face"],
        deepfake_score=analysis_result["scores"]["deepfake"],
        behavior_score=analysis_result["scores"]["behavior"],
        metadata_score=analysis_result["scores"]["metadata"],
        network_score=analysis_result["scores"]["network"],
        
        risk_score=analysis_result["risk_score"],
        risk_level=analysis_result["risk_level"],
        evidence=analysis_result["evidence"],
        
        model_version=analysis_result["model_version"],
        analyzed_at=analysis_result["analyzed_at"]
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    return db_profile
