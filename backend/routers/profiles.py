from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models.profile import Profile
from backend.schemas.profile import ProfileCreate, ProfileResponse, ProfileByUsername
from backend.services.profile_analyzer import ProfileAnalyzerService
from backend.services.ingestion import ingest_profile_by_username

router = APIRouter(prefix="/profiles", tags=["profiles"])
analyzer_service = ProfileAnalyzerService()


@router.get("/", response_model=List[ProfileResponse])
def get_profiles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all analyzed profiles stored in the database."""
    profiles = db.query(Profile).offset(skip).limit(limit).all()
    return profiles


@router.get("/{profile_id}", response_model=ProfileResponse)
def get_profile(profile_id: int, db: Session = Depends(get_db)):
    """Fetch a single analyzed profile by its database ID."""
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/analyze/username", response_model=ProfileResponse)
def analyze_by_username(payload: ProfileByUsername, db: Session = Depends(get_db)):
    """
    Takes ONLY a username. Automatically ingests profile data (mock scraper)
    then runs it through the full ML pipeline.
    """
    # 1. Ingest profile features from username
    profile_data = ingest_profile_by_username(payload.username)

    # 2. Run ML pipeline
    analysis_result = analyzer_service.analyze_profile(profile_data)

    # 3. Store result in DB
    db_profile = Profile(
        username=profile_data["username"],
        account_age_days=profile_data["account_age_days"],
        followers=profile_data["followers"],
        following=profile_data["following"],
        posts_per_day=profile_data["posts_per_day"],
        profile_completeness=profile_data.get("profile_completeness", 0.5),
        follow_burst_rate=profile_data.get("follow_burst_rate", 0.0),
        posting_variance=profile_data.get("posting_variance", 0.0),
        engagement_rate=profile_data.get("engagement_rate", 0.0),
        profile_image_url=profile_data.get("profile_image_url"),

        face_score=analysis_result["scores"]["face"],
        deepfake_score=analysis_result["scores"]["deepfake"],
        behavior_score=analysis_result["scores"]["behavior"],
        metadata_score=analysis_result["scores"]["metadata"],
        network_score=analysis_result["scores"]["network"],

        risk_score=analysis_result["risk_score"],
        risk_level=analysis_result["risk_level"],
        recommended_action=analysis_result["recommended_action"],
        evidence=analysis_result["evidence"],

        model_version=analysis_result["model_version"],
        analyzed_at=analysis_result["analyzed_at"],
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile


@router.post("/analyze", response_model=ProfileResponse)
def analyze_full_profile(profile_data: ProfileCreate, db: Session = Depends(get_db)):
    """
    Takes full profile feature data and runs through the ML pipeline.
    Useful for testing and the mock data generator.
    """
    analysis_result = analyzer_service.analyze_profile(profile_data.model_dump())

    db_profile = Profile(
        username=profile_data.username,
        account_age_days=profile_data.account_age_days,
        followers=profile_data.followers,
        following=profile_data.following,
        posts_per_day=profile_data.posts_per_day,
        profile_completeness=profile_data.profile_completeness,
        follow_burst_rate=profile_data.follow_burst_rate,
        posting_variance=profile_data.posting_variance,
        engagement_rate=profile_data.engagement_rate,
        profile_image_url=profile_data.profile_image_url,

        face_score=analysis_result["scores"]["face"],
        deepfake_score=analysis_result["scores"]["deepfake"],
        behavior_score=analysis_result["scores"]["behavior"],
        metadata_score=analysis_result["scores"]["metadata"],
        network_score=analysis_result["scores"]["network"],

        risk_score=analysis_result["risk_score"],
        risk_level=analysis_result["risk_level"],
        recommended_action=analysis_result["recommended_action"],
        evidence=analysis_result["evidence"],

        model_version=analysis_result["model_version"],
        analyzed_at=analysis_result["analyzed_at"],
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile
