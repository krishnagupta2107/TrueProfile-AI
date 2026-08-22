import os
import uuid
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Form, File, UploadFile
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.database import get_db
from backend.models.profile import Profile
from backend.schemas.profile import ProfileCreate, ProfileResponse
from backend.services.profile_analyzer import ProfileAnalyzerService
from backend.auth import require_api_key

router = APIRouter(prefix="/profiles", tags=["profiles"])
analyzer_service = ProfileAnalyzerService()
limiter = Limiter(key_func=get_remote_address)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_profile_analysis(db: Session, profile_dict: dict, analysis_result: dict) -> Profile:
    """Helper to persist analyzed profile and its ML scores into the database."""
    db_profile = Profile(
        username=profile_dict.get("username"),
        account_age_days=profile_dict.get("account_age_days", 0),
        followers=profile_dict.get("followers", 0),
        following=profile_dict.get("following", 0),
        posts_per_day=profile_dict.get("posts_per_day", 0.0),
        profile_completeness=profile_dict.get("profile_completeness", 0.5),
        follow_burst_rate=profile_dict.get("follow_burst_rate", 0.0),
        posting_variance=profile_dict.get("posting_variance", 0.0),
        engagement_rate=profile_dict.get("engagement_rate", 0.0),
        profile_image_url=profile_dict.get("profile_image_url"),

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


@router.get("/", response_model=List[ProfileResponse])
def get_profiles(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """List all analyzed profiles stored in the database."""
    profiles = db.query(Profile).offset(skip).limit(limit).all()
    return profiles


@router.get("/{profile_id}", response_model=ProfileResponse)
def get_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """Fetch a single analyzed profile by its database ID."""
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/analyze/username")
def analyze_by_username():
    """
    Deprecated stub endpoint.
    Automated scraping/ingestion was removed to prevent Terms of Service violations
    and data fabrication.
    """
    raise HTTPException(
        status_code=501,
        detail="Automated ingestion not available — use /profiles/analyze/manual to submit verified account data."
    )


@router.post("/analyze/manual", response_model=ProfileResponse)
@limiter.limit("20/minute")
async def analyze_manual_profile(
    request: Request,
    username: Optional[str] = Form(None),
    account_age_days: int = Form(0),
    followers: int = Form(0),
    following: int = Form(0),
    posts_per_day: float = Form(0.0),
    profile_completeness: float = Form(0.5),
    follow_burst_rate: float = Form(0.0),
    posting_variance: float = Form(0.0),
    engagement_rate: float = Form(0.0),
    profile_image_url: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    Accepts real profile data via multipart form data alongside an optional image upload file.
    Runs the full 5-signal ML fusion pipeline without fabricating or pre-classifying any data.
    """
    final_image_url = profile_image_url

    # Handle image file upload if provided
    if image and image.filename:
        ext = os.path.splitext(image.filename)[1] or ".jpg"
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        saved_filepath = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(saved_filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # Serve via local static URL so image models can download/process the real image bytes
        final_image_url = f"http://127.0.0.1:8000/uploads/{unique_filename}"

    profile_dict = {
        "username": username,
        "account_age_days": account_age_days,
        "followers": followers,
        "following": following,
        "posts_per_day": posts_per_day,
        "profile_completeness": profile_completeness,
        "follow_burst_rate": follow_burst_rate,
        "posting_variance": posting_variance,
        "engagement_rate": engagement_rate,
        "profile_image_url": final_image_url,
    }

    # Execute ML pipeline on genuine input data
    analysis_result = analyzer_service.analyze_profile(profile_dict)

    # Persist and return
    return _save_profile_analysis(db, profile_dict, analysis_result)


@router.post("/analyze", response_model=ProfileResponse)
@limiter.limit("20/minute")
def analyze_full_profile(
    request: Request,
    profile_data: ProfileCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    Takes full profile feature data as JSON and runs through the ML pipeline.
    Useful for testing and direct API integration.
    """
    profile_dict = profile_data.model_dump()
    analysis_result = analyzer_service.analyze_profile(profile_dict)
    return _save_profile_analysis(db, profile_dict, analysis_result)
