"""
Human Review Queue router.

Provides endpoints for:
- Listing all profiles that need human review (BORDERLINE risk)
- Marking a profile as reviewed (approve / flag)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime, timezone

from backend.database import get_db
from backend.models.profile import Profile
from backend.schemas.profile import ProfileResponse
from backend.auth import require_api_key

router = APIRouter(prefix="/review", tags=["human-review"])


class ReviewDecision(BaseModel):
    decision: str   # "APPROVE" or "FLAG"
    reviewer_note: str = ""


@router.get("/queue", response_model=List[ProfileResponse])
def get_review_queue(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    Returns all profiles with recommended_action == 'HUMAN_REVIEW'.
    These are the borderline cases that need a human decision.
    """
    profiles = (
        db.query(Profile)
        .filter(Profile.recommended_action == "HUMAN_REVIEW")
        .order_by(Profile.analyzed_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return profiles


@router.get("/stats")
def get_review_stats(
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """High-level counts across all risk levels — useful for a dashboard."""
    total = db.query(Profile).count()
    high = db.query(Profile).filter(Profile.risk_level == "HIGH").count()
    borderline = db.query(Profile).filter(Profile.risk_level == "BORDERLINE").count()
    low = db.query(Profile).filter(Profile.risk_level == "LOW").count()
    pending_review = db.query(Profile).filter(Profile.recommended_action == "HUMAN_REVIEW").count()

    return {
        "total_analyzed": total,
        "high_risk": high,
        "borderline": borderline,
        "low_risk": low,
        "pending_human_review": pending_review,
    }


@router.post("/{profile_id}/decide", response_model=ProfileResponse)
def decide_on_profile(
    profile_id: int,
    decision: ReviewDecision,
    db: Session = Depends(get_db),
    reviewer: str = Depends(require_api_key),
):
    """
    Mark a borderline profile as either APPROVED (legitimate) or FLAGGED (fake).
    Updates recommended_action accordingly so it's removed from the queue.
    """
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    decision_upper = decision.decision.upper()
    if decision_upper not in ("APPROVE", "FLAG"):
        raise HTTPException(status_code=400, detail="Decision must be 'APPROVE' or 'FLAG'")

    profile.recommended_action = "NO_ACTION" if decision_upper == "APPROVE" else "FLAG"

    # Append reviewer note to evidence log
    note_entry = {
        "type": "human_review",
        "decision": decision_upper,
        "reviewer": reviewer,
        "note": decision.reviewer_note,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if profile.evidence is None:
        profile.evidence = []
    profile.evidence = profile.evidence + [note_entry]  # reassign to trigger JSON change detection

    db.commit()
    db.refresh(profile)
    return profile
