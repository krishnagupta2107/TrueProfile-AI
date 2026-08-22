from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from backend.database import Base

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True, nullable=True)
    
    # Raw features (example fields for the data generator)
    account_age_days = Column(Integer, default=0)
    followers = Column(Integer, default=0)
    following = Column(Integer, default=0)
    posts_per_day = Column(Float, default=0.0)
    profile_image_url = Column(String, nullable=True)

    # Component Scores (0.0 to 1.0)
    face_score = Column(Float, nullable=True)
    deepfake_score = Column(Float, nullable=True)
    behavior_score = Column(Float, nullable=True)
    metadata_score = Column(Float, nullable=True)
    network_score = Column(Float, nullable=True)

    # Final Risk Assessment
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True) # HIGH, BORDERLINE, LOW
    
    # Evidence array stored as JSON
    evidence = Column(JSON, default=list)

    # Audit fields
    model_version = Column(String, default="v0.1-weighted")
    created_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)
