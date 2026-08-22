"""
API tests for TrueProfile AI backend.
conftest.py sets DATABASE_URL=sqlite:///:memory: before any imports.
"""
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import Base, engine

# Create tables in test DB (in-memory SQLite, as set by conftest.py)
Base.metadata.create_all(bind=engine)

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "service": "TrueProfile AI",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }


def test_analyze_profile_legitimate():
    profile_data = {
        "username": "test_user_legit",
        "account_age_days": 1000,
        "followers": 1000,
        "following": 300,
        "posts_per_day": 0.5,
        "profile_completeness": 0.9,
        "follow_burst_rate": 0.05,
        "posting_variance": 0.1,
        "engagement_rate": 0.08,
        "profile_image_url": "http://example.com/legit.jpg"
    }
    response = client.post("/profiles/analyze", json=profile_data)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert data["username"] == "test_user_legit"
    assert data["risk_level"] in ["LOW", "BORDERLINE"]
    assert "recommended_action" in data


def test_analyze_profile_fake():
    profile_data = {
        "username": "bot_spammer",
        "account_age_days": 2,
        "followers": 5,
        "following": 4000,
        "posts_per_day": 25.0,
        "profile_completeness": 0.1,
        "follow_burst_rate": 0.9,
        "posting_variance": 0.9,
        "engagement_rate": 0.001,
        "profile_image_url": "http://example.com/bot.jpg"
    }
    response = client.post("/profiles/analyze", json=profile_data)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["HIGH", "BORDERLINE"]
    assert len(data["evidence"]) > 1
    assert data["recommended_action"] in ["FLAG", "HUMAN_REVIEW"]


def test_analyze_manual_multipart():
    form_data = {
        "username": "manual_user_real",
        "account_age_days": "500",
        "followers": "800",
        "following": "300",
        "posts_per_day": "1.5",
        "profile_completeness": "0.85",
        "follow_burst_rate": "0.1",
        "posting_variance": "0.2",
        "engagement_rate": "0.06",
        "profile_image_url": "http://example.com/manual.jpg"
    }
    response = client.post("/profiles/analyze/manual", data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "manual_user_real"
    assert data["followers"] == 800
    assert "risk_score" in data
    assert data["risk_level"] in ["LOW", "BORDERLINE", "HIGH"]


def test_analyze_manual_with_image_upload(tmp_path):
    # Create a small valid test jpeg
    import cv2
    import numpy as np
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test_avatar.jpg")
    cv2.imwrite(img_path, img)

    with open(img_path, "rb") as f:
        response = client.post(
            "/profiles/analyze/manual",
            data={
                "username": "avatar_uploader",
                "account_age_days": "120",
                "followers": "250",
                "following": "180",
                "posts_per_day": "0.8",
            },
            files={"image": ("test_avatar.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "avatar_uploader"
    assert "uploads/" in data["profile_image_url"]
    assert "risk_score" in data


def test_analyze_by_username_deprecated_stub():
    # Stub endpoint returns HTTP 501
    payload = {"username": "old_scraper_call"}
    response = client.post("/profiles/analyze/username", json=payload)
    assert response.status_code == 501
    assert "Automated ingestion not available" in response.json()["detail"]
