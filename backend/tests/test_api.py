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
    assert response.json() == {"message": "Welcome to TrueProfile AI backend"}


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


def test_analyze_by_username():
    payload = {"username": "bot_clone_99"}
    response = client.post("/profiles/analyze/username", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "bot_clone_99"
    assert "risk_score" in data
    assert data["risk_level"] in ["HIGH", "BORDERLINE", "LOW"]
