from fastapi.testclient import TestClient
from backend.main import app
from backend.database import Base, engine

# Create tables in the test database
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
        "profile_image_url": "http://example.com/legit.jpg"
    }
    
    response = client.post("/profiles/analyze", json=profile_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_score" in data
    assert data["username"] == "test_user_legit"
    # A legitimate profile should ideally not be HIGH risk
    assert data["risk_level"] in ["LOW", "BORDERLINE"]

def test_analyze_profile_fake():
    profile_data = {
        "username": "bot_spammer",
        "account_age_days": 2,
        "followers": 5,
        "following": 4000,
        "posts_per_day": 25.0,
        "profile_image_url": "http://example.com/bot.jpg"
    }
    
    response = client.post("/profiles/analyze", json=profile_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_score" in data
    # High posting frequency and high following to followers ratio should flag it
    assert data["risk_level"] in ["HIGH", "BORDERLINE"]
    assert "evidence" in data
    assert len(data["evidence"]) > 0
