import os
import sys
import random
from datetime import datetime

# Add the root directory to sys.path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import SessionLocal, engine, Base
from backend.models.profile import Profile

def create_tables():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)

def generate_legitimate_profile():
    # Legitimate users usually have older accounts, moderate followers/following ratio, and normal posting behavior.
    return Profile(
        username=f"real_user_{random.randint(1000, 9999)}",
        account_age_days=random.randint(365, 2000),
        followers=random.randint(100, 2000),
        following=random.randint(100, 500),
        posts_per_day=round(random.uniform(0.1, 2.5), 2),
        profile_image_url="https://example.com/real_profile.jpg"
    )

def generate_fake_profile():
    # Fake accounts often have very low age, high following compared to followers, and erratic posting behavior.
    return Profile(
        username=f"bot_{random.randint(1000, 9999)}",
        account_age_days=random.randint(1, 30),
        followers=random.randint(0, 50),
        following=random.randint(1000, 5000),
        posts_per_day=round(random.uniform(10.0, 50.0), 2),
        profile_image_url="https://example.com/fake_profile.jpg"
    )

def seed_database(num_profiles=20):
    db = SessionLocal()
    print(f"Seeding database with {num_profiles} profiles...")
    
    profiles = []
    for _ in range(num_profiles):
        # 30% chance of fake profile
        is_fake = random.random() < 0.3
        
        if is_fake:
            profile = generate_fake_profile()
        else:
            profile = generate_legitimate_profile()
            
        profiles.append(profile)

    db.add_all(profiles)
    db.commit()
    print("Database seeding completed.")
    db.close()

if __name__ == "__main__":
    create_tables()
    seed_database(20)
