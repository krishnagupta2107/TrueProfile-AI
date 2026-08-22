import random
import hashlib


def ingest_profile_by_username(username: str) -> dict:
    """
    Simulates scraping a social media profile by username.
    Returns a deterministic set of realistic profile features.
    In Phase 2, replace this with real API calls (Twitter API, Instagram Basic API etc.)
    """
    # Use a seed derived from username so the same username always gives the same data
    seed = int(hashlib.md5(username.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    # Heuristic: usernames with bot-like patterns get worse features
    bot_signals = any(kw in username.lower() for kw in ["bot", "fake", "spam", "auto", "clone"])
    
    if bot_signals:
        return {
            "username": username,
            "account_age_days": rng.randint(1, 20),
            "followers": rng.randint(0, 60),
            "following": rng.randint(1500, 6000),
            "posts_per_day": round(rng.uniform(15.0, 60.0), 2),
            "profile_completeness": round(rng.uniform(0.1, 0.3), 2),
            "follow_burst_rate": round(rng.uniform(0.7, 1.0), 2),
            "posting_variance": round(rng.uniform(0.8, 1.0), 2),
            "engagement_rate": round(rng.uniform(0.001, 0.01), 4),
            "profile_image_url": f"https://example.com/profiles/{username}.jpg",
        }
    else:
        return {
            "username": username,
            "account_age_days": rng.randint(200, 2500),
            "followers": rng.randint(200, 5000),
            "following": rng.randint(100, 600),
            "posts_per_day": round(rng.uniform(0.1, 3.0), 2),
            "profile_completeness": round(rng.uniform(0.6, 1.0), 2),
            "follow_burst_rate": round(rng.uniform(0.0, 0.2), 2),
            "posting_variance": round(rng.uniform(0.0, 0.3), 2),
            "engagement_rate": round(rng.uniform(0.02, 0.15), 4),
            "profile_image_url": f"https://example.com/profiles/{username}.jpg",
        }
