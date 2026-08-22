import random
import hashlib
from typing import Dict, Any


def ingest_profile_by_username(username: str, platform: str = "instagram") -> Dict[str, Any]:
    """
    Automated profile ingestion engine.
    Given ONLY a platform and username, fetches/extracts all necessary 
    account telemetry, graph signals, and media features.
    
    In Phase 2, this hooks into official Graph/OAuth APIs or proxy scrapers.
    Currently generates deterministic, correlated profile feature vectors.
    """
    clean_username = username.strip().lstrip("@")
    platform = platform.lower().strip()
    
    # Deterministic hash seed so the exact same username always yields reproducible analysis
    seed_str = f"{platform}:{clean_username.lower()}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    # Detect bot / fake patterns in handle
    is_bot_pattern = any(kw in clean_username.lower() for kw in [
        "bot", "fake", "spam", "auto", "clone", "crypto", "free", "dm_me", "promo", "giveaway", "follower", "boost"
    ]) or (sum(c.isdigit() for c in clean_username) >= 5)

    is_celebrity_or_official = clean_username.lower() in [
        "elonmusk", "cristiano", "nasa", "billgates", "google", "openai", "taylorswift", "barackobama", "mrbeast"
    ]

    # Avatar generation based on handle
    avatar_color = rng.choice(["6366f1", "ec4899", "8b5cf6", "3b82f6", "10b981", "f59e0b", "ef4444"])
    avatar_url = f"https://ui-avatars.com/api/?name={clean_username}&background={avatar_color}&color=fff&size=200&bold=true"

    if is_celebrity_or_official:
        account_age_days = rng.randint(2500, 5000)
        followers = rng.randint(5_000_000, 150_000_000)
        following = rng.randint(50, 800)
        posts_per_day = round(rng.uniform(0.5, 3.5), 2)
        profile_completeness = 1.0
        follow_burst_rate = round(rng.uniform(0.01, 0.05), 2)
        posting_variance = round(rng.uniform(0.1, 0.3), 2)
        engagement_rate = round(rng.uniform(0.03, 0.08), 4)
    elif is_bot_pattern:
        account_age_days = rng.randint(1, 45)
        followers = rng.randint(0, 85)
        following = rng.randint(2000, 7500)
        posts_per_day = round(rng.uniform(12.0, 55.0), 2)
        profile_completeness = round(rng.uniform(0.1, 0.35), 2)
        follow_burst_rate = round(rng.uniform(0.75, 1.0), 2)
        posting_variance = round(rng.uniform(0.7, 1.0), 2)
        engagement_rate = round(rng.uniform(0.0005, 0.008), 4)
    else:
        # Standard human user with natural variances
        account_age_days = rng.randint(150, 2200)
        followers = rng.randint(120, 3500)
        following = rng.randint(100, 950)
        posts_per_day = round(rng.uniform(0.1, 2.5), 2)
        profile_completeness = round(rng.uniform(0.65, 0.95), 2)
        follow_burst_rate = round(rng.uniform(0.05, 0.25), 2)
        posting_variance = round(rng.uniform(0.1, 0.4), 2)
        engagement_rate = round(rng.uniform(0.025, 0.12), 4)

    return {
        "username": clean_username,
        "platform": platform,
        "account_age_days": account_age_days,
        "followers": followers,
        "following": following,
        "posts_per_day": posts_per_day,
        "profile_completeness": profile_completeness,
        "follow_burst_rate": follow_burst_rate,
        "posting_variance": posting_variance,
        "engagement_rate": engagement_rate,
        "profile_image_url": avatar_url,
    }
