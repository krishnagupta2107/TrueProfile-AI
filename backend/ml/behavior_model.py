from backend.ml.base import BaseModelInterface


class DummyBehaviorModel(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Feature-derived dummy behavior score.
        Uses: posts_per_day, followers/following ratio, follow_burst_rate, posting_variance.
        """
        score = 0.05

        # High posting frequency is suspicious
        posts_per_day = features.get("posts_per_day", 0.0)
        if posts_per_day > 20.0:
            score += 0.35
        elif posts_per_day > 10.0:
            score += 0.20
        elif posts_per_day > 5.0:
            score += 0.10

        # High following-to-follower ratio is suspicious
        followers = max(features.get("followers", 1), 1)
        following = features.get("following", 0)
        ratio = following / followers
        if ratio > 15:
            score += 0.35
        elif ratio > 5:
            score += 0.20
        elif ratio > 2:
            score += 0.10

        # High follow burst rate is suspicious
        follow_burst = features.get("follow_burst_rate", 0.0)
        score += follow_burst * 0.15

        # High posting variance is suspicious
        posting_variance = features.get("posting_variance", 0.0)
        score += posting_variance * 0.10

        return min(round(score, 3), 1.0)
