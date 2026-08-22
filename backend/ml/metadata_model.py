from backend.ml.base import BaseModelInterface


class DummyMetadataModel(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Feature-derived dummy metadata score.
        Uses: account_age_days, profile_completeness, engagement_rate.
        """
        score = 0.05

        # Very new accounts are suspicious
        account_age_days = features.get("account_age_days", 0)
        if account_age_days < 7:
            score += 0.55
        elif account_age_days < 30:
            score += 0.35
        elif account_age_days < 90:
            score += 0.15

        # Low profile completeness is suspicious
        completeness = features.get("profile_completeness", 1.0)
        if completeness < 0.3:
            score += 0.25
        elif completeness < 0.6:
            score += 0.10

        # Very low engagement rate suggests bot or purchased followers
        engagement = features.get("engagement_rate", 1.0)
        if engagement < 0.003:
            score += 0.15
        elif engagement < 0.01:
            score += 0.05

        return min(round(score, 3), 1.0)
