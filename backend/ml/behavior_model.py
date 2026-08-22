from backend.ml.base import BaseModelInterface

class DummyBehaviorModel(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Feature-derived dummy score.
        High posts per day or high following-to-follower ratio increases the risk score.
        """
        score = 0.1
        
        posts_per_day = features.get("posts_per_day", 0.0)
        followers = features.get("followers", 1)
        following = features.get("following", 0)
        
        # Penalize high posting frequency
        if posts_per_day > 10.0:
            score += 0.4
        elif posts_per_day > 3.0:
            score += 0.2
            
        # Penalize high following to followers ratio
        ratio = following / (followers + 1)
        if ratio > 10:
            score += 0.4
        elif ratio > 3:
            score += 0.2
            
        return min(score, 1.0)
