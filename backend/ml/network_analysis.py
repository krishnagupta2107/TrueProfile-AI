from backend.ml.base import BaseModelInterface
import random

class DummyNetworkAnalysisModel(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Network anomaly score based on followers and following.
        """
        followers = features.get("followers", 1)
        following = features.get("following", 0)
        
        # High following but very low followers indicates network anomaly
        if following > 1000 and followers < 50:
            return round(random.uniform(0.7, 0.95), 2)
        return round(random.uniform(0.1, 0.3), 2)
