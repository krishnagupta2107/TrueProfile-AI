from backend.ml.base import BaseModelInterface
import random

class DummyDeepfakeDetector(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Since we don't have images yet, we base this on the username containing 'bot' or random chance for now.
        """
        username = features.get("username", "")
        if "bot" in username.lower():
            return round(random.uniform(0.6, 0.99), 2)
        return round(random.uniform(0.01, 0.2), 2)
