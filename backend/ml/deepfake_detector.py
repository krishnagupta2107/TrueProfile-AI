from backend.ml.base import BaseModelInterface
import random

class DummyDeepfakeDetector(BaseModelInterface):
    """
    Placeholder for a real Deepfake detection model (e.g. MesoNet, XceptionNet).
    Currently returns simulated scores.
    """
    def __init__(self):
        self.model_name = "DeepfakeDetector-Mock"
        
    def predict(self, features: dict) -> float:
        """
        Returns a mock deepfake probability score based on the username for deterministic testing.
        """
        username = features.get("username", "")
        if "bot" in username.lower():
            return round(random.uniform(0.6, 0.9), 2)
        return round(random.uniform(0.01, 0.2), 2)
