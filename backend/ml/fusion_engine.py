import os
import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class FusionEngine:
    def __init__(self):
        # Heuristic fallback weights (Face 30%, Deepfake 20%, Behavior 25%, Metadata 15%, Network 10%)
        self.weights = {
            "face": 0.30,
            "deepfake": 0.20,
            "behavior": 0.25,
            "metadata": 0.15,
            "network": 0.10
        }
        self.meta_model = None

        # Load ML meta-classifier if available
        if _JOBLIB_AVAILABLE:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, "models", "fusion_meta_model.pkl")
            if os.path.exists(model_path):
                try:
                    self.meta_model = joblib.load(model_path)
                except Exception as e:
                    print(f"Warning: Could not load ML fusion model: {e}")

    def calculate_risk(self, scores: dict) -> float:
        """
        Calculates final risk score using ML meta-classifier if available,
        otherwise falls back to the weighted average baseline.
        Expects a dictionary with keys: face, deepfake, behavior, metadata, network.
        """
        if self.meta_model is not None:
            try:
                # Order matches training: face, deepfake, behavior, metadata, network
                feature_vector = np.array([[scores.get(k, 0.0) for k in self.weights.keys()]])
                prob = self.meta_model.predict_proba(feature_vector)[0][1]
                return round(float(prob), 2)
            except Exception as e:
                print(f"ML fusion predict failed, falling back to weighted average: {e}")

        # Fallback: Weighted average
        final_score = 0.0
        for component, weight in self.weights.items():
            final_score += scores.get(component, 0.0) * weight

        return round(final_score, 2)

    def determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.85:
            return "HIGH"
        elif risk_score >= 0.50:
            return "BORDERLINE"
        else:
            return "LOW"
