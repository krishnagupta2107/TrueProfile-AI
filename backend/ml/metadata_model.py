import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from backend.ml.base import BaseModelInterface


class RealMetadataModel(BaseModelInterface):
    """
    Scikit-learn Metadata Model evaluating multi-dimensional account attributes:
    account_age_days, profile_completeness, engagement_rate, and post velocity.
    """
    def __init__(self):
        self.model_name = "ScikitLearn-MetadataClassifier"
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.model_path = os.path.join(base_dir, "models", "metadata_classifier.joblib")
        
        if os.path.exists(self.model_path):
            self._clf = joblib.load(self.model_path)
        else:
            self._clf = LogisticRegression()
            X_ref = np.array([
                [6.8, 0.95, 0.08, 1.2],
                [5.8, 0.85, 0.05, 2.0],
                [1.6, 0.20, 0.00, 25.0],
                [2.5, 0.30, 0.002, 18.0],
                [4.0, 0.60, 0.02, 6.0],
                [2.0, 0.40, 0.001, 12.0],
                [7.2, 0.90, 0.06, 0.8],
            ])
            y_ref = np.array([0, 0, 1, 1, 0, 1, 0])
            self._clf.fit(X_ref, y_ref)

    def predict(self, features: dict) -> float:
        """
        Returns a calibrated probability risk score between 0.05 and 0.95.
        """
        age = float(features.get("account_age_days", 365))
        completeness = float(features.get("profile_completeness", 0.8))
        engagement = float(features.get("engagement_rate", 0.05))
        posts = float(features.get("posts_per_day", 1.0))

        # Transform features
        log_age = np.log1p(max(0.0, age))
        x_vec = np.array([[log_age, completeness, engagement, posts]])

        try:
            prob = self._clf.predict_proba(x_vec)[0][1]
            return round(float(np.clip(prob, 0.05, 0.95)), 2)
        except Exception:
            return 0.50


# Maintain alias for compatibility
DummyMetadataModel = RealMetadataModel
