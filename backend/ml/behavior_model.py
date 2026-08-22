import os
import numpy as np
import pandas as pd
import xgboost as xgb
from backend.ml.base import BaseModelInterface


class RealBehaviorModel(BaseModelInterface):
    """
    Real behavioral analysis using a pre-trained XGBoost model.
    """
    def __init__(self):
        self.model_name = "XGBoost-Behavior"
        self.model = None
        self.features = [
            "account_age_days", 
            "followers", 
            "following", 
            "posts_per_day", 
            "profile_completeness", 
            "follow_burst_rate", 
            "posting_variance", 
            "engagement_rate"
        ]
        
        # Determine model path relative to this file
        # backend/ml/behavior_model.py -> ../../models/behavior_xgboost.json
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.model_path = os.path.join(base_dir, "models", "behavior_xgboost.json")
        
        # Load the model if it exists
        if os.path.exists(self.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
        else:
            print(f"Warning: XGBoost model not found at {self.model_path}")

    def predict(self, features: dict) -> float:
        """
        Returns the predicted probability of the account being a bot (0.0 to 1.0).
        """
        if self.model is None:
            # Fallback if model hasn't been trained yet
            return 0.5
            
        # Extract the required features in the correct order
        # Default to neutral/safe values if missing
        feature_values = []
        feature_values.append(features.get("account_age_days", 365))
        feature_values.append(features.get("followers", 100))
        feature_values.append(features.get("following", 100))
        feature_values.append(features.get("posts_per_day", 1.0))
        feature_values.append(features.get("profile_completeness", 0.8))
        feature_values.append(features.get("follow_burst_rate", 0.0))
        feature_values.append(features.get("posting_variance", 0.1))
        feature_values.append(features.get("engagement_rate", 0.1))
        
        # Convert to numpy array with shape (1, num_features)
        X = np.array([feature_values])
        
        try:
            # Get the probability of the positive class (class 1 = fake/bot)
            probabilities = self.model.predict_proba(X)
            # probabilities is shape (1, 2) where col 0 is class 0, col 1 is class 1
            risk_score = float(probabilities[0][1])
            return round(risk_score, 2)
        except Exception as e:
            print(f"Error predicting with XGBoost: {e}")
            return 0.5
