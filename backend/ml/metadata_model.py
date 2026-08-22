from backend.ml.base import BaseModelInterface

class DummyMetadataModel(BaseModelInterface):
    def predict(self, features: dict) -> float:
        """
        Feature-derived dummy score.
        Low account age increases the risk score.
        """
        score = 0.1
        
        account_age_days = features.get("account_age_days", 0)
        
        if account_age_days < 7:
            score += 0.7
        elif account_age_days < 30:
            score += 0.4
        elif account_age_days < 90:
            score += 0.2
            
        return min(score, 1.0)
