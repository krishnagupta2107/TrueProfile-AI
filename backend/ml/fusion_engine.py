class FusionEngine:
    def __init__(self):
        # Initial heuristic weights
        self.weights = {
            "face": 0.30,
            "deepfake": 0.20,
            "behavior": 0.25,
            "metadata": 0.15,
            "network": 0.10
        }
        
    def calculate_risk(self, scores: dict) -> float:
        """
        Calculates final risk score using weighted average.
        Expects a dictionary with keys: face, deepfake, behavior, metadata, network.
        """
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
