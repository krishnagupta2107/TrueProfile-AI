from datetime import datetime
from backend.ml.face_analysis import DummyFaceAnalysisModel
from backend.ml.deepfake_detector import DummyDeepfakeDetector
from backend.ml.behavior_model import DummyBehaviorModel
from backend.ml.metadata_model import DummyMetadataModel
from backend.ml.network_analysis import DummyNetworkAnalysisModel
from backend.ml.fusion_engine import FusionEngine

class ProfileAnalyzerService:
    def __init__(self):
        # Initialize models (currently dummies)
        self.face_model = DummyFaceAnalysisModel()
        self.deepfake_model = DummyDeepfakeDetector()
        self.behavior_model = DummyBehaviorModel()
        self.metadata_model = DummyMetadataModel()
        self.network_model = DummyNetworkAnalysisModel()
        
        self.fusion_engine = FusionEngine()
        self.model_version = "v0.1-weighted"
        
    def analyze_profile(self, profile_data: dict) -> dict:
        """
        Orchestrates the ML pipeline to analyze a profile and return component scores, final risk, and evidence.
        """
        # 1. Gather component scores
        scores = {
            "face": self.face_model.predict(profile_data),
            "deepfake": self.deepfake_model.predict(profile_data),
            "behavior": self.behavior_model.predict(profile_data),
            "metadata": self.metadata_model.predict(profile_data),
            "network": self.network_model.predict(profile_data)
        }
        
        # 2. Fusion
        risk_score = self.fusion_engine.calculate_risk(scores)
        risk_level = self.fusion_engine.determine_risk_level(risk_score)
        
        # 3. Generate evidence
        evidence = self._generate_evidence(scores, profile_data)
        
        return {
            "scores": scores,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "evidence": evidence,
            "model_version": self.model_version,
            "analyzed_at": datetime.utcnow()
        }
        
    def _generate_evidence(self, scores: dict, profile_data: dict) -> list:
        evidence = []
        if scores["behavior"] > 0.7:
            evidence.append("Unusual posting frequency or follow behavior")
        if scores["metadata"] > 0.7:
            evidence.append("Account is very new or lacks completeness")
        if scores["deepfake"] > 0.7:
            evidence.append("Possible synthetic profile image detected")
        if scores["network"] > 0.7:
            evidence.append("Abnormal follow/unfollow network activity")
        if not evidence and scores["face"] > 0.5:
            evidence.append("Facial similarity flag raised")
            
        if not evidence:
            evidence.append("No significant anomalies detected")
            
        return evidence
