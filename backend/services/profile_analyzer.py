from datetime import datetime, timezone
from backend.ml.face_analysis import ArcFaceModel
from backend.ml.deepfake_detector import RealDeepfakeDetector
from backend.ml.behavior_model import DummyBehaviorModel
from backend.ml.metadata_model import DummyMetadataModel
from backend.ml.network_analysis import DummyNetworkAnalysisModel
from backend.ml.fusion_engine import FusionEngine


class ProfileAnalyzerService:
    def __init__(self):
        self.face_model = ArcFaceModel()
        self.deepfake_model = RealDeepfakeDetector()
        self.behavior_model = DummyBehaviorModel()
        self.metadata_model = DummyMetadataModel()
        self.network_model = DummyNetworkAnalysisModel()
        self.fusion_engine = FusionEngine()
        self.model_version = "v0.1-weighted"

    def analyze_profile(self, profile_data: dict) -> dict:
        """
        Orchestrates the ML pipeline to analyze a profile.
        Expects a dict containing all profile features.
        """
        scores = {
            "face":      self.face_model.predict(profile_data),
            "deepfake":  self.deepfake_model.predict(profile_data),
            "behavior":  self.behavior_model.predict(profile_data),
            "metadata":  self.metadata_model.predict(profile_data),
            "network":   self.network_model.predict(profile_data),
        }

        risk_score = self.fusion_engine.calculate_risk(scores)
        risk_level = self.fusion_engine.determine_risk_level(risk_score)
        recommended_action = self._recommend_action(risk_level)
        evidence = self._generate_evidence(scores, profile_data)

        return {
            "scores": scores,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "evidence": evidence,
            "model_version": self.model_version,
            "analyzed_at": datetime.now(timezone.utc),
        }

    def _recommend_action(self, risk_level: str) -> str:
        mapping = {
            "HIGH": "FLAG",
            "BORDERLINE": "HUMAN_REVIEW",
            "LOW": "NO_ACTION",
        }
        return mapping.get(risk_level, "HUMAN_REVIEW")

    def _generate_evidence(self, scores: dict, profile_data: dict) -> list:
        evidence = []

        if scores["behavior"] > 0.7:
            evidence.append("Unusual posting frequency or follow behavior detected")
        if scores["metadata"] > 0.7:
            evidence.append("Account is very new or has low profile completeness")
        if scores["deepfake"] > 0.7:
            evidence.append("Possible synthetic or AI-generated profile image detected")
        if scores["network"] > 0.7:
            evidence.append("Abnormal follower/following ratio or network clustering")
        if scores["face"] > 0.6:
            evidence.append("Face similarity flag raised against known profiles")

        follow_burst = profile_data.get("follow_burst_rate", 0)
        if follow_burst > 0.6:
            evidence.append("High follow/unfollow burst rate detected")

        engagement = profile_data.get("engagement_rate", 1)
        if engagement < 0.005:
            evidence.append("Abnormally low engagement rate relative to follower count")

        if not evidence:
            evidence.append("No significant anomalies detected")

        return evidence
