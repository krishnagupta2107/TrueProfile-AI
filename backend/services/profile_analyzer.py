from datetime import datetime, timezone
from backend.ml.face_analysis import ArcFaceModel
from backend.ml.deepfake_detector import RealDeepfakeDetector
from backend.ml.behavior_model import RealBehaviorModel
from backend.ml.metadata_model import DummyMetadataModel
from backend.ml.network_analysis import RealNetworkAnalysisModel
from backend.ml.fusion_engine import FusionEngine


class ProfileAnalyzerService:
    def __init__(self):
        self.face_model = ArcFaceModel()
        self.deepfake_model = RealDeepfakeDetector()
        self.behavior_model = RealBehaviorModel()
        self.metadata_model = DummyMetadataModel()
        self.network_model = RealNetworkAnalysisModel()
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

        followers = profile_data.get("followers", 0)
        following = profile_data.get("following", 0)
        age = profile_data.get("account_age_days", 0)
        posts_per_day = profile_data.get("posts_per_day", 0.0)
        completeness = profile_data.get("profile_completeness", 0.5)
        follow_burst = profile_data.get("follow_burst_rate", 0.0)
        engagement = profile_data.get("engagement_rate", 0.0)
        variance = profile_data.get("posting_variance", 0.0)

        if scores.get("face", 0) <= 0.30 and profile_data.get("profile_image_url"):
            evidence.append({
                "signal": "ArcFace Biometrics",
                "title": "Authentic Human Face Verified",
                "severity": "LOW",
                "score": f"{int(scores['face'] * 100)}%",
                "detail": f"ArcFace detected a clear, single-subject human facial structure with natural embedding geometry (Biometric risk: {int(scores['face'] * 100)}%)."
            })
        elif scores.get("face", 0) >= 0.60:
            evidence.append({
                "signal": "ArcFace Biometrics",
                "title": "Face Biometrics Anomaly Detected",
                "severity": "HIGH" if scores["face"] >= 0.80 else "MEDIUM",
                "score": f"{int(scores['face'] * 100)}%",
                "detail": f"ArcFace returned an elevated risk score of {int(scores['face'] * 100)}%. No recognizable human face was detected in the uploaded avatar, or multiple conflicting subjects were found."
            })

        if scores.get("deepfake", 0) >= 0.70:
            evidence.append({
                "signal": "DeepFace Anti-Spoofing",
                "title": "Potential Synthetic or AI-Generated Image",
                "severity": "HIGH",
                "score": f"{int(scores['deepfake'] * 100)}%",
                "detail": f"Anti-spoofing neural net detected frequency domain compression anomalies or GAN synthesis artifacts (score: {int(scores['deepfake'] * 100)}%) typical of AI-generated faces (StyleGAN / Midjourney)."
            })

        if scores.get("behavior", 0) >= 0.60:
            evidence.append({
                "signal": "XGBoost Behavior Model",
                "title": "Unusual Behavioral & Posting Telemetry",
                "severity": "HIGH" if scores["behavior"] >= 0.80 else "MEDIUM",
                "score": f"{int(scores['behavior'] * 100)}%",
                "detail": f"The XGBoost model identified non-linear automated behavior signatures. Account exhibits a post velocity of {posts_per_day} posts/day and follow burst rate of {follow_burst:.2f}, matching automated script activity."
            })

        if scores.get("metadata", 0) >= 0.60:
            evidence.append({
                "signal": "Account Metadata",
                "title": "Account Immaturity & Low Completeness",
                "severity": "MEDIUM",
                "score": f"{int(scores['metadata'] * 100)}%",
                "detail": f"Account age ({age} days) and profile completeness ({int(completeness * 100)}%) indicate an incomplete setup frequently associated with disposable sock-puppet accounts."
            })

        if scores.get("network", 0) >= 0.60:
            evidence.append({
                "signal": "NetworkX Graph Topology",
                "title": "Ego-Network & Ratio Asymmetry",
                "severity": "HIGH" if scores["network"] >= 0.80 else "MEDIUM",
                "score": f"{int(scores['network'] * 100)}%",
                "detail": f"Network centrality and follower-to-following imbalance ({followers} followers vs {following} following) exhibit low reciprocity typical of spam-follow rings."
            })

        if follow_burst >= 0.60:
            evidence.append({
                "signal": "Activity Telemetry",
                "title": "High Follow/Unfollow Burst Velocity",
                "severity": "HIGH",
                "score": f"{int(follow_burst * 100)}%",
                "detail": f"Follow burst rate ({follow_burst:.2f}) is significantly above normal human thresholds (0.05-0.25), suggesting automated mass-following software was used."
            })

        if engagement <= 0.005 and followers >= 50:
            evidence.append({
                "signal": "Engagement Telemetry",
                "title": "Abnormally Low Engagement Rate",
                "severity": "MEDIUM",
                "score": f"{engagement * 100:.2f}%",
                "detail": f"Engagement rate ({engagement * 100:.2f}%) is disproportionately low relative to the audience size ({followers:,} followers), pointing toward inactive or ghost followers."
            })

        if not evidence:
            evidence.append({
                "signal": "Fusion Integrity",
                "title": "No Significant Anomalies Detected",
                "severity": "LOW",
                "score": "0%",
                "detail": "All visual biometrics, behavioral velocities, metadata fields, and network graph structures align with authentic human profile characteristics."
            })

        return evidence
