from backend.ml.fusion_engine import FusionEngine

def test_fusion_engine_weights():
    engine = FusionEngine()
    total_weight = sum(engine.weights.values())
    # Weights should sum to 1.0 (or very close due to float arithmetic)
    assert abs(total_weight - 1.0) < 0.001

def test_calculate_risk_high():
    engine = FusionEngine()
    scores = {
        "face": 0.9,
        "deepfake": 0.9,
        "behavior": 0.9,
        "metadata": 0.9,
        "network": 0.9
    }
    risk = engine.calculate_risk(scores)
    # All scores are high -> final risk should be high (>= 0.5)
    assert 0.0 <= risk <= 1.0
    assert risk >= 0.5, f"Expected high risk, got {risk}"
    assert engine.determine_risk_level(risk) in ("HIGH", "BORDERLINE")

def test_calculate_risk_low():
    engine = FusionEngine()
    scores = {
        "face": 0.05,
        "deepfake": 0.05,
        "behavior": 0.05,
        "metadata": 0.05,
        "network": 0.05
    }
    risk = engine.calculate_risk(scores)
    # All scores are very low -> final risk should be low (< 0.5)
    assert 0.0 <= risk <= 1.0
    assert risk < 0.5, f"Expected low risk, got {risk}"
    assert engine.determine_risk_level(risk) == "LOW"

def test_calculate_risk_mixed():
    engine = FusionEngine()
    high_scores = {
        "face": 0.9,
        "deepfake": 0.9,
        "behavior": 0.9,
        "metadata": 0.9,
        "network": 0.9
    }
    low_scores = {
        "face": 0.05,
        "deepfake": 0.05,
        "behavior": 0.05,
        "metadata": 0.05,
        "network": 0.05
    }
    # The engine should score high-input profiles higher than low-input ones
    risk_high = engine.calculate_risk(high_scores)
    risk_low = engine.calculate_risk(low_scores)
    assert risk_high > risk_low, f"Expected risk_high ({risk_high}) > risk_low ({risk_low})"
    assert 0.0 <= risk_high <= 1.0
    assert 0.0 <= risk_low <= 1.0
