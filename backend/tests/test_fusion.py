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
    # 0.9 * sum of weights (1.0) = 0.9
    assert abs(risk - 0.9) < 0.001
    assert engine.determine_risk_level(risk) == "HIGH"

def test_calculate_risk_low():
    engine = FusionEngine()
    scores = {
        "face": 0.1,
        "deepfake": 0.1,
        "behavior": 0.1,
        "metadata": 0.1,
        "network": 0.1
    }
    risk = engine.calculate_risk(scores)
    assert abs(risk - 0.1) < 0.001
    assert engine.determine_risk_level(risk) == "LOW"

def test_calculate_risk_mixed():
    engine = FusionEngine()
    # Face(0.3)+Deepfake(0.2)+Behavior(0.25)+Metadata(0.15)+Network(0.1)
    # 0.5*0.3 + 0.1*0.2 + 0.8*0.25 + 0.2*0.15 + 0.9*0.1
    # 0.15 + 0.02 + 0.20 + 0.03 + 0.09 = 0.49
    scores = {
        "face": 0.5,
        "deepfake": 0.1,
        "behavior": 0.8,
        "metadata": 0.2,
        "network": 0.9
    }
    risk = engine.calculate_risk(scores)
    assert abs(risk - 0.49) < 0.001
    assert engine.determine_risk_level(risk) == "LOW"
