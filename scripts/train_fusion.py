import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

NUM_SAMPLES = 10000
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fusion_meta_model.pkl")

def generate_synthetic_subscores(num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic sub-scores from the 5 ML modules.
    Label 1 = Fake/Bot, Label 0 = Legitimate
    """
    data = []
    
    for _ in range(num_samples):
        is_bot = random.random() < 0.25
        
        if is_bot:
            # High risk scores across modules (but with some variance)
            face_score = min(max(random.normalvariate(0.8, 0.2), 0.0), 1.0)
            deepfake_score = min(max(random.normalvariate(0.7, 0.25), 0.0), 1.0)
            behavior_score = min(max(random.normalvariate(0.85, 0.15), 0.0), 1.0)
            metadata_score = min(max(random.normalvariate(0.75, 0.2), 0.0), 1.0)
            network_score = min(max(random.normalvariate(0.9, 0.1), 0.0), 1.0)
        else:
            # Low risk scores
            face_score = min(max(random.normalvariate(0.1, 0.1), 0.0), 1.0)
            deepfake_score = min(max(random.normalvariate(0.1, 0.1), 0.0), 1.0)
            behavior_score = min(max(random.normalvariate(0.2, 0.2), 0.0), 1.0)
            metadata_score = min(max(random.normalvariate(0.2, 0.2), 0.0), 1.0)
            network_score = min(max(random.normalvariate(0.1, 0.1), 0.0), 1.0)
            
        data.append({
            "face_score": face_score,
            "deepfake_score": deepfake_score,
            "behavior_score": behavior_score,
            "metadata_score": metadata_score,
            "network_score": network_score,
            "is_bot": int(is_bot)
        })
        
    return pd.DataFrame(data)

def weighted_average_baseline(X):
    # Current baseline logic (Face 30%, Deepfake 20%, Behavior 25%, Metadata 15%, Network 10%)
    return (X['face_score'] * 0.30 +
            X['deepfake_score'] * 0.20 +
            X['behavior_score'] * 0.25 +
            X['metadata_score'] * 0.15 +
            X['network_score'] * 0.10)

def main():
    print(f"Generating {NUM_SAMPLES} synthetic sub-scores...")
    df = generate_synthetic_subscores(NUM_SAMPLES)
    
    features = ["face_score", "deepfake_score", "behavior_score", "metadata_score", "network_score"]
    
    X = df[features]
    y = df["is_bot"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Evaluating Weighted Average Baseline ---")
    y_pred_baseline_prob = weighted_average_baseline(X_test)
    y_pred_baseline = (y_pred_baseline_prob >= 0.5).astype(int)
    print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
    print(f"Baseline AUC-ROC:  {roc_auc_score(y_test, y_pred_baseline_prob):.4f}")
    
    print("\n--- Training ML Fusion Classifier ---")
    # Using Logistic Regression because it's highly interpretable, fast, and 
    # essentially learns the optimal "weights" instead of our hardcoded ones.
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_ml = model.predict(X_test)
    y_pred_ml_prob = model.predict_proba(X_test)[:, 1]
    
    acc_ml = accuracy_score(y_test, y_pred_ml)
    auc_ml = roc_auc_score(y_test, y_pred_ml_prob)
    
    print(f"ML Fusion Accuracy: {acc_ml:.4f}")
    print(f"ML Fusion AUC-ROC:  {auc_ml:.4f}")
    
    print("\nLearned Weights:")
    for feature, weight in zip(features, model.coef_[0]):
        print(f"  {feature}: {weight:.4f}")
    print(f"  Bias (intercept): {model.intercept_[0]:.4f}")
    
    print("\nSaving ML Fusion model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
