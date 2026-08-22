import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuration
NUM_SAMPLES = 5000
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "behavior_xgboost.json")

def generate_synthetic_data(num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic social media behavior data.
    Label 1 = Fake/Bot, Label 0 = Legitimate
    """
    data = []
    
    for _ in range(num_samples):
        is_bot = random.random() < 0.3  # 30% fake accounts
        
        if is_bot:
            # Bot characteristics
            account_age_days = random.randint(0, 30)
            followers = random.randint(0, 50)
            following = random.randint(1000, 5000)
            posts_per_day = random.uniform(10.0, 50.0)
            profile_completeness = random.uniform(0.1, 0.5)
            follow_burst_rate = random.uniform(0.7, 1.0)
            posting_variance = random.uniform(0.6, 1.0) # Highly variance or highly uniform (we just use high variance)
            engagement_rate = random.uniform(0.0, 0.02)
        else:
            # Legitimate user characteristics
            account_age_days = random.randint(30, 3000)
            followers = random.randint(50, 5000)
            following = random.randint(50, 1000)
            posts_per_day = random.uniform(0.1, 3.0)
            profile_completeness = random.uniform(0.7, 1.0)
            follow_burst_rate = random.uniform(0.0, 0.2)
            posting_variance = random.uniform(0.1, 0.4)
            engagement_rate = random.uniform(0.05, 0.2)
            
        data.append({
            "account_age_days": account_age_days,
            "followers": followers,
            "following": following,
            "posts_per_day": posts_per_day,
            "profile_completeness": profile_completeness,
            "follow_burst_rate": follow_burst_rate,
            "posting_variance": posting_variance,
            "engagement_rate": engagement_rate,
            "is_bot": int(is_bot)
        })
        
    return pd.DataFrame(data)

def main():
    print(f"Generating {NUM_SAMPLES} synthetic profiles...")
    df = generate_synthetic_data(NUM_SAMPLES)
    
    features = [
        "account_age_days", 
        "followers", 
        "following", 
        "posts_per_day", 
        "profile_completeness", 
        "follow_burst_rate", 
        "posting_variance", 
        "engagement_rate"
    ]
    
    X = df[features]
    y = df["is_bot"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier...")
    # Configure the XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the model in JSON format (recommended by XGBoost for interoperability)
    model.save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
