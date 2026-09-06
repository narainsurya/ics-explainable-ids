import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Fix the import path so it works from anywhere
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.load_data import load_dataset

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models", "saved")

def preprocess_data():
    """Cleans, splits, and scales the dataset for the LSTM Autoencoder."""
    print("Loading mapped dataset...")
    df = load_dataset()

    # 1. Split features and labels
    # We drop 'time' and 'attack_type' as they aren't fed into the neural network
    features = ['pressure', 'flow_rate', 'temperature', 'command']
    
    # 2. Separate Normal data (for training) and Attack data (for testing)
    normal_data = df[df['label'] == 0].copy()
    attack_data = df[df['label'] == 1].copy()

    # We use 80% of normal data for training
    train_size = int(len(normal_data) * 0.8)
    train_df = normal_data.iloc[:train_size]
    
    # The remaining 20% normal + ALL attack data becomes our test/evaluation set
    test_normal = normal_data.iloc[train_size:]
    test_df = pd.concat([test_normal, attack_data]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training Set (Normal Only): {len(train_df)} rows")
    print(f"Testing Set (Mixed): {len(test_df)} rows")

    # 3. Scale the data
    scaler = MinMaxScaler()
    
    # Fit scaler ONLY on training data to prevent data leakage
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df[features]), columns=features)
    test_scaled = pd.DataFrame(scaler.transform(test_df[features]), columns=features)

    # Re-attach labels to the test set for evaluation later
    test_scaled['label'] = test_df['label']
    test_scaled['attack_type'] = test_df['attack_type']

    # 4. Save processed datasets and the scaler
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_scaled.to_csv(os.path.join(PROCESSED_DIR, "train_scaled.csv"), index=False)
    test_scaled.to_csv(os.path.join(PROCESSED_DIR, "test_scaled.csv"), index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.save"))
    
    print("Preprocessing complete. Scaled data saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()