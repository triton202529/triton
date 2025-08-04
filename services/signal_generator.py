import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/processed_data.csv"
OUTPUT_DIR = "latest_predictions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predicted_signals.csv")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

# Features and target
features = ["open", "high", "low", "close", "volume"]
target = "target"

# Optional: scale features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model (optional for later)
joblib.dump(model, "model/random_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Predict signals on entire dataset
predictions = model.predict(X)
df["predicted_signal"] = predictions

# Save predictions
signal_output = df[["date", "symbol", "close", "predicted_signal"]]
signal_output.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Predicted signals saved to {OUTPUT_FILE}")
