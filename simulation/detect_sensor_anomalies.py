
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest

# Load sensor monitoring dataset
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

input_file = os.path.join(
    project_path,
    "dataset",
    "sensor_monitoring_data.csv"
)

df = pd.read_csv(input_file)

# Select sensor measurements for anomaly detection
features = [
    "temperature_c",
    "humidity_percent",
    "pressure_kpa",
    "deformation_mm"
]

X = df[features]

# Train Isolation Forest model
model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

# Predict: 1 = Normal, -1 = Anomaly
predictions = model.fit_predict(X)

df["ai_status"] = np.where(
    predictions == -1,
    "Anomaly",
    "Normal"
)

# Calculate anomaly scores
df["anomaly_score"] = model.decision_function(X)

# Save results
output_file = os.path.join(
    project_path,
    "dataset",
    "sensor_anomaly_results.csv"
)

df.to_csv(output_file, index=False)

# Display results
print("BioGrip+ AI Anomaly Detection Completed!")

print("\nTotal sensor readings:", len(df))

print("\nAI Detection Summary:")
print(df["ai_status"].value_counts())

print("\nDetected anomalies:")
print(
    df[df["ai_status"] == "Anomaly"][
        features + ["anomaly_score"]
    ].to_string(index=False)
)

print("\nSaved at:", output_file)
