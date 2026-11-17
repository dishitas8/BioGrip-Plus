
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Version 2 dataset
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(
    project_path,
    "dataset",
    "simulated_sensor_data_v2.csv"
)

df = pd.read_csv(dataset_path)

print("Training BioGrip+ V2 AI Model...")

# Input features
X = df[
    [
        "crop_type",
        "size_cm",
        "maturity",
        "temperature_c",
        "humidity_percent",
        "pressure_kpa",
        "deformation_mm"
    ]
]

# Target
y = df["target_grip_force_n"]

# Encode crop type
preprocessor = ColumnTransformer(
    transformers=[
        ("crop", OneHotEncoder(handle_unknown="ignore"),
         ["crop_type"])
    ],
    remainder="passthrough"
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# Save new model separately
model_path = os.path.join(
    project_path,
    "ai_models",
    "grip_force_model_v2.joblib"
)

joblib.dump(model, model_path)

print("\nV2 Model saved successfully!")
print("Location:", model_path)
