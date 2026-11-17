
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Locate the project folder
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load simulated dataset
data_path = os.path.join(
    project_path, "dataset", "simulated_sensor_data.csv"
)

df = pd.read_csv(data_path)

# Select input features
X = df[["crop_type", "size_cm", "maturity"]]

# Select target variable
y = df["target_grip_force_n"]

# Convert crop names into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("crop", OneHotEncoder(handle_unknown="ignore"), ["crop_type"])
    ],
    remainder="passthrough"
)

# Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training Random Forest model...")
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
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

# Save trained model
model_path = os.path.join(
    project_path, "ai_models", "grip_force_model.joblib"
)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)

# Test with a sample crop
sample = pd.DataFrame({
    "crop_type": ["Tomato"],
    "size_cm": [5.7],
    "maturity": [0.8]
})

predicted_force = model.predict(sample)

print("\nSAMPLE PREDICTION")
print("Crop: Tomato")
print("Size: 5.7 cm")
print("Maturity: 0.8")
print("Predicted target grip force:",
      round(predicted_force[0], 3), "N")

print("\nTraining completed!")
