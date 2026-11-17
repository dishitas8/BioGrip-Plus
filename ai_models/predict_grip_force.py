
import pandas as pd
import joblib
import os

# Locate saved model
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    project_path,
    "ai_models",
    "grip_force_model.joblib"
)

# Load trained model
model = joblib.load(model_path)

print("===================================")
print("       BioGrip+ AI Predictor")
print("===================================")

crop_types = ["Tomato", "Strawberry", "Grapes", "Mushroom"]

print("\nAvailable crops:")
for crop in crop_types:
    print("-", crop)

crop = input("\nEnter crop type: ").strip().title()

if crop not in crop_types:
    print("Invalid crop type!")
    raise SystemExit

try:
    size = float(input("Enter crop size (2 to 8 cm): "))
    maturity = float(input("Enter maturity (0.4 to 1.0): "))

    if not 2 <= size <= 8:
        raise ValueError("Crop size must be between 2 and 8 cm.")

    if not 0.4 <= maturity <= 1.0:
        raise ValueError("Maturity must be between 0.4 and 1.0.")

except ValueError as error:
    print("Invalid input:", error)
    raise SystemExit

# Prepare input for model
sample = pd.DataFrame({
    "crop_type": [crop],
    "size_cm": [size],
    "maturity": [maturity]
})

# Predict synthetic target
prediction = model.predict(sample)[0]

print("\n========== PREDICTION ==========")
print("Crop:", crop)
print("Size:", size, "cm")
print("Maturity:", maturity)
print("Predicted target grip force:", round(prediction, 3), "N")
print("================================")

print("\nNote: This is a simulation-based prediction,")
print("not a validated safe force for physical harvesting.")
