
import os
import joblib
import pandas as pd

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

model_path = os.path.join(
    project_path,
    "ai_models",
    "grip_force_model_v2.joblib"
)

model = joblib.load(model_path)

print("===================================")
print("       BioGrip+ V2 AI Predictor")
print("===================================")

crop = input(
    "Enter crop type (Tomato/Strawberry/Grapes/Mushroom): "
).strip().title()

if crop not in ["Tomato", "Strawberry", "Grapes", "Mushroom"]:
    print("Invalid crop type.")
    raise SystemExit

size = float(input("Enter crop size (2 to 8 cm): "))
maturity = float(input("Enter maturity (0.4 to 1.0): "))
temperature = float(input("Enter temperature (20 to 35 C): "))
humidity = float(input("Enter humidity (40 to 90%): "))
pressure = float(input("Enter pressure (kPa): "))
deformation = float(input("Enter deformation (mm): "))

if not (2 <= size <= 8 and 0.4 <= maturity <= 1.0
        and 20 <= temperature <= 35
        and 40 <= humidity <= 90
        and pressure >= 0 and deformation >= 0):
    print("One or more input values are outside the allowed range.")
    raise SystemExit

input_data = pd.DataFrame([{
    "crop_type": crop,
    "size_cm": size,
    "maturity": maturity,
    "temperature_c": temperature,
    "humidity_percent": humidity,
    "pressure_kpa": pressure,
    "deformation_mm": deformation
}])

prediction = model.predict(input_data)[0]

print("\n========== V2 PREDICTION ==========")
print("Crop:", crop)
print("Predicted target grip force:", round(prediction, 3), "N")
print("==================================")

print("\nNote: This is a simulation-based prediction.")
print("It is not a validated safe force for physical harvesting.")
