
import numpy as np
import pandas as pd
import os

# Generate reproducible simulated data
np.random.seed(42)

number_of_samples = 500

crop_types = ["Tomato", "Strawberry", "Grapes", "Mushroom"]

data = []

for i in range(number_of_samples):

    crop = np.random.choice(crop_types)

    size_cm = np.random.uniform(2, 8)
    maturity = np.random.uniform(0.4, 1.0)

    temperature = np.random.uniform(20, 35)
    humidity = np.random.uniform(40, 90)
    deformation = np.random.uniform(1, 10)
    pressure = np.random.uniform(5, 30)

    crop_factor = {
        "Tomato": 1.0,
        "Strawberry": 0.55,
        "Grapes": 0.45,
        "Mushroom": 0.65
    }

    target_grip_force = (
        crop_factor[crop] * size_cm * 0.35
        + maturity * 0.4
        + np.random.normal(0, 0.08)
    )

    data.append([
        crop, size_cm, maturity, temperature,
        humidity, pressure, deformation,
        target_grip_force
    ])

columns = [
    "crop_type",
    "size_cm",
    "maturity",
    "temperature_c",
    "humidity_percent",
    "pressure_kpa",
    "deformation_mm",
    "target_grip_force_n"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset in the project dataset folder
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_path, "dataset")

os.makedirs(dataset_path, exist_ok=True)

file_path = os.path.join(dataset_path, "simulated_sensor_data.csv")

df.to_csv(file_path, index=False)

print("Dataset generated successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved at:", file_path)
