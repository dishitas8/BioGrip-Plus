
import numpy as np
import pandas as pd
import os

np.random.seed(42)

number_of_samples = 1000

crop_types = ["Tomato", "Strawberry", "Grapes", "Mushroom"]

crop_factors = {
    "Tomato": 1.0,
    "Strawberry": 0.55,
    "Grapes": 0.45,
    "Mushroom": 0.65
}

data = []

for i in range(number_of_samples):

    crop = np.random.choice(crop_types)

    size = np.random.uniform(2, 8)
    maturity = np.random.uniform(0.4, 1.0)

    temperature = np.random.uniform(20, 35)
    humidity = np.random.uniform(40, 90)

    # Illustrative synthetic target force
    grip_force = (
        crop_factors[crop] * size * 0.35
        + maturity * 0.4
        + np.random.normal(0, 0.08)
    )

    # Keep simulated target values positive
    grip_force = max(grip_force, 0.1)

    # Illustrative contact area in square millimetres
    contact_area = size * 20

    # Pressure is related to force and contact area
    pressure = (
        grip_force / contact_area
    ) * 1000 + np.random.normal(0, 0.5)

    # Illustrative material-response relationship
    temperature_effect = 0.006 * (temperature - 25)
    humidity_effect = 0.004 * (humidity - 60)

    compliance = (
        0.8
        + temperature_effect
        + humidity_effect
    )

    deformation = (
        grip_force * compliance
        + np.random.normal(0, 0.08)
    )

    deformation = max(deformation, 0.01)

    data.append([
        crop,
        size,
        maturity,
        temperature,
        humidity,
        contact_area,
        pressure,
        deformation,
        grip_force
    ])

columns = [
    "crop_type",
    "size_cm",
    "maturity",
    "temperature_c",
    "humidity_percent",
    "contact_area_mm2",
    "pressure_kpa",
    "deformation_mm",
    "target_grip_force_n"
]

df = pd.DataFrame(data, columns=columns)

# Save Version 2 without overwriting Version 1
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(project_path, "dataset")
os.makedirs(dataset_path, exist_ok=True)

file_path = os.path.join(
    dataset_path,
    "simulated_sensor_data_v2.csv"
)

df.to_csv(file_path, index=False)

print("BioGrip+ Version 2 dataset generated!")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSensor correlation with target:")
print(
    df[[
        "pressure_kpa",
        "deformation_mm",
        "target_grip_force_n"
    ]].corr()
)

print("\nSaved at:", file_path)
