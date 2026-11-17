
import numpy as np
import pandas as pd
import os

np.random.seed(42)

number_of_readings = 120

timestamps = pd.date_range(
    start="2026-11-10 10:00:00",
    periods=number_of_readings,
    freq="min"
)

temperature = (
    27
    + 0.8 * np.sin(np.linspace(0, 4 * np.pi, number_of_readings))
    + np.random.normal(0, 0.15, number_of_readings)
)

humidity = (
    60
    + 3 * np.sin(np.linspace(0, 3 * np.pi, number_of_readings))
    + np.random.normal(0, 0.5, number_of_readings)
)

pressure = (
    18
    + 0.5 * np.sin(np.linspace(0, 5 * np.pi, number_of_readings))
    + np.random.normal(0, 0.2, number_of_readings)
)

deformation = (
    1.5
    + (pressure - 18) * 0.04
    + np.random.normal(0, 0.025, number_of_readings)
)

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_c": temperature,
    "humidity_percent": humidity,
    "pressure_kpa": pressure,
    "deformation_mm": deformation,
    "sensor_status": "Normal"
})

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_path = os.path.join(project_path, "dataset")
os.makedirs(dataset_path, exist_ok=True)

file_path = os.path.join(
    dataset_path,
    "sensor_monitoring_data.csv"
)

df.to_csv(file_path, index=False)

print("BioGrip+ sensor monitoring dataset generated!")
print("\nFirst 5 readings:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nSensor summary:")
print(df.describe())

print("\nSaved at:", file_path)
