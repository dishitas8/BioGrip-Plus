
import pandas as pd
import matplotlib.pyplot as plt
import os

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

input_file = os.path.join(
    project_path,
    "dataset",
    "sensor_anomaly_results.csv"
)

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sensor measurements to visualize
sensors = [
    ("temperature_c", "Temperature (°C)"),
    ("humidity_percent", "Humidity (%)"),
    ("pressure_kpa", "Pressure (kPa)"),
    ("deformation_mm", "Deformation (mm)")
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, (column, title) in zip(axes.flatten(), sensors):

    normal = df[df["ai_status"] == "Normal"]
    anomalies = df[df["ai_status"] == "Anomaly"]

    ax.plot(
        normal["timestamp"],
        normal[column],
        label="Normal readings",
        color="green",
        marker=".",
        markersize=4
    )

    ax.scatter(
        anomalies["timestamp"],
        anomalies[column],
        label="AI-detected anomalies",
        color="red",
        marker="x",
        s=45
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "BioGrip+ AI Sensor Anomaly Monitoring",
    fontsize=16
)

plt.tight_layout()

output_folder = os.path.join(
    project_path,
    "documentation",
    "visualizations"
)

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "sensor_anomaly_monitoring.png"
)

plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("Anomaly visualization generated!")
print("Saved at:", output_file)

plt.show()
