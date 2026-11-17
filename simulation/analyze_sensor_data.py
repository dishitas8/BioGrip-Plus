
import pandas as pd
import matplotlib.pyplot as plt
import os

# Find and load the dataset
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(
    project_path, "dataset", "simulated_sensor_data.csv"
)

df = pd.read_csv(file_path)

# 1. Basic dataset information
print("DATASET INFORMATION")
print(df.info())

# 2. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 3. Number of records for each crop
print("\nCROP DISTRIBUTION")
print(df["crop_type"].value_counts())

# 4. Average grip force for each crop
print("\nAVERAGE GRIP FORCE")
print(df.groupby("crop_type")["target_grip_force_n"].mean())

# Create folder for charts
output_folder = os.path.join(
    project_path, "documentation", "visualizations"
)
os.makedirs(output_folder, exist_ok=True)

# Chart 1: Average grip force by crop
average_force = df.groupby("crop_type")[
    "target_grip_force_n"
].mean()

average_force.plot(kind="bar", title="Average Grip Force by Crop")
plt.xlabel("Crop Type")
plt.ylabel("Simulated Grip Force (N)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "average_grip_force.png"))
plt.show()

# Chart 2: Crop size vs grip force
plt.figure(figsize=(8, 5))

for crop in df["crop_type"].unique():
    crop_data = df[df["crop_type"] == crop]
    plt.scatter(
        crop_data["size_cm"],
        crop_data["target_grip_force_n"],
        label=crop,
        alpha=0.6
    )

plt.title("Crop Size vs Simulated Grip Force")
plt.xlabel("Crop Size (cm)")
plt.ylabel("Simulated Grip Force (N)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "size_vs_grip_force.png"))
plt.show()

# Chart 3: Grip force distribution
plt.figure(figsize=(8, 5))
plt.hist(df["target_grip_force_n"], bins=20, edgecolor="black")
plt.title("Distribution of Simulated Grip Force")
plt.xlabel("Simulated Grip Force (N)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "grip_force_distribution.png"))
plt.show()

print("\nAnalysis completed successfully!")
print("Charts saved in:", output_folder)
