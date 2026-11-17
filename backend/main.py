
import os
import sqlite3
import joblib
import pandas as pd

from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="BioGrip+ AI API",
    description="AI-powered soft robotic gripper and sensor monitoring",
    version="3.0"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

model_path = os.path.join(
    project_path,
    "ai_models",
    "grip_force_model_v2.joblib"
)

database_folder = os.path.join(
    project_path,
    "database"
)

os.makedirs(database_folder, exist_ok=True)

database_path = os.path.join(
    database_folder,
    "biogrip_history.db"
)

sensor_file = os.path.join(
    project_path,
    "dataset",
    "sensor_anomaly_results.csv"
)


# ==========================================
# 3. LOAD MACHINE LEARNING MODEL
# ==========================================

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )

model = joblib.load(model_path)


# ==========================================
# 4. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 5. CREATE SQLITE DATABASE
# ==========================================

def create_database():

    with sqlite3.connect(database_path) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_type TEXT NOT NULL,
                size_cm REAL,
                maturity REAL,
                temperature_c REAL,
                humidity_percent REAL,
                pressure_kpa REAL,
                deformation_mm REAL,
                predicted_grip_force_n REAL,
                created_at TEXT
            )
        """)


create_database()


# ==========================================
# 6. INPUT VALIDATION
# ==========================================

class GripInput(BaseModel):

    crop_type: str

    size_cm: float = Field(ge=2, le=8)

    maturity: float = Field(ge=0.4, le=1.0)

    temperature_c: float = Field(ge=20, le=35)

    humidity_percent: float = Field(ge=40, le=90)

    pressure_kpa: float = Field(ge=0)

    deformation_mm: float = Field(ge=0)


# ==========================================
# 7. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "BioGrip+",
        "message": "Welcome to BioGrip+ AI API",
        "status": "Running",
        "version": "3.0"
    }


# ==========================================
# 8. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_exists": os.path.exists(database_path),
        "sensor_dataset_exists": os.path.exists(sensor_file)
    }


# ==========================================
# 9. PREDICT GRIP FORCE
# ==========================================

@app.post("/predict")
def predict_grip_force(data: GripInput):

    crop = data.crop_type.strip().title()

    supported_crops = [
        "Tomato",
        "Strawberry",
        "Grapes",
        "Mushroom"
    ]

    if crop not in supported_crops:
        raise HTTPException(
            status_code=400,
            detail="Unsupported crop type"
        )

    input_data = pd.DataFrame([{
        "crop_type": crop,
        "size_cm": data.size_cm,
        "maturity": data.maturity,
        "temperature_c": data.temperature_c,
        "humidity_percent": data.humidity_percent,
        "pressure_kpa": data.pressure_kpa,
        "deformation_mm": data.deformation_mm
    }])

    prediction = float(model.predict(input_data)[0])

    prediction = round(prediction, 3)

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(database_path) as conn:

        cursor = conn.execute("""
            INSERT INTO predictions (
                crop_type,
                size_cm,
                maturity,
                temperature_c,
                humidity_percent,
                pressure_kpa,
                deformation_mm,
                predicted_grip_force_n,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            crop,
            data.size_cm,
            data.maturity,
            data.temperature_c,
            data.humidity_percent,
            data.pressure_kpa,
            data.deformation_mm,
            prediction,
            timestamp
        ))

        prediction_id = cursor.lastrowid

    return {
        "project": "BioGrip+",
        "prediction_id": prediction_id,
        "crop": crop,
        "predicted_grip_force_n": prediction,
        "model": "Random Forest V2",
        "note": (
            "Simulation-based prediction; "
            "not a validated safe harvesting force."
        )
    }


# ==========================================
# 10. GET PREDICTION HISTORY
# ==========================================

@app.get("/history")
def get_prediction_history():

    with sqlite3.connect(database_path) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

    return {
        "total_returned": len(rows),
        "history": [dict(row) for row in rows]
    }


# ==========================================
# 11. DASHBOARD STATISTICS
# ==========================================

@app.get("/stats")
def get_dashboard_stats():

    with sqlite3.connect(database_path) as conn:

        conn.row_factory = sqlite3.Row

        total = conn.execute("""
            SELECT COUNT(*) AS total
            FROM predictions
        """).fetchone()["total"]

        average = conn.execute("""
            SELECT AVG(predicted_grip_force_n) AS average
            FROM predictions
        """).fetchone()["average"]

        latest = conn.execute("""
            SELECT crop_type
            FROM predictions
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        crop_stats = conn.execute("""
            SELECT
                crop_type,
                COUNT(*) AS prediction_count,
                ROUND(AVG(predicted_grip_force_n), 3)
                    AS average_force
            FROM predictions
            GROUP BY crop_type
        """).fetchall()

    return {
        "total_predictions": total,

        "average_predicted_force": (
            round(average, 3) if average is not None else 0
        ),

        "most_recent_crop": (
            latest["crop_type"] if latest else "No data"
        ),

        "crop_wise_analytics": [
            dict(row) for row in crop_stats
        ]
    }


# ==========================================
# 12. AI SENSOR HEALTH MONITORING
# ==========================================

@app.get("/sensor-health")
def get_sensor_health():

    if not os.path.exists(sensor_file):

        raise HTTPException(
            status_code=404,
            detail="Sensor anomaly dataset not found. "
                   "Generate the sensor data and run anomaly detection."
        )

    df = pd.read_csv(sensor_file)

    required_columns = [
        "timestamp",
        "temperature_c",
        "humidity_percent",
        "pressure_kpa",
        "deformation_mm",
        "ai_status"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise HTTPException(
            status_code=500,
            detail=f"Missing columns in sensor dataset: {missing_columns}"
        )

    total = len(df)

    normal_count = int(
        (df["ai_status"] == "Normal").sum()
    )

    anomaly_count = int(
        (df["ai_status"] == "Anomaly").sum()
    )

    anomaly_percentage = (
        round((anomaly_count / total) * 100, 2)
        if total > 0 else 0
    )

    # Calculate average sensor values

    sensor_averages = {
        "temperature_c": round(
            float(df["temperature_c"].mean()), 3
        ),

        "humidity_percent": round(
            float(df["humidity_percent"].mean()), 3
        ),

        "pressure_kpa": round(
            float(df["pressure_kpa"].mean()), 3
        ),

        "deformation_mm": round(
            float(df["deformation_mm"].mean()), 3
        )
    }

    # Latest 20 sensor readings

    latest_readings = (
        df.tail(20)
        .fillna("")
        .to_dict(orient="records")
    )

    # Latest 10 detected anomalies

    anomalies = (
        df[df["ai_status"] == "Anomaly"]
        .tail(10)
        .fillna("")
        .to_dict(orient="records")
    )

    return {
        "project": "BioGrip+",

        "status": "available",

        "data_source": "Simulated sensor dataset",

        "total_readings": total,

        "normal_readings": normal_count,

        "anomaly_readings": anomaly_count,

        "anomaly_percentage": anomaly_percentage,

        "sensor_averages": sensor_averages,

        "latest_readings": latest_readings,

        "detected_anomalies": anomalies,

        "note": (
            "AI anomaly detection results are based "
            "on simulated sensor readings."
        )
    }
