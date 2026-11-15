
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

FORECAST_FILE = DATA_DIR / "forecast_clean.csv"
HISTORICAL_FILE = DATA_DIR / "historical_clean.csv"

ML_RESULTS_DIR = DATA_DIR / "ml_results"

PREDICTIONS_FILE = ML_RESULTS_DIR / "test_predictions.csv"
EVALUATION_FILE = ML_RESULTS_DIR / "model_evaluation.csv"
ANOMALIES_FILE = ML_RESULTS_DIR / "weather_anomalies.csv"
METADATA_FILE = ML_RESULTS_DIR / "ml_run_metadata.json"

SUPPORTED_CITIES = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Jaipur",
    "Kolkata",
    "Mumbai",
    "Pune",
    "Srinagar",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "weather_code",
]

MODEL_FILES = {
    "temperature_2m": MODEL_DIR / "temperature_2m_model.joblib",
    "relative_humidity_2m": MODEL_DIR / "relative_humidity_2m_model.joblib",
    "precipitation": MODEL_DIR / "precipitation_model.joblib",
    "wind_speed_10m": MODEL_DIR / "wind_speed_10m_model.joblib",
    "anomaly_detection": MODEL_DIR / "isolation_forest.joblib",
}


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="National Weather Analytics Platform API",
    description=(
        "Backend API for multi-city weather data, analytics, "
        "anomaly records and machine-learning evaluation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 3. COMMON HELPERS
# =========================================================

def read_csv_file(
    file_path: Path,
    required: bool = True,
) -> Optional[pd.DataFrame]:

    if not file_path.exists():

        if required:
            raise HTTPException(
                status_code=404,
                detail=f"Required data file not found: {file_path.name}",
            )

        return None

    try:
        return pd.read_csv(file_path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {file_path.name}: {str(exc)}",
        )


def dataframe_records(
    df: pd.DataFrame,
) -> list[dict]:

    clean_df = df.copy()

    clean_df = clean_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    clean_df = clean_df.astype(object).where(
        pd.notna(clean_df),
        None,
    )

    return clean_df.to_dict(orient="records")


def normalize_city(city: str) -> str:

    requested_city = city.strip().lower()

    for supported_city in SUPPORTED_CITIES:

        if supported_city.lower() == requested_city:
            return supported_city

    raise HTTPException(
        status_code=404,
        detail=(
            f"City '{city}' is not supported. "
            f"Available cities: {SUPPORTED_CITIES}"
        ),
    )


def get_weather_data() -> pd.DataFrame:

    df = read_csv_file(FORECAST_FILE)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Forecast dataset is unavailable.",
        )

    required = {"time", "city"}

    missing = required - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast dataset missing columns: {sorted(missing)}",
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["time", "city"]
    ).copy()

    df["city"] = df["city"].astype(str)

    return df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)


def get_model_status() -> list[dict]:

    status = []

    for model_name, model_path in MODEL_FILES.items():

        status.append({
            "model": model_name,
            "available": model_path.exists(),
            "file": model_path.name,
        })

    return status


def get_city_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    available_columns = [
        column
        for column in WEATHER_COLUMNS
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    aggregations = {}

    if "temperature_2m" in available_columns:
        aggregations["temperature_2m"] = ["mean", "min", "max"]

    if "relative_humidity_2m" in available_columns:
        aggregations["relative_humidity_2m"] = ["mean"]

    if "precipitation" in available_columns:
        aggregations["precipitation"] = ["sum", "max"]

    if "wind_speed_10m" in available_columns:
        aggregations["wind_speed_10m"] = ["mean", "max"]

    if "cloud_cover" in available_columns:
        aggregations["cloud_cover"] = ["mean"]

    if not aggregations:
        return pd.DataFrame()

    summary = df.groupby("city").agg(aggregations)

    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()

    if "precipitation_sum" in summary.columns:
        rainy_hours = (
            df.assign(
                wet_hour=df["precipitation"] > 0.1
            )
            .groupby("city")["wet_hour"]
            .sum()
            .rename("rainy_hours")
            .reset_index()
        )

        summary = summary.merge(
            rainy_hours,
            on="city",
            how="left",
        )

    return summary


# =========================================================
# 4. ROOT AND HEALTH ENDPOINTS
# =========================================================

@app.get("/")
def root():

    return {
        "project": "National Weather Analytics Platform",
        "message": "Weather analytics API is running.",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_endpoint": "/health",
    }


@app.get("/health")
def health_check():

    required_files = {
        "forecast_data": FORECAST_FILE,
        "historical_data": HISTORICAL_FILE,
        "test_predictions": PREDICTIONS_FILE,
        "model_evaluation": EVALUATION_FILE,
        "anomalies": ANOMALIES_FILE,
    }

    file_status = {
        name: path.exists()
        for name, path in required_files.items()
    }

    models = get_model_status()

    return {
        "status": "running",
        "data_files": file_status,
        "models": models,
        "all_models_available": all(
            item["available"] for item in models
        ),
    }


# =========================================================
# 5. CITY ENDPOINTS
# =========================================================

@app.get("/weather/cities")
def get_cities():

    df = get_weather_data()

    cities = sorted(
        df["city"].dropna().unique().tolist()
    )

    return {
        "total_cities": len(cities),
        "cities": cities,
    }


# =========================================================
# 6. LATEST AVAILABLE FORECAST SNAPSHOT
# =========================================================

@app.get("/weather/latest")
def get_latest_weather(
    city: Optional[str] = Query(
        default=None,
        description="Optional city filter.",
    ),
):

    df = get_weather_data()

    if city:
        selected_city = normalize_city(city)

        df = df[
            df["city"].str.lower()
            == selected_city.lower()
        ]

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast records found for {selected_city}.",
            )

    # The earliest available forecast timestamp is the closest
    # forecast snapshot to the beginning of the forecast file.
    first_times = (
        df.groupby("city")["time"]
        .min()
        .rename("snapshot_time")
        .reset_index()
    )

    latest = df.merge(
        first_times,
        on="city",
        how="inner",
    )

    latest = latest[
        latest["time"] == latest["snapshot_time"]
    ].copy()

    return {
        "description": (
            "First available timestamp in the stored forecast dataset; "
            "not a live sensor observation."
        ),
        "total_cities": int(latest["city"].nunique()),
        "data": dataframe_records(latest),
    }


# =========================================================
# 7. WEATHER FORECAST ENDPOINT
# =========================================================

@app.get("/weather/forecast/{city}")
def get_city_forecast(
    city: str,
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
        description="Number of forecast hours to return.",
    ),
):

    selected_city = normalize_city(city)

    df = get_weather_data()

    city_df = df[
        df["city"].str.lower()
        == selected_city.lower()
    ].copy()

    if city_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast data found for {selected_city}.",
        )

    city_df = city_df.sort_values(
        "time"
    ).head(hours)

    return {
        "city": selected_city,
        "forecast_source": "Stored Open-Meteo forecast dataset",
        "forecast_type": "Provider weather forecast, not ML prediction",
        "requested_hours": hours,
        "returned_hours": len(city_df),
        "data": dataframe_records(city_df),
    }


# =========================================================
# 8. WEATHER ANALYTICS ENDPOINT
# =========================================================

@app.get("/weather/analytics")
def get_weather_analytics(
    city: Optional[str] = Query(
        default=None,
        description="Optional city filter.",
    ),
):

    df = get_weather_data()

    if city:
        selected_city = normalize_city(city)

        df = df[
            df["city"].str.lower()
            == selected_city.lower()
        ]

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No records found for {selected_city}.",
            )

    summary = get_city_summary(df)

    if summary.empty:
        raise HTTPException(
            status_code=500,
            detail="Weather variables required for analytics are missing.",
        )

    return {
        "description": (
            "Statistics calculated over the stored forecast period, "
            "not long-term climate averages."
        ),
        "records_analysed": int(len(df)),
        "cities_analysed": int(df["city"].nunique()),
        "data": dataframe_records(summary),
    }


# =========================================================
# 9. ANOMALY ENDPOINT
# =========================================================

@app.get("/weather/anomalies")
def get_anomalies(
    city: Optional[str] = Query(default=None),
    limit: int = Query(
        default=100,
        ge=1,
        le=5000,
    ),
):

    df = read_csv_file(ANOMALIES_FILE)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Anomaly dataset is unavailable.",
        )

    if "anomaly" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Anomaly dataset has no anomaly column.",
        )

    df["anomaly"] = df["anomaly"].astype(bool)

    anomalies = df[
        df["anomaly"]
    ].copy()

    if city:
        selected_city = normalize_city(city)

        anomalies = anomalies[
            anomalies["city"].str.lower()
            == selected_city.lower()
        ]

    anomalies = anomalies.sort_values(
        "time",
        ascending=False,
    ).head(limit)

    return {
        "description": (
            "Isolation Forest statistical outliers. "
            "These are not automatically severe weather events."
        ),
        "total_returned": len(anomalies),
        "data": dataframe_records(anomalies),
    }


# =========================================================
# 10. MODEL EVALUATION ENDPOINT
# =========================================================

@app.get("/ml/evaluation")
def get_model_evaluation():

    df = read_csv_file(EVALUATION_FILE)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Model evaluation report is unavailable.",
        )

    return {
        "description": (
            "Held-out historical test evaluation. "
            "MAE, RMSE and R-squared are regression metrics; "
            "R-squared is not percentage accuracy."
        ),
        "models_evaluated": len(df),
        "data": dataframe_records(df),
    }


# =========================================================
# 11. HISTORICAL TEST PREDICTIONS ENDPOINT
# =========================================================

@app.get("/ml/predictions/{city}")
def get_ml_predictions(
    city: str,
    target: Optional[str] = Query(
        default=None,
        description="Optional target variable filter.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=5000,
    ),
):

    selected_city = normalize_city(city)

    df = read_csv_file(PREDICTIONS_FILE)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Historical test predictions are unavailable.",
        )

    required = {
        "city",
        "target_name",
        "actual_next_hour",
        "predicted_next_hour",
    }

    missing = required - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction file missing columns: {sorted(missing)}",
        )

    predictions = df[
        df["city"].str.lower()
        == selected_city.lower()
    ].copy()

    if target:
        valid_targets = TARGETS

        if target not in valid_targets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target. Choose from: {valid_targets}",
            )

        predictions = predictions[
            predictions["target_name"] == target
        ]

    predictions = predictions.sort_values(
        "target_time",
        ascending=False,
    ).head(limit)

    return {
        "city": selected_city,
        "prediction_type": "Historical held-out test predictions",
        "is_live_forecast": False,
        "returned_records": len(predictions),
        "data": dataframe_records(predictions),
    }


# =========================================================
# 12. MODEL STATUS ENDPOINT
# =========================================================

@app.get("/ml/models")
def get_models():

    models = get_model_status()

    return {
        "total_models": len(models),
        "available_models": sum(
            item["available"] for item in models
        ),
        "models": models,
    }


# =========================================================
# 13. STARTUP INFORMATION
# =========================================================

@app.on_event("startup")
def startup_event():

    logger_message = (
        "National Weather Analytics Platform API started."
    )

    print("=" * 65)
    print(logger_message)
    print(f"Project directory: {ROOT}")
    print(f"Forecast dataset exists: {FORECAST_FILE.exists()}")
    print(f"Historical dataset exists: {HISTORICAL_FILE.exists()}")
    print("=" * 65)
