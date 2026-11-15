
"""
National Weather Analytics Platform
Module 1: Data Ingestion, Cleaning and Storage

Data source:
Open-Meteo Forecast API and Historical Weather API
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"

for directory in (RAW_DIR, PROCESSED_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

TIMEZONE = "Asia/Kolkata"
TIMEOUT = 30
MAX_RETRIES = 3

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "weather_code",
]

CITIES = {
    "Srinagar": {"latitude": 34.0837, "longitude": 74.7973},
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
    "Kolkata": {"latitude": 22.5726, "longitude": 88.3639},
    "Ahmedabad": {"latitude": 23.0225, "longitude": 72.5714},
    "Jaipur": {"latitude": 26.9124, "longitude": 75.7873},
    "Pune": {"latitude": 18.5204, "longitude": 73.8567},
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# API CLIENT
# ============================================================

class WeatherAPIClient:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NationalWeatherAnalyticsPlatform/1.0"
        })

    def get(self, url, params):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=TIMEOUT
                )

                response.raise_for_status()
                result = response.json()

                if result.get("error"):
                    raise ValueError(result.get("reason"))

                return result

            except (requests.RequestException, ValueError) as error:
                logger.warning(
                    "API attempt %s/%s failed: %s",
                    attempt,
                    MAX_RETRIES,
                    error,
                )

                if attempt == MAX_RETRIES:
                    raise

                time.sleep(2 ** attempt)

    def close(self):
        self.session.close()


# ============================================================
# DATA EXTRACTION
# ============================================================

def fetch_weather(
    client,
    city,
    coordinates,
    historical=False,
    start_date=None,
    end_date=None,
):
    """
    Fetch hourly weather data for one city.
    """

    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    if historical:
        if not start_date or not end_date:
            raise ValueError("Historical dates are required.")

        if end_date > date.today() - timedelta(days=6):
            raise ValueError(
                "Historical archive dates must end at least "
                "six days before today. Use the forecast API "
                "for recent data."
            )

        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

        url = HISTORICAL_URL

    else:
        params["forecast_days"] = 7
        params["past_days"] = 0

        url = FORECAST_URL

    result = client.get(url, params)

    hourly = result.get("hourly")

    if not hourly or "time" not in hourly:
        raise ValueError(f"No hourly records returned for {city}")

    dataframe = pd.DataFrame(hourly)

    dataframe["city"] = city
    dataframe["latitude"] = latitude
    dataframe["longitude"] = longitude

    dataframe["timezone"] = result.get("timezone", TIMEZONE)

    logger.info(
        "%s: received %s hourly records",
        city,
        len(dataframe),
    )

    return dataframe


# ============================================================
# COLLECT ALL CITIES
# ============================================================

def collect_all_cities(
    historical=False,
    start_date=None,
    end_date=None,
):
    client = WeatherAPIClient()
    frames = []

    try:
        for city, coordinates in CITIES.items():

            logger.info("Collecting weather data for %s", city)

            try:
                dataframe = fetch_weather(
                    client=client,
                    city=city,
                    coordinates=coordinates,
                    historical=historical,
                    start_date=start_date,
                    end_date=end_date,
                )

                frames.append(dataframe)

            except Exception:
                logger.exception(
                    "Failed to collect data for %s",
                    city,
                )

    finally:
        client.close()

    if not frames:
        raise RuntimeError("No city data was collected.")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(dataframe):

    dataframe = dataframe.copy()

    required_columns = [
        "time",
        "city",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dataframe["time"] = pd.to_datetime(
        dataframe["time"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=["time", "city"]
    )

    numeric_columns = [
        column
        for column in HOURLY_VARIABLES
        if column in dataframe.columns
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

    dataframe = dataframe.drop_duplicates(
        subset=["city", "time"],
        keep="last"
    )

    dataframe = dataframe.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    logger.info("Data validation completed.")
    logger.info("Validated rows: %s", len(dataframe))

    return dataframe


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(dataframe):

    dataframe = dataframe.copy()

    dataframe["hour"] = dataframe["time"].dt.hour
    dataframe["day"] = dataframe["time"].dt.day
    dataframe["month"] = dataframe["time"].dt.month
    dataframe["year"] = dataframe["time"].dt.year

    dataframe["day_of_week"] = (
        dataframe["time"].dt.day_of_week
    )

    dataframe["season"] = (
        dataframe["month"].map({
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Summer",
            4: "Summer",
            5: "Summer",
            6: "Monsoon",
            7: "Monsoon",
            8: "Monsoon",
            9: "Monsoon",
            10: "Post-Monsoon",
            11: "Post-Monsoon",
        })
    )

    dataframe["is_weekend"] = (
        dataframe["day_of_week"] >= 5
    ).astype(int)

    if "temperature_2m" in dataframe.columns:
        dataframe["temperature_category"] = pd.cut(
            dataframe["temperature_2m"],
            bins=[-float("inf"), 10, 20, 30, 40, float("inf")],
            labels=[
                "Cold",
                "Cool",
                "Moderate",
                "Hot",
                "Very Hot",
            ],
        )

    if "precipitation" in dataframe.columns:
        dataframe["rain_indicator"] = (
            dataframe["precipitation"] > 0
        ).astype(int)

    logger.info("Feature engineering completed.")

    return dataframe


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def generate_quality_report(dataframe):

    report = {
        "total_records": len(dataframe),
        "total_cities": dataframe["city"].nunique(),
        "missing_values": dataframe.isnull().sum().to_dict(),
        "duplicate_records": int(
            dataframe.duplicated(
                subset=["city", "time"]
            ).sum()
        ),
        "date_start": str(dataframe["time"].min()),
        "date_end": str(dataframe["time"].max()),
    }

    quality_path = PROCESSED_DIR / "data_quality_report.json"

    import json

    with open(quality_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    logger.info("Quality report saved: %s", quality_path)

    return report


# ============================================================
# STORAGE
# ============================================================

def save_data(dataframe, filename, directory):

    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info("Dataset saved: %s", output_path)

    return output_path


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    mode="forecast",
    start_date=None,
    end_date=None,
):
    """
    mode:
        forecast  -> Seven-day hourly forecast.
        historical -> Historical hourly weather.
    """

    logger.info("=" * 60)
    logger.info("NATIONAL WEATHER DATA PIPELINE STARTED")
    logger.info("Mode: %s", mode)
    logger.info("=" * 60)

    historical = mode == "historical"

    if mode not in ("forecast", "historical"):
        raise ValueError("Mode must be forecast or historical.")

    raw_data = collect_all_cities(
        historical=historical,
        start_date=start_date,
        end_date=end_date,
    )

    prefix = "historical" if historical else "forecast"

    save_data(
        raw_data,
        f"{prefix}_raw.csv",
        RAW_DIR,
    )

    clean_data = validate_data(raw_data)

    clean_data = engineer_features(clean_data)

    save_data(
        clean_data,
        f"{prefix}_clean.csv",
        PROCESSED_DIR,
    )

    report = generate_quality_report(clean_data)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Total records: %s", report["total_records"])
    logger.info("Cities: %s", report["total_cities"])
    logger.info("=" * 60)

    return clean_data


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        dataset = run_pipeline(mode="forecast")

        print("\n========== PIPELINE SUMMARY ==========")
        print("Total records:", len(dataset))
        print("Cities:", dataset["city"].nunique())
        print("\nRecords per city:")
        print(dataset.groupby("city").size())
        print("\nSample data:")
        print(dataset.head(10).to_string(index=False))

    except Exception:
        logger.exception("Pipeline execution failed.")
        raise
