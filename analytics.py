
"""
National Weather Analytics Platform
Module 2: Exploratory Data Analysis and Weather Analytics
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "forecast_clean.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "analytics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Run data_pipeline.py first."
        )

    dataframe = pd.read_csv(INPUT_FILE)

    dataframe["time"] = pd.to_datetime(
        dataframe["time"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=["time", "city"]
    )

    if dataframe.empty:
        raise ValueError("Weather dataset is empty.")

    logger.info("Dataset loaded successfully.")
    logger.info("Records: %s", len(dataframe))
    logger.info("Cities: %s", dataframe["city"].nunique())

    return dataframe


# ============================================================
# DATA OVERVIEW
# ============================================================

def dataset_overview(df):

    overview = {
        "total_records": len(df),
        "total_cities": df["city"].nunique(),
        "start_time": str(df["time"].min()),
        "end_time": str(df["time"].max()),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_records": int(
            df.duplicated(subset=["city", "time"]).sum()
        ),
    }

    report = pd.DataFrame(
        list(overview.items()),
        columns=["Metric", "Value"]
    )

    report.to_csv(
        OUTPUT_DIR / "dataset_overview.csv",
        index=False
    )

    return overview


# ============================================================
# CITY-WISE WEATHER ANALYSIS
# ============================================================

def city_analysis(df):

    summary = df.groupby("city").agg(
        temperature_avg=("temperature_2m", "mean"),
        temperature_min=("temperature_2m", "min"),
        temperature_max=("temperature_2m", "max"),

        humidity_avg=("relative_humidity_2m", "mean"),

        rainfall_total=("precipitation", "sum"),

        wind_speed_avg=("wind_speed_10m", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),

        pressure_avg=("surface_pressure", "mean"),

        cloud_cover_avg=("cloud_cover", "mean"),

        rainy_hours=("rain_indicator", "sum"),

        records=("time", "count")
    ).round(2)

    summary = summary.sort_values(
        "temperature_avg",
        ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / "city_weather_summary.csv"
    )

    logger.info("City analysis completed.")

    return summary


# ============================================================
# DAILY WEATHER TRENDS
# ============================================================

def daily_analysis(df):

    data = df.copy()

    data["date"] = data["time"].dt.date

    daily = data.groupby(
        ["city", "date"]
    ).agg(
        temperature_avg=("temperature_2m", "mean"),
        temperature_min=("temperature_2m", "min"),
        temperature_max=("temperature_2m", "max"),

        humidity_avg=("relative_humidity_2m", "mean"),

        rainfall_total=("precipitation", "sum"),

        wind_speed_avg=("wind_speed_10m", "mean")
    ).reset_index()

    daily.to_csv(
        OUTPUT_DIR / "daily_weather_summary.csv",
        index=False
    )

    logger.info("Daily weather analysis completed.")

    return daily


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "surface_pressure",
        "cloud_cover"
    ]

    correlation = df[columns].corr().round(3)

    correlation.to_csv(
        OUTPUT_DIR / "weather_correlation.csv"
    )

    return correlation


# ============================================================
# TEMPERATURE VISUALIZATION
# ============================================================

def temperature_chart(df):

    fig = px.line(
        df,
        x="time",
        y="temperature_2m",
        color="city",
        title="Hourly Temperature Trends Across Indian Cities",
        labels={
            "time": "Date and Time",
            "temperature_2m": "Temperature (°C)",
            "city": "City"
        },
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html(
        OUTPUT_DIR / "temperature_trends.html"
    )

    return fig


# ============================================================
# RAINFALL VISUALIZATION
# ============================================================

def rainfall_chart(summary):

    fig = px.bar(
        summary.reset_index(),
        x="city",
        y="rainfall_total",
        color="rainfall_total",
        title="Total Forecast Rainfall by City",
        labels={
            "city": "City",
            "rainfall_total": "Rainfall (mm)"
        },
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "rainfall_comparison.html"
    )

    return fig


# ============================================================
# HUMIDITY VISUALIZATION
# ============================================================

def humidity_chart(summary):

    fig = px.bar(
        summary.reset_index(),
        x="city",
        y="humidity_avg",
        title="Average Relative Humidity by City",
        labels={
            "city": "City",
            "humidity_avg": "Humidity (%)"
        },
        color="humidity_avg",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "humidity_comparison.html"
    )

    return fig


# ============================================================
# WEATHER CORRELATION VISUALIZATION
# ============================================================

def correlation_chart(correlation):

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Weather Variable Correlation Matrix"
    )

    fig.write_html(
        OUTPUT_DIR / "weather_correlation.html"
    )

    return fig


# ============================================================
# MAIN ANALYTICS ENGINE
# ============================================================

def run_analytics():

    logger.info("=" * 55)
    logger.info("WEATHER ANALYTICS ENGINE STARTED")
    logger.info("=" * 55)

    df = load_data()

    overview = dataset_overview(df)

    city_summary = city_analysis(df)

    daily_summary = daily_analysis(df)

    correlation = correlation_analysis(df)

    temperature_chart(df)

    rainfall_chart(city_summary)

    humidity_chart(city_summary)

    correlation_chart(correlation)

    logger.info("All analytics completed successfully.")

    print("\n========== NATIONAL WEATHER ANALYTICS ==========")

    print("\nDATASET OVERVIEW")

    for key, value in overview.items():
        print(f"{key}: {value}")

    print("\nCITY-WISE WEATHER SUMMARY")

    print(city_summary.to_string())

    print("\nTOP 3 WARMEST CITIES")

    print(
        city_summary.nlargest(
            3,
            "temperature_avg"
        )[["temperature_avg"]]
    )

    print("\nTOP 3 RAINIEST CITIES")

    print(
        city_summary.nlargest(
            3,
            "rainfall_total"
        )[["rainfall_total"]]
    )

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    return {
        "overview": overview,
        "city_summary": city_summary,
        "daily_summary": daily_summary,
        "correlation": correlation
    }


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    run_analytics()
