
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="National Weather Analytics Platform",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# API HELPERS
# ============================================================

def api_get(endpoint, params=None):
    """Send GET request to FastAPI."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.RequestException as error:
        return None, str(error)


@st.cache_data(ttl=300)
def get_cities():
    data, error = api_get("/weather/cities")

    if error or not isinstance(data, dict):
        return [], error

    return data.get("cities", []), None


@st.cache_data(ttl=300)
def get_forecast(city, hours):
    return api_get(
        f"/weather/forecast/{city}",
        params={"hours": hours}
    )


@st.cache_data(ttl=300)
def get_analytics():
    return api_get("/weather/analytics")


@st.cache_data(ttl=300)
def get_evaluation():
    return api_get("/ml/evaluation")


@st.cache_data(ttl=300)
def get_anomalies():
    return api_get("/weather/anomalies")


@st.cache_data(ttl=300)
def get_predictions(city):
    return api_get(f"/ml/predictions/{city}")


@st.cache_data(ttl=300)
def get_model_status():
    return api_get("/ml/models")


def extract_records(payload):
    """Find a list of dictionary records in an API response."""

    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            return payload
        return []

    if isinstance(payload, dict):
        preferred_keys = [
            "data", "records", "results", "predictions",
            "anomalies", "evaluation", "models"
        ]

        for key in preferred_keys:
            value = payload.get(key)

            if isinstance(value, list):
                if not value or all(isinstance(x, dict) for x in value):
                    return value

        for value in payload.values():
            if isinstance(value, (dict, list)):
                records = extract_records(value)
                if records:
                    return records

    return []


def make_dataframe(payload):
    records = extract_records(payload)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def find_column(df, possible_names):
    """Find a column without depending on letter case."""

    lookup = {str(col).lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def prepare_forecast_dataframe(df):
    """Standardize forecast fields for dashboard charts."""

    if df.empty:
        return df

    time_col = find_column(df, ["time", "timestamp", "datetime"])

    if time_col:
        df[time_col] = pd.to_datetime(
            df[time_col],
            errors="coerce"
        )
        df = df.rename(columns={time_col: "time"})

    return df


def metric_column(df, possible_names):
    return find_column(df, possible_names)


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ National Weather Analytics Platform")

st.markdown(
    "### Intelligent Weather Monitoring, Analytics & Machine Learning"
)

st.caption(
    "Multi-city weather intelligence platform | "
    "Data pipeline • Analytics • Machine Learning • REST API"
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("### Location Selection")

cities, cities_error = get_cities()

if cities_error or not cities:
    st.sidebar.error("Unable to retrieve cities from the API.")
    st.error(
        "Cannot load city information. Make sure the FastAPI server "
        "is running at http://127.0.0.1:8000."
    )
    st.stop()

city = st.sidebar.selectbox(
    "Select City",
    options=cities,
    index=0
)

hours = st.sidebar.slider(
    "Forecast Duration (Hours)",
    min_value=1,
    max_value=168,
    value=24,
    step=1
)

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("API: " + API_BASE_URL)

# ============================================================
# API HEALTH
# ============================================================

health, health_error = api_get("/health")

if health_error:
    st.error("Backend API is unavailable.")
    st.stop()

st.sidebar.success("Backend API Connected")

# ============================================================
# FETCH FORECAST DATA
# ============================================================

forecast_response, forecast_error = get_forecast(city, hours)

if forecast_error:
    st.error(f"Unable to retrieve forecast: {forecast_error}")
    st.stop()

forecast_df = make_dataframe(forecast_response)
forecast_df = prepare_forecast_dataframe(forecast_df)

if forecast_df.empty:
    st.warning("No forecast records were returned by the API.")
    st.stop()

# ============================================================
# STANDARD WEATHER COLUMNS
# ============================================================

temperature_col = metric_column(
    forecast_df,
    ["temperature_2m", "temperature", "temperature_c"]
)

humidity_col = metric_column(
    forecast_df,
    ["relative_humidity_2m", "relative_humidity", "humidity"]
)

rain_col = metric_column(
    forecast_df,
    ["precipitation", "rainfall", "rain"]
)

wind_col = metric_column(
    forecast_df,
    ["wind_speed_10m", "wind_speed", "windspeed"]
)


def numeric_values(column):
    if column is None:
        return pd.Series(dtype="float64")

    return pd.to_numeric(
        forecast_df[column],
        errors="coerce"
    ).dropna()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader(f"Weather Overview — {city}")

temperature_values = numeric_values(temperature_col)
humidity_values = numeric_values(humidity_col)
rain_values = numeric_values(rain_col)
wind_values = numeric_values(wind_col)

col1, col2, col3, col4 = st.columns(4)

with col1:
    value = (
        f"{temperature_values.mean():.1f} °C"
        if not temperature_values.empty else "N/A"
    )
    st.metric("Average Temperature", value)

with col2:
    value = (
        f"{humidity_values.mean():.1f}%"
        if not humidity_values.empty else "N/A"
    )
    st.metric("Average Humidity", value)

with col3:
    value = (
        f"{rain_values.sum():.2f} mm"
        if not rain_values.empty else "N/A"
    )
    st.metric("Total Forecast Rainfall", value)

with col4:
    value = (
        f"{wind_values.mean():.1f} km/h"
        if not wind_values.empty else "N/A"
    )
    st.metric("Average Wind Speed", value)

st.caption(
    f"Showing {len(forecast_df)} forecast records. "
    "Metrics summarize the selected forecast period."
)

st.divider()

# ============================================================
# DASHBOARD TABS
# ============================================================

overview_tab, forecast_tab, ml_tab, anomaly_tab, data_tab = st.tabs(
    [
        "📊 Weather Overview",
        "🌤️ Forecast Analysis",
        "🤖 ML Performance",
        "⚠️ Anomaly Monitoring",
        "📁 Raw Data"
    ]
)

# ============================================================
# TAB 1: WEATHER OVERVIEW
# ============================================================

with overview_tab:

    st.subheader("Weather Trends")

    if "time" not in forecast_df.columns:
        st.warning("The forecast response does not contain a time column.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if temperature_col:
                fig = px.line(
                    forecast_df,
                    x="time",
                    y=temperature_col,
                    title="Temperature Trend",
                    markers=True,
                    labels={
                        temperature_col: "Temperature (°C)",
                        "time": "Time"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            if humidity_col:
                fig = px.line(
                    forecast_df,
                    x="time",
                    y=humidity_col,
                    title="Humidity Trend",
                    markers=True,
                    labels={
                        humidity_col: "Humidity (%)",
                        "time": "Time"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if rain_col:
                fig = px.bar(
                    forecast_df,
                    x="time",
                    y=rain_col,
                    title="Hourly Precipitation",
                    labels={
                        rain_col: "Precipitation (mm)",
                        "time": "Time"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            if wind_col:
                fig = px.line(
                    forecast_df,
                    x="time",
                    y=wind_col,
                    title="Wind Speed Trend",
                    markers=True,
                    labels={
                        wind_col: "Wind Speed (km/h)",
                        "time": "Time"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Multi-City Analytics")

    analytics_data, analytics_error = get_analytics()

    if analytics_error:
        st.warning(f"Analytics unavailable: {analytics_error}")
    else:
        analytics_df = make_dataframe(analytics_data)

        if not analytics_df.empty:
            st.dataframe(
                analytics_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.json(analytics_data)

# ============================================================
# TAB 2: FORECAST ANALYSIS
# ============================================================

with forecast_tab:

    st.subheader(f"{city} — Hourly Weather Forecast")

    if "time" in forecast_df.columns:
        available_metrics = {
            "Temperature": temperature_col,
            "Humidity": humidity_col,
            "Precipitation": rain_col,
            "Wind Speed": wind_col
        }

        available_metrics = {
            name: column
            for name, column in available_metrics.items()
            if column is not None
        }

        if available_metrics:
            selected_metric = st.selectbox(
                "Choose Weather Variable",
                options=list(available_metrics.keys())
            )

            selected_column = available_metrics[selected_metric]

            fig = px.line(
                forecast_df,
                x="time",
                y=selected_column,
                title=f"{selected_metric} Forecast — {city}",
                markers=True
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Records")

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 3: MACHINE LEARNING PERFORMANCE
# ============================================================

with ml_tab:

    st.subheader("Machine Learning Model Evaluation")

    evaluation_data, evaluation_error = get_evaluation()

    if evaluation_error:
        st.error(f"Unable to load ML evaluation: {evaluation_error}")
    else:
        evaluation_df = make_dataframe(evaluation_data)

        if not evaluation_df.empty:
            st.dataframe(
                evaluation_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Model Evaluation Metrics")

            numeric_cols = evaluation_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_cols:
                selected_metric = st.selectbox(
                    "Select Metric to Compare",
                    options=numeric_cols,
                    key="evaluation_metric"
                )

                label_col = find_column(
                    evaluation_df,
                    ["target", "model", "target_name", "variable"]
                )

                if label_col:
                    fig = px.bar(
                        evaluation_df,
                        x=label_col,
                        y=selected_metric,
                        title=f"Model Comparison — {selected_metric}",
                        text_auto=".3f"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        else:
            st.json(evaluation_data)

    st.divider()

    st.subheader("Historical ML Predictions")

    prediction_data, prediction_error = get_predictions(city)

    if prediction_error:
        st.warning(f"Predictions unavailable: {prediction_error}")
    else:
        prediction_df = make_dataframe(prediction_data)

        if not prediction_df.empty:
            st.info(
                "These are historical held-out test predictions, "
                "not live forecasts."
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

            actual_col = find_column(
                prediction_df,
                ["actual_next_hour"]
            )

            predicted_col = find_column(
                prediction_df,
                ["predicted_next_hour"]
            )

            if actual_col and predicted_col:
                prediction_time = find_column(
                    prediction_df,
                    ["target_time", "observation_time"]
                )

                fig = px.line(
                    prediction_df,
                    x=prediction_time if prediction_time else None,
                    y=[actual_col, predicted_col],
                    title="Actual vs Predicted Values"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
        else:
            st.json(prediction_data)

    st.divider()

    st.subheader("Saved ML Models")

    model_data, model_error = get_model_status()

    if model_error:
        st.warning(f"Model status unavailable: {model_error}")
    else:
        st.json(model_data)

# ============================================================
# TAB 4: ANOMALY MONITORING
# ============================================================

with anomaly_tab:

    st.subheader("Weather Anomaly Detection")

    anomaly_data, anomaly_error = get_anomalies()

    if anomaly_error:
        st.error(f"Unable to load anomalies: {anomaly_error}")
    else:
        anomaly_df = make_dataframe(anomaly_data)

        if not anomaly_df.empty:

            city_column = find_column(
                anomaly_df,
                ["city"]
            )

            if city_column:
                city_anomalies = anomaly_df[
                    anomaly_df[city_column].astype(str).str.lower()
                    == city.lower()
                ]
            else:
                city_anomalies = anomaly_df

            st.metric(
                "Anomaly Records for Selected City",
                len(city_anomalies)
            )

            st.dataframe(
                city_anomalies,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No anomaly records were returned.")
            st.json(anomaly_data)

    st.caption(
        "Anomalies are statistical outliers detected by the model. "
        "They are not automatically confirmed severe weather events."
    )

# ============================================================
# TAB 5: RAW DATA
# ============================================================

with data_tab:

    st.subheader("Forecast Dataset")

    st.write(
        "Inspect the raw forecast records returned by the backend API."
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=f"{city.lower()}_weather_forecast.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "National Weather Analytics Platform | "
    "FastAPI • Streamlit • Pandas • Plotly • Scikit-learn • XGBoost"
)

st.caption(
    "Weather forecasts are provider-supplied. "
    "ML prediction records represent historical test-set evaluations."
)
