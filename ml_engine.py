
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "processed" / "historical_clean.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "data" / "processed" / "ml_results"
LOG_DIR = ROOT / "logs"

for folder in (MODEL_DIR, RESULT_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "weather_code",
]

LAGS = [1, 3, 6, 24, 168]
ROLLING_WINDOWS = [24, 168]

RAIN_THRESHOLD_MM = 0.1
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "ml_engine.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("weather_ml")


# =========================================================
# 2. DATA LOADING
# =========================================================

def load_data() -> pd.DataFrame:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}\n"
            "Run data_pipeline.py in historical mode first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {"time", "city", *TARGETS}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["time", "city"]
    ).copy()

    df["city"] = df["city"].astype(str)

    for column in WEATHER_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    duplicate_count = int(
        df.duplicated(["city", "time"]).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate city-hour records. "
            "Resolve duplicates before training."
        )

    logger.info(
        "Loaded %s rows across %s cities.",
        len(df),
        df["city"].nunique(),
    )

    logger.info(
        "Date range: %s to %s",
        df["time"].min(),
        df["time"].max(),
    )

    return df


# =========================================================
# 3. FEATURE ENGINEERING AND FUTURE LABELS
# =========================================================

def build_forecasting_dataset(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict]:

    df = source.copy()

    grouped = df.groupby(
        "city",
        sort=False,
    )

    # Next timestamp within each city.
    next_time = grouped["time"].shift(-1)

    df["target_time"] = next_time

    # Only exact one-hour transitions are valid.
    exact_next_hour = next_time.eq(
        df["time"] + pd.Timedelta(hours=1)
    )

    # Future targets: observation at t predicts target at t+1.
    for target in TARGETS:
        df[f"future_{target}"] = grouped[target].shift(-1)

    feature_columns = []

    available_weather = [
        column
        for column in WEATHER_FEATURES
        if column in df.columns
    ]

    # Current measurements, historical lags and rolling means.
    for column in available_weather:

        feature_columns.append(column)

        for lag in LAGS:

            feature_name = f"{column}_lag_{lag}"

            df[feature_name] = grouped[column].shift(lag)

            feature_columns.append(feature_name)

        for window in ROLLING_WINDOWS:

            feature_name = f"{column}_rolling_mean_{window}"

            df[feature_name] = grouped[column].transform(
                lambda values: values.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )

            feature_columns.append(feature_name)

    # Calendar features based on observation time.
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.day_of_year
    df["day_of_week"] = df["time"].dt.day_of_week
    df["month"] = df["time"].dt.month

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )
    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["weekday_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    calendar_columns = [
        "hour",
        "month",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "weekday_sin",
        "weekday_cos",
    ]

    feature_columns.extend(calendar_columns)

    # One-hot encode city.
    city_dummies = pd.get_dummies(
        df["city"],
        prefix="city",
        dtype=np.uint8,
    )

    df = pd.concat(
        [df, city_dummies],
        axis=1,
    )

    city_columns = city_dummies.columns.tolist()
    feature_columns.extend(city_columns)

    future_columns = [
        f"future_{target}"
        for target in TARGETS
    ]

    # Keep only valid one-hour labels and complete predictors.
    valid_rows = exact_next_hour.copy()

    valid_rows &= df["target_time"].notna()
    valid_rows &= df[future_columns].notna().all(axis=1)
    valid_rows &= df[feature_columns].notna().all(axis=1)

    model_df = df.loc[valid_rows].copy()

    model_df = model_df.sort_values(
        ["target_time", "city"]
    ).reset_index(drop=True)

    if model_df.empty:
        raise ValueError(
            "No usable forecasting rows remain. "
            "Check hourly continuity and missing values."
        )

    # Confirm target timestamps.
    expected_target_time = (
        model_df["time"] + pd.Timedelta(hours=1)
    )

    if not model_df["target_time"].equals(expected_target_time):
        raise ValueError(
            "Invalid target timestamps. Expected exactly one hour ahead."
        )

    metadata = {
        "source_rows": int(len(source)),
        "usable_rows": int(len(model_df)),
        "excluded_rows": int(len(source) - len(model_df)),
        "cities": sorted(
            model_df["city"].unique().tolist()
        ),
        "first_observation": str(model_df["time"].min()),
        "last_observation": str(model_df["time"].max()),
        "first_target_time": str(model_df["target_time"].min()),
        "last_target_time": str(model_df["target_time"].max()),
        "forecast_horizon": "1 hour ahead",
        "continuity_rule": (
            "Only exact one-hour transitions within each city are retained."
        ),
        "feature_count": len(feature_columns),
        "features": feature_columns,
    }

    logger.info(
        "Forecast dataset: %s usable rows; excluded %s rows.",
        len(model_df),
        len(source) - len(model_df),
    )

    logger.info(
        "Predictor count: %s",
        len(feature_columns),
    )

    return model_df, feature_columns, metadata


# =========================================================
# 4. CHRONOLOGICAL DATA SPLITTING
# =========================================================

def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    timestamps = np.array(
        sorted(df["target_time"].dropna().unique())
    )

    if len(timestamps) < 20:
        raise ValueError(
            "Not enough unique timestamps for chronological splitting."
        )

    train_index = int(len(timestamps) * 0.70)
    validation_index = int(len(timestamps) * 0.85)

    train_end = pd.Timestamp(
        timestamps[train_index - 1]
    )

    validation_end = pd.Timestamp(
        timestamps[validation_index - 1]
    )

    train = df[
        df["target_time"] <= train_end
    ].copy()

    validation = df[
        (df["target_time"] > train_end)
        & (df["target_time"] <= validation_end)
    ].copy()

    test = df[
        df["target_time"] > validation_end
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "One of the chronological splits is empty."
        )

    split_info = {
        "split_basis": "target_time",
        "train_target_start": str(train["target_time"].min()),
        "train_target_end": str(train["target_time"].max()),
        "validation_target_start": str(validation["target_time"].min()),
        "validation_target_end": str(validation["target_time"].max()),
        "test_target_start": str(test["target_time"].min()),
        "test_target_end": str(test["target_time"].max()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }

    logger.info(
        "Split sizes — train: %s | validation: %s | test: %s",
        len(train),
        len(validation),
        len(test),
    )

    logger.info(
        "Target-time boundaries — train ends: %s | validation ends: %s",
        train["target_time"].max(),
        validation["target_time"].max(),
    )

    return train, validation, test, split_info


# =========================================================
# 5. COMMON EVALUATION METRICS
# =========================================================

def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


def rainfall_event_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    actual_wet = actual > RAIN_THRESHOLD_MM
    predicted_wet = predicted > RAIN_THRESHOLD_MM

    return {
        "precision": float(
            precision_score(
                actual_wet,
                predicted_wet,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_wet,
                predicted_wet,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_wet,
                predicted_wet,
                zero_division=0,
            )
        ),
        "actual_wet_hours": int(actual_wet.sum()),
        "predicted_wet_hours": int(predicted_wet.sum()),
        "false_rain_alarms": int(
            (predicted_wet & ~actual_wet).sum()
        ),
        "missed_rain_hours": int(
            (actual_wet & ~predicted_wet).sum()
        ),
    }


# =========================================================
# 6. MODEL CANDIDATES
# =========================================================

def get_model_candidates() -> dict:

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    }

    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    except ImportError:
        logger.warning(
            "XGBoost unavailable. Random Forest will be used."
        )

    return candidates


def get_rainfall_classifier_candidates() -> dict:

    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoostClassifier"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    except ImportError:
        logger.warning(
            "XGBoost unavailable. Random Forest classifier will be used."
        )

    return candidates


# =========================================================
# 7. RAINFALL THRESHOLD SELECTION
# =========================================================

def select_rain_threshold(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict]:

    actual_wet = actual > RAIN_THRESHOLD_MM

    best_threshold = 0.5
    best_metrics = None
    best_key = (-1.0, -1.0)

    for threshold in np.arange(0.05, 0.951, 0.025):

        predicted_wet = probabilities >= threshold

        precision = precision_score(
            actual_wet,
            predicted_wet,
            zero_division=0,
        )

        recall = recall_score(
            actual_wet,
            predicted_wet,
            zero_division=0,
        )

        f1 = f1_score(
            actual_wet,
            predicted_wet,
            zero_division=0,
        )

        # Select highest F1; use precision to break ties.
        candidate_key = (f1, precision)

        if candidate_key > best_key:
            best_key = candidate_key
            best_threshold = float(threshold)

            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best_threshold, best_metrics


# =========================================================
# 8. TWO-STAGE RAINFALL MODEL
# =========================================================

def train_rainfall_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:

    target = "precipitation"
    label = f"future_{target}"

    X_train = train[feature_columns]
    X_validation = validation[feature_columns]
    X_test = test[feature_columns]

    y_train = train[label].to_numpy(dtype=float)
    y_validation = validation[label].to_numpy(dtype=float)
    y_test = test[label].to_numpy(dtype=float)

    y_train_wet = (
        y_train > RAIN_THRESHOLD_MM
    ).astype(int)

    y_validation_wet = (
        y_validation > RAIN_THRESHOLD_MM
    ).astype(int)

    if len(np.unique(y_train_wet)) < 2:
        raise ValueError(
            "Rainfall training data must contain wet and dry hours."
        )

    if y_train_wet.sum() < 20:
        raise ValueError(
            "Not enough wet training records for rainfall modelling."
        )

    if y_validation_wet.sum() == 0:
        raise ValueError(
            "Validation data contains no wet hours."
        )

    # -----------------------------------------------------
    # STAGE 1: RAINFALL CLASSIFICATION
    # -----------------------------------------------------

    logger.info("STAGE 1: Training rainfall classifiers.")

    classifier_results = {}
    classifier_models = {}

    for name, classifier in get_rainfall_classifier_candidates().items():

        logger.info("Training %s", name)

        classifier.fit(
            X_train,
            y_train_wet,
        )

        validation_probabilities = classifier.predict_proba(
            X_validation
        )[:, 1]

        threshold, threshold_metrics = select_rain_threshold(
            y_validation,
            validation_probabilities,
        )

        classifier_results[name] = {
            "threshold": threshold,
            "validation_metrics": threshold_metrics,
        }

        classifier_models[name] = classifier

        logger.info(
            "%s validation precision: %.4f | recall: %.4f | "
            "F1: %.4f | threshold: %.3f",
            name,
            threshold_metrics["precision"],
            threshold_metrics["recall"],
            threshold_metrics["f1"],
            threshold,
        )

    best_classifier_name = max(
        classifier_results,
        key=lambda name: (
            classifier_results[name]["validation_metrics"]["f1"],
            classifier_results[name]["validation_metrics"]["precision"],
        ),
    )

    selected_classifier = classifier_models[
        best_classifier_name
    ]

    selected_threshold = classifier_results[
        best_classifier_name
    ]["threshold"]

    logger.info(
        "Selected rainfall classifier: %s",
        best_classifier_name,
    )

    # The classifier remains fitted on training data because
    # its threshold was calibrated on validation probabilities.
    test_probabilities = selected_classifier.predict_proba(
        X_test
    )[:, 1]

    test_rain_event_prediction = (
        test_probabilities >= selected_threshold
    )

    # -----------------------------------------------------
    # STAGE 2: RAINFALL AMOUNT REGRESSION
    # -----------------------------------------------------

    logger.info(
        "STAGE 2: Training rainfall amount regressors on wet hours."
    )

    train_wet_mask = (
        y_train > RAIN_THRESHOLD_MM
    )

    validation_wet_mask = (
        y_validation > RAIN_THRESHOLD_MM
    )

    X_train_wet = X_train.loc[train_wet_mask]
    y_train_wet_amount = y_train[train_wet_mask]

    X_validation_wet = X_validation.loc[validation_wet_mask]
    y_validation_wet_amount = y_validation[validation_wet_mask]

    if len(y_validation_wet_amount) == 0:
        raise ValueError(
            "No wet validation observations for amount-regressor selection."
        )

    regressor_results = {}

    for name, regressor in get_model_candidates().items():

        logger.info(
            "Training rainfall amount regressor: %s",
            name,
        )

        regressor.fit(
            X_train_wet,
            y_train_wet_amount,
        )

        validation_amount_prediction = np.maximum(
            regressor.predict(X_validation_wet),
            0,
        )

        regressor_results[name] = regression_metrics(
            y_validation_wet_amount,
            validation_amount_prediction,
        )

        logger.info(
            "%s wet-hour validation MAE: %.5f",
            name,
            regressor_results[name]["mae"],
        )

    best_regressor_name = min(
        regressor_results,
        key=lambda name: regressor_results[name]["mae"],
    )

    logger.info(
        "Selected rainfall amount regressor: %s",
        best_regressor_name,
    )

    # Refit amount regressor using train + validation rainy hours.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    train_validation_wet = (
        train_validation[label] > RAIN_THRESHOLD_MM
    )

    final_regressor = get_model_candidates()[
        best_regressor_name
    ]

    final_regressor.fit(
        train_validation.loc[
            train_validation_wet,
            feature_columns,
        ],
        train_validation.loc[
            train_validation_wet,
            label,
        ].to_numpy(dtype=float),
    )

    # -----------------------------------------------------
    # FINAL TEST PREDICTIONS
    # -----------------------------------------------------

    amount_prediction = np.maximum(
        final_regressor.predict(X_test),
        0,
    )

    final_prediction = np.where(
        test_rain_event_prediction,
        amount_prediction,
        0.0,
    )

    # Persistence baseline: next hour equals current hour.
    baseline_prediction = np.maximum(
        test[target].to_numpy(dtype=float),
        0,
    )

    model_test_metrics = regression_metrics(
        y_test,
        final_prediction,
    )

    baseline_test_metrics = regression_metrics(
        y_test,
        baseline_prediction,
    )

    model_event_metrics = rainfall_event_metrics(
        y_test,
        final_prediction,
    )

    baseline_event_metrics = rainfall_event_metrics(
        y_test,
        baseline_prediction,
    )

    actual_wet_mask = (
        y_test > RAIN_THRESHOLD_MM
    )

    wet_hour_amount_mae = (
        float(
            mean_absolute_error(
                y_test[actual_wet_mask],
                final_prediction[actual_wet_mask],
            )
        )
        if actual_wet_mask.any()
        else None
    )

    result = {
        "target": target,
        "forecast_horizon": "1 hour ahead",
        "method": "Two-stage classification and regression",
        "selected_classifier": best_classifier_name,
        "selected_regressor": best_regressor_name,
        "classification_threshold": selected_threshold,
        "classification_validation_metrics": classifier_results[
            best_classifier_name
        ]["validation_metrics"],
        "regression_validation_metrics": regressor_results,
        "selected_model_test": model_test_metrics,
        "persistence_baseline_test": baseline_test_metrics,
        "rainfall_event_test_metrics": model_event_metrics,
        "persistence_baseline_event_metrics": baseline_event_metrics,
        "wet_hour_amount_mae": wet_hour_amount_mae,
        "test_improvement_over_baseline_mae": float(
            baseline_test_metrics["mae"] - model_test_metrics["mae"]
        ),
    }

    # Save both models and all inference metadata together.
    model_path = MODEL_DIR / "precipitation_model.joblib"

    joblib.dump(
        {
            "classifier": selected_classifier,
            "regressor": final_regressor,
            "features": feature_columns,
            "target": target,
            "forecast_horizon": "1 hour ahead",
            "wet_threshold_mm": RAIN_THRESHOLD_MM,
            "classification_threshold": selected_threshold,
            "classifier_name": best_classifier_name,
            "regressor_name": best_regressor_name,
        },
        model_path,
    )

    logger.info(
        "Saved two-stage rainfall model: %s",
        model_path,
    )

    # Save predictions.
    prediction_df = test[
        [
            "time",
            "target_time",
            "city",
            target,
            label,
        ]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={
            "time": "observation_time",
            target: "current_observation",
            label: "actual_next_hour",
        }
    )

    prediction_df["predicted_next_hour"] = final_prediction
    prediction_df["persistence_prediction"] = baseline_prediction
    prediction_df["rain_probability"] = test_probabilities
    prediction_df["rain_predicted"] = test_rain_event_prediction
    prediction_df["rain_threshold"] = selected_threshold
    prediction_df["target_name"] = target

    logger.info(
        "Rainfall test MAE: %.5f | baseline MAE: %.5f",
        model_test_metrics["mae"],
        baseline_test_metrics["mae"],
    )

    logger.info(
        "Rainfall test precision: %.4f | recall: %.4f | F1: %.4f",
        model_event_metrics["precision"],
        model_event_metrics["recall"],
        model_event_metrics["f1"],
    )

    return result, prediction_df


# =========================================================
# 9. OTHER WEATHER REGRESSION MODELS
# =========================================================

def train_target_model(
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:

    label = f"future_{target}"

    X_train = train[feature_columns]
    y_train = train[label].to_numpy(dtype=float)

    X_validation = validation[feature_columns]
    y_validation = validation[label].to_numpy(dtype=float)

    X_test = test[feature_columns]
    y_test = test[label].to_numpy(dtype=float)

    baseline_validation = validation[target].to_numpy(dtype=float)
    baseline_test = test[target].to_numpy(dtype=float)

    validation_results = {}

    for name, model in get_model_candidates().items():

        logger.info(
            "Training %s for %s",
            name,
            target,
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_prediction = model.predict(
            X_validation
        )

        if target in ("precipitation", "wind_speed_10m"):
            validation_prediction = np.maximum(
                validation_prediction,
                0,
            )

        validation_results[name] = regression_metrics(
            y_validation,
            validation_prediction,
        )

        logger.info(
            "%s validation MAE: %.5f",
            name,
            validation_results[name]["mae"],
        )

    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["mae"],
    )

    logger.info(
        "Selected %s for %s",
        best_name,
        target,
    )

    # Refit on train + validation.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    final_model = get_model_candidates()[best_name]

    final_model.fit(
        train_validation[feature_columns],
        train_validation[label].to_numpy(dtype=float),
    )

    test_prediction = final_model.predict(X_test)

    if target in ("precipitation", "wind_speed_10m"):
        test_prediction = np.maximum(
            test_prediction,
            0,
        )

    model_test_metrics = regression_metrics(
        y_test,
        test_prediction,
    )

    baseline_test_metrics = regression_metrics(
        y_test,
        baseline_test,
    )

    result = {
        "target": target,
        "forecast_horizon": "1 hour ahead",
        "selected_model": best_name,
        "validation_metrics": validation_results,
        "selected_model_test": model_test_metrics,
        "persistence_baseline_test": baseline_test_metrics,
        "test_improvement_over_baseline_mae": float(
            baseline_test_metrics["mae"] - model_test_metrics["mae"]
        ),
    }

    model_path = MODEL_DIR / f"{target}_model.joblib"

    joblib.dump(
        {
            "model": final_model,
            "features": feature_columns,
            "target": target,
            "forecast_horizon": "1 hour ahead",
            "selected_model": best_name,
        },
        model_path,
    )

    logger.info(
        "Saved model: %s",
        model_path,
    )

    # Save feature importance.
    if hasattr(final_model, "feature_importances_"):

        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": final_model.feature_importances_,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        importance.to_csv(
            RESULT_DIR / f"{target}_feature_importance.csv",
            index=False,
        )

    prediction_df = test[
        [
            "time",
            "target_time",
            "city",
            target,
            label,
        ]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={
            "time": "observation_time",
            target: "current_observation",
            label: "actual_next_hour",
        }
    )

    prediction_df["predicted_next_hour"] = test_prediction
    prediction_df["persistence_prediction"] = baseline_test
    prediction_df["target_name"] = target

    logger.info(
        "%s test MAE: %.5f | baseline MAE: %.5f | R2: %.5f",
        target,
        model_test_metrics["mae"],
        baseline_test_metrics["mae"],
        model_test_metrics["r2"],
    )

    return result, prediction_df


# =========================================================
# 10. ANOMALY DETECTION
# =========================================================

def run_anomaly_detection(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:

    X = df[feature_columns].astype(float)

    timestamps = np.array(
        sorted(df["time"].unique())
    )

    train_end_index = int(
        len(timestamps) * 0.70
    ) - 1

    train_end = pd.Timestamp(
        timestamps[train_end_index]
    )

    train_mask = df["time"] <= train_end
    X_train = X.loc[train_mask]

    sample_size = min(
        50000,
        len(X_train),
    )

    if sample_size < 100:
        raise ValueError(
            "Not enough training records for anomaly detection."
        )

    X_sample = X_train.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    anomaly_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=150,
                    contamination=0.01,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logger.info(
        "Fitting Isolation Forest on %s training rows.",
        sample_size,
    )

    anomaly_pipeline.fit(X_sample)

    predictions = anomaly_pipeline.predict(X)
    scores = anomaly_pipeline.decision_function(X)

    output = df[
        ["time", "city"]
        + [
            column
            for column in TARGETS
            if column in df.columns
        ]
    ].copy()

    output["anomaly"] = predictions == -1
    output["anomaly_score"] = scores

    anomaly_file = RESULT_DIR / "weather_anomalies.csv"

    output.to_csv(
        anomaly_file,
        index=False,
    )

    joblib.dump(
        {
            "pipeline": anomaly_pipeline,
            "features": feature_columns,
            "contamination": 0.01,
        },
        MODEL_DIR / "isolation_forest.joblib",
    )

    anomaly_count = int(
        output["anomaly"].sum()
    )

    anomaly_percentage = (
        100 * anomaly_count / len(output)
    )

    logger.info(
        "Anomalies identified: %s / %s (%.3f%%)",
        anomaly_count,
        len(output),
        anomaly_percentage,
    )

    return {
        "records_checked": int(len(output)),
        "anomalies_detected": anomaly_count,
        "anomaly_percentage": float(anomaly_percentage),
        "model": "IsolationForest",
        "interpretation": (
            "Statistical outliers requiring domain validation; "
            "not automatically severe weather events."
        ),
        "output_file": str(anomaly_file),
    }


# =========================================================
# 11. EVALUATION REPORT CREATION
# =========================================================

def build_evaluation_table(
    all_results: dict,
) -> pd.DataFrame:

    rows = []

    for target, result in all_results.items():

        model_metrics = result["selected_model_test"]
        baseline_metrics = result["persistence_baseline_test"]

        if target == "precipitation":
            selected_model = (
                f"{result['selected_classifier']} + "
                f"{result['selected_regressor']}"
            )
        else:
            selected_model = result["selected_model"]

        row = {
            "target": target,
            "selected_model": selected_model,
            "test_mae": model_metrics["mae"],
            "test_rmse": model_metrics["rmse"],
            "test_r2": model_metrics["r2"],
            "baseline_test_mae": baseline_metrics["mae"],
            "baseline_test_rmse": baseline_metrics["rmse"],
            "baseline_test_r2": baseline_metrics["r2"],
            "mae_improvement_over_baseline": (
                result["test_improvement_over_baseline_mae"]
            ),
        }

        if target == "precipitation":

            event_metrics = result["rainfall_event_test_metrics"]

            row.update({
                "classification_threshold": (
                    result["classification_threshold"]
                ),
                "wet_hour_precision": event_metrics["precision"],
                "wet_hour_recall": event_metrics["recall"],
                "wet_hour_f1": event_metrics["f1"],
                "actual_wet_hours": event_metrics["actual_wet_hours"],
                "predicted_wet_hours": event_metrics["predicted_wet_hours"],
                "false_rain_alarms": event_metrics["false_rain_alarms"],
                "missed_rain_hours": event_metrics["missed_rain_hours"],
                "wet_hour_amount_mae": result["wet_hour_amount_mae"],
            })

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# 12. MAIN ML PIPELINE
# =========================================================

def run_ml_engine() -> None:

    logger.info("=" * 65)
    logger.info(
        "NATIONAL WEATHER ANALYTICS PLATFORM — ML ENGINE"
    )
    logger.info("=" * 65)

    # Load historical weather observations.
    source = load_data()

    # Build next-hour forecasting dataset.
    model_df, feature_columns, dataset_info = (
        build_forecasting_dataset(source)
    )

    # Chronological split based on target timestamp.
    train, validation, test, split_info = (
        chronological_split(model_df)
    )

    all_results = {}
    all_predictions = []

    # Train all four forecasting targets.
    for target in TARGETS:

        logger.info("-" * 60)
        logger.info(
            "Forecasting next-hour target: %s",
            target,
        )

        if target == "precipitation":

            result, predictions = train_rainfall_model(
                train=train,
                validation=validation,
                test=test,
                feature_columns=feature_columns,
            )

        else:

            result, predictions = train_target_model(
                target=target,
                train=train,
                validation=validation,
                test=test,
                feature_columns=feature_columns,
            )

        all_results[target] = result
        all_predictions.append(predictions)

    # Save all test predictions.
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = RESULT_DIR / "test_predictions.csv"

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # Save detailed model evaluation JSON.
    evaluation_json_path = (
        RESULT_DIR / "model_evaluation.json"
    )

    with open(
        evaluation_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            allow_nan=False,
        )

    # Save evaluation CSV.
    evaluation_df = build_evaluation_table(
        all_results
    )

    evaluation_csv_path = (
        RESULT_DIR / "model_evaluation.csv"
    )

    evaluation_df.to_csv(
        evaluation_csv_path,
        index=False,
    )

    # Run anomaly detection.
    anomaly_results = run_anomaly_detection(
        model_df,
        feature_columns,
    )

    # Save run metadata.
    metadata = {
        "project": "National Weather Analytics Platform",
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "forecast_horizon": "1 hour ahead",
        "dataset": dataset_info,
        "split": split_info,
        "models": all_results,
        "anomaly_detection": anomaly_results,
        "important_notes": [
            "Forecast labels are shifted exactly one hour ahead within each city.",
            "Rows without exact one-hour transitions are excluded.",
            "Train, validation and test splits use target_time.",
            "Model selection uses validation data.",
            "Final model metrics use the held-out test data.",
            "Rainfall uses a two-stage classifier and amount regressor.",
            "Rainfall threshold is tuned using validation data only.",
            "Persistence predicts that the next hour equals the current observation.",
            "R-squared is not percentage accuracy.",
            "Anomaly flags are statistical outliers requiring domain validation.",
        ],
    }

    metadata_path = (
        RESULT_DIR / "ml_run_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    # -----------------------------------------------------
    # FINAL CONSOLE REPORT
    # -----------------------------------------------------

    logger.info("=" * 65)
    logger.info("ML ENGINE COMPLETED")
    logger.info("Predictions: %s", predictions_path)
    logger.info("Evaluation CSV: %s", evaluation_csv_path)
    logger.info("Evaluation JSON: %s", evaluation_json_path)
    logger.info("Metadata: %s", metadata_path)
    logger.info("=" * 65)

    print("\n========== FINAL TEST RESULTS ==========")

    for target, result in all_results.items():

        metrics = result["selected_model_test"]
        baseline = result["persistence_baseline_test"]

        print(f"\n{target}")

        if target == "precipitation":

            print(
                "Classifier:",
                result["selected_classifier"],
            )

            print(
                "Regressor:",
                result["selected_regressor"],
            )

            print(
                "Classification threshold:",
                result["classification_threshold"],
            )

        else:

            print(
                "Selected model:",
                result["selected_model"],
            )

        print(
            f"Model test MAE: {metrics['mae']:.5f}"
        )

        print(
            f"Baseline test MAE: {baseline['mae']:.5f}"
        )

        print(
            f"Model test RMSE: {metrics['rmse']:.5f}"
        )

        print(
            f"Model test R2: {metrics['r2']:.5f}"
        )

        if target == "precipitation":

            event_metrics = result[
                "rainfall_event_test_metrics"
            ]

            print("\nRainfall event metrics:")

            for key, value in event_metrics.items():
                print(f"  {key}: {value}")

            print(
                "Wet-hour amount MAE:",
                result["wet_hour_amount_mae"],
            )

    print("\n========== EVALUATION TABLE ==========")
    print(
        evaluation_df.to_string(
            index=False
        )
    )

    print("\n========== ANOMALY DETECTION ==========")
    print(
        json.dumps(
            anomaly_results,
            indent=2,
        )
    )


if __name__ == "__main__":
    run_ml_engine()
