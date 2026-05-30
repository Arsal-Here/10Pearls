"""
Feature engineering for the Karachi AQI prediction service.

Computes time-based, rolling window, lag, and AQI change-rate features
from raw pollutant + weather data. Also computes forward-looking AQI
targets for model training.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.aqi_calculator import calculate_aqi
from src.config import (
    POLLUTANT_COLS,
    ROLLING_WINDOWS,
    LAG_STEPS,
    CHANGE_RATE_WINDOWS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AQI Computation (Vectorized)
# =============================================================================

def compute_aqi_column(df: pd.DataFrame) -> pd.Series:
    """
    Compute the AQI for each row in the DataFrame.

    Expects columns: pm2_5, pm10, no2, so2, o3, co
    Returns: Series of AQI values.
    """
    logger.info("Computing AQI for %d rows...", len(df))

    def _row_aqi(row):
        try:
            return calculate_aqi(
                pm25=row.get("pm2_5"),
                pm10=row.get("pm10"),
                no2=row.get("no2"),
                so2=row.get("so2"),
                o3=row.get("o3"),
                co=row.get("co"),
            )
        except Exception:
            return np.nan

    return df.apply(_row_aqi, axis=1)


# =============================================================================
# Time-Based Features
# =============================================================================

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the timestamp column.

    Features:
        - hour (0–23)
        - day_of_week (0=Monday, 6=Sunday)
        - month (1–12)
        - is_weekend (0 or 1)
        - hour_sin, hour_cos (cyclical encoding)
        - month_sin, month_cos (cyclical encoding)
    """
    ts = pd.to_datetime(df["timestamp"])

    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Computed time features: %d columns added", 8)
    return df


# =============================================================================
# Rolling Window Features
# =============================================================================

def compute_rolling_features(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation for key pollutants.

    Args:
        df: DataFrame sorted by timestamp.
        windows: List of rolling window sizes in hours.

    Returns:
        DataFrame with rolling feature columns added.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    roll_cols = ["pm2_5", "pm10", "aqi"]

    for col in roll_cols:
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_rolling_mean_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_rolling_std_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).std().fillna(0)
            )

    n_new = len(roll_cols) * len(windows) * 2
    logger.info("Computed rolling features: %d columns added", n_new)
    return df


# =============================================================================
# Lag Features
# =============================================================================

def compute_lag_features(
    df: pd.DataFrame,
    lag_steps: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute lagged values for pollutants and AQI.

    Args:
        df: DataFrame sorted by timestamp.
        lag_steps: List of lag periods in hours.

    Returns:
        DataFrame with lag feature columns added.
    """
    if lag_steps is None:
        lag_steps = LAG_STEPS

    df = df.copy()
    lag_cols = ["pm2_5", "pm10", "aqi"]

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_steps:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    n_new = len(lag_cols) * len(lag_steps)
    logger.info("Computed lag features: %d columns added", n_new)
    return df


# =============================================================================
# AQI Change Rate Features
# =============================================================================

def compute_change_rate_features(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute the rate of AQI change over specified time windows.

    Change rate = (current_aqi - aqi_N_hours_ago) / N

    Args:
        df: DataFrame sorted by timestamp with 'aqi' column.
        windows: List of look-back windows in hours.

    Returns:
        DataFrame with change rate columns added.
    """
    if windows is None:
        windows = CHANGE_RATE_WINDOWS

    df = df.copy()

    if "aqi" not in df.columns:
        logger.warning("Cannot compute change rate: 'aqi' column missing")
        return df

    for window in windows:
        df[f"aqi_change_rate_{window}h"] = (
            (df["aqi"] - df["aqi"].shift(window)) / window
        )

    logger.info("Computed AQI change rate features: %d columns added", len(windows))
    return df


# =============================================================================
# Target Computation
# =============================================================================

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward-looking AQI targets for training.

    Targets:
        - aqi_target_24h: AQI value 24 hours ahead
        - aqi_target_48h: AQI value 48 hours ahead
        - aqi_target_72h: AQI value 72 hours ahead
    """
    df = df.copy()

    if "aqi" not in df.columns:
        logger.warning("Cannot compute targets: 'aqi' column missing")
        return df

    df["aqi_target_24h"] = df["aqi"].shift(-24)
    df["aqi_target_48h"] = df["aqi"].shift(-48)
    df["aqi_target_72h"] = df["aqi"].shift(-72)

    n_valid = df["aqi_target_24h"].notna().sum()
    logger.info("Computed targets: %d rows have valid 24h target", n_valid)
    return df


# =============================================================================
# Full Feature Pipeline Orchestrator
# =============================================================================

def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on merged air quality +
    weather data.

    Pipeline steps:
        1. Compute current AQI from raw pollutants
        2. Compute time-based features
        3. Compute rolling window features
        4. Compute lag features
        5. Compute AQI change rate features
        6. Compute forward-looking targets

    Args:
        df: Merged DataFrame with columns: timestamp, pm2_5, pm10,
            no2, so2, o3, co, temperature, humidity, wind_speed,
            wind_direction, pressure.

    Returns:
        Feature-enriched DataFrame ready for insertion into the
        feature store.
    """
    logger.info("Starting feature engineering pipeline on %d rows", len(df))

    # Ensure sorted by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Step 1: Compute current AQI
    df["aqi"] = compute_aqi_column(df)

    # Step 2: Time features
    df = compute_time_features(df)

    # Step 3: Rolling features
    df = compute_rolling_features(df)

    # Step 4: Lag features
    df = compute_lag_features(df)

    # Step 5: Change rate features
    df = compute_change_rate_features(df)

    # Step 6: Targets (only meaningful for training data)
    df = compute_targets(df)

    # Convert timestamp to integer (Unix epoch ms) for storage compatibility
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .astype("int64") // 10**6  # milliseconds
    )

    initial_cols = len(df.columns)
    logger.info(
        "Feature engineering complete: %d rows × %d columns",
        len(df), initial_cols,
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns (excluding timestamp and targets).
    """
    exclude = {"timestamp", "aqi_target_24h", "aqi_target_48h", "aqi_target_72h"}
    return [col for col in df.columns if col not in exclude]
