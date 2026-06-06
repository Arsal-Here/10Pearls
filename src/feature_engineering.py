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

from .aqi_calculator import calculate_aqi
from .config import (
    POLLUTANT_COLS,
    ROLLING_WINDOWS,
    LAG_STEPS,
    CHANGE_RATE_WINDOWS,
    EWMA_SPANS,
    INTERACTION_PAIRS,
    DAY_SEGMENTS,
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
# Exponential Weighted Moving Average (EWMA) Features
# =============================================================================

def compute_ewma_features(
    df: pd.DataFrame,
    spans: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute exponential weighted moving averages for key pollutants and AQI.

    EWMA gives more weight to recent observations, making it better at
    capturing trend shifts compared to simple rolling means.

    Args:
        df: DataFrame sorted by timestamp.
        spans: List of span (half-life) sizes in hours.

    Returns:
        DataFrame with EWMA columns added.
    """
    if spans is None:
        spans = EWMA_SPANS

    df = df.copy()
    ewma_cols = ["pm2_5", "pm10", "aqi"]

    for col in ewma_cols:
        if col not in df.columns:
            continue
        for span in spans:
            df[f"{col}_ewma_{span}h"] = (
                df[col].ewm(span=span, min_periods=1).mean()
            )

    n_new = len(ewma_cols) * len(spans)
    logger.info("Computed EWMA features: %d columns added", n_new)
    return df


# =============================================================================
# Rolling Min / Max / Range Features
# =============================================================================

def compute_rolling_minmax_features(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute rolling min, max, and range (max-min) for key variables.

    Range captures volatility — high range indicates unstable air quality.

    Args:
        df: DataFrame sorted by timestamp.
        windows: List of rolling window sizes in hours.

    Returns:
        DataFrame with rolling min/max/range columns added.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    minmax_cols = ["pm2_5", "pm10", "aqi"]

    for col in minmax_cols:
        if col not in df.columns:
            continue
        for window in windows:
            roll = df[col].rolling(window=window, min_periods=1)
            df[f"{col}_rolling_min_{window}h"] = roll.min()
            df[f"{col}_rolling_max_{window}h"] = roll.max()
            df[f"{col}_range_{window}h"] = (
                df[f"{col}_rolling_max_{window}h"] - df[f"{col}_rolling_min_{window}h"]
            )

    n_new = len(minmax_cols) * len(windows) * 3
    logger.info("Computed rolling min/max/range features: %d columns added", n_new)
    return df


# =============================================================================
# Pollutant × Weather Interaction Features
# =============================================================================

def compute_interaction_features(
    df: pd.DataFrame,
    pairs: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Compute interaction features between pollutants and weather variables.

    Captures effects like humidity amplifying PM2.5 impact, or wind speed
    aiding pollutant dispersion.

    Args:
        df: DataFrame with pollutant and weather columns.
        pairs: List of (pollutant_col, weather_col) tuples.

    Returns:
        DataFrame with interaction columns added.
    """
    if pairs is None:
        pairs = INTERACTION_PAIRS

    df = df.copy()
    n_new = 0

    for col_a, col_b in pairs:
        if col_a in df.columns and col_b in df.columns:
            df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
            n_new += 1

    # Temperature-Humidity Index: proxy for atmospheric conditions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_idx"] = df["temperature"] * df["humidity"] / 100.0
        n_new += 1

    logger.info("Computed interaction features: %d columns added", n_new)
    return df


# =============================================================================
# Advanced Time Features
# =============================================================================

def compute_advanced_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional time-based features beyond the basics.

    Features:
        - day_segment (0=Night, 1=Morning, 2=Afternoon, 3=Evening)
        - day_of_year_sin, day_of_year_cos (cyclical day-of-year)
        - week_of_year_sin, week_of_year_cos (cyclical week)
    """
    df = df.copy()

    # Day segment from hour (requires 'hour' column to exist)
    if "hour" in df.columns:
        conditions = []
        choices = []
        for idx, (segment_name, (start, end)) in enumerate(DAY_SEGMENTS.items()):
            conditions.append((df["hour"] >= start) & (df["hour"] < end))
            choices.append(idx)
        df["day_segment"] = np.select(conditions, choices, default=0)

    # Day-of-year cyclical encoding — requires timestamp parsing
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            day_of_year = ts.dt.dayofyear
            df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
            df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

            week_of_year = ts.dt.isocalendar().week.astype(int)
            df["week_of_year_sin"] = np.sin(2 * np.pi * week_of_year / 52)
            df["week_of_year_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    logger.info("Computed advanced time features: 5 columns added")
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
        3. Compute advanced time features (day segment, day-of-year)
        4. Compute rolling window features
        5. Compute EWMA features
        6. Compute rolling min/max/range features
        7. Compute lag features
        8. Compute AQI change rate features
        9. Compute pollutant × weather interaction features
        10. Compute forward-looking targets

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

    # Step 3: Advanced time features (day segment, day-of-year cyclical)
    df = compute_advanced_time_features(df)

    # Step 4: Rolling features
    df = compute_rolling_features(df)

    # Step 5: EWMA features (exponential weighted moving averages)
    df = compute_ewma_features(df)

    # Step 6: Rolling min/max/range features
    df = compute_rolling_minmax_features(df)

    # Step 7: Lag features
    df = compute_lag_features(df)

    # Step 8: Change rate features
    df = compute_change_rate_features(df)

    # Step 9: Pollutant × weather interaction features
    df = compute_interaction_features(df)

    # Step 10: Targets (only meaningful for training data)
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
