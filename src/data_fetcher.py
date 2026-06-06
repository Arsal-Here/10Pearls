"""
Open-Meteo API client for fetching air quality and weather data.

Handles both real-time (forecast + recent past) and historical archive
queries. Implements retry logic with exponential backoff for robustness.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .config import (
    KARACHI_LAT,
    KARACHI_LON,
    AIR_QUALITY_URL,
    WEATHER_FORECAST_URL,
    WEATHER_ARCHIVE_URL,
    AIR_QUALITY_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry-capable HTTP session
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0
_TIMEOUT_SECONDS = 30


def _get_with_retry(url: str, params: dict, retries: int = _MAX_RETRIES) -> dict:
    """
    Perform a GET request with exponential backoff retry logic.

    Args:
        url: API endpoint URL.
        params: Query parameters.
        retries: Maximum retry attempts.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    session = requests.Session()
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()

            # Open-Meteo returns an "error" key on bad requests
            if "error" in data and data["error"]:
                raise ValueError(f"Open-Meteo API error: {data.get('reason', 'unknown')}")

            return data

        except (requests.RequestException, ValueError) as exc:
            last_exception = exc
            if attempt < retries:
                wait = _BACKOFF_FACTOR ** attempt
                logger.warning(
                    "API request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("API request failed after %d attempts: %s", retries, exc)

    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Air Quality Data Fetchers
# ---------------------------------------------------------------------------

def _parse_air_quality_response(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo air quality JSON response into a DataFrame."""
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("No hourly data found in air quality API response")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "time": "timestamp",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "carbon_monoxide": "co",
    })
    return df


def fetch_air_quality_history(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical air quality data from Open-Meteo for Karachi.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date:   End date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: timestamp, pm2_5, pm10, no2, so2, o3, co
    """
    logger.info("Fetching air quality history: %s to %s", start_date, end_date)

    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "start_date": start_date,
        "end_date": end_date,
    }

    data = _get_with_retry(AIR_QUALITY_URL, params)
    df = _parse_air_quality_response(data)

    logger.info("Fetched %d air quality records", len(df))
    return df


def fetch_air_quality_recent(
    past_days: int = 5,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch recent and forecast air quality data.

    Args:
        past_days: Number of past days to include.
        forecast_days: Number of forecast days to include.

    Returns:
        DataFrame with air quality variables.
    """
    logger.info(
        "Fetching recent air quality: past_days=%d, forecast_days=%d",
        past_days, forecast_days,
    )

    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    data = _get_with_retry(AIR_QUALITY_URL, params)
    df = _parse_air_quality_response(data)

    logger.info("Fetched %d recent air quality records", len(df))
    return df


# ---------------------------------------------------------------------------
# Weather Data Fetchers
# ---------------------------------------------------------------------------

def _parse_weather_response(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo weather JSON response into a DataFrame."""
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("No hourly data found in weather API response")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "time": "timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
    })
    return df


def fetch_weather_history(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo archive for Karachi.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date:   End date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: timestamp, temperature, humidity,
                                wind_speed, wind_direction, pressure
    """
    logger.info("Fetching weather history: %s to %s", start_date, end_date)

    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "start_date": start_date,
        "end_date": end_date,
    }

    data = _get_with_retry(WEATHER_ARCHIVE_URL, params)
    df = _parse_weather_response(data)

    logger.info("Fetched %d weather records", len(df))
    return df


def fetch_weather_recent(
    past_days: int = 5,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Fetch recent and forecast weather data.

    Args:
        past_days: Number of past days to include.
        forecast_days: Number of forecast days to include.

    Returns:
        DataFrame with weather variables.
    """
    logger.info(
        "Fetching recent weather: past_days=%d, forecast_days=%d",
        past_days, forecast_days,
    )

    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    data = _get_with_retry(WEATHER_FORECAST_URL, params)
    df = _parse_weather_response(data)

    logger.info("Fetched %d recent weather records", len(df))
    return df


# ---------------------------------------------------------------------------
# Combined Fetchers
# ---------------------------------------------------------------------------

def fetch_combined_history(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch and merge historical air quality + weather data.

    Returns:
        Merged DataFrame on timestamp, containing all pollutant and
        weather columns.
    """
    aq_df = fetch_air_quality_history(start_date, end_date)
    weather_df = fetch_weather_history(start_date, end_date)

    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    logger.info(
        "Merged historical data: %d rows (%d AQ + %d weather → %d matched)",
        len(merged), len(aq_df), len(weather_df), len(merged),
    )
    return merged


def fetch_combined_recent(
    past_days: int = 5,
    forecast_days: int = 0,
) -> pd.DataFrame:
    """
    Fetch and merge recent air quality + weather data.

    Returns:
        Merged DataFrame on timestamp.
    """
    aq_df = fetch_air_quality_recent(past_days, forecast_days)
    weather_df = fetch_weather_recent(past_days, forecast_days)

    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    logger.info("Merged recent data: %d rows", len(merged))
    return merged


def fetch_historical_in_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int = 30,
    sleep_between: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch historical data in monthly chunks to avoid API timeouts.

    Args:
        start_date: Overall start date 'YYYY-MM-DD'.
        end_date:   Overall end date 'YYYY-MM-DD'.
        chunk_days: Days per chunk.
        sleep_between: Seconds to sleep between API calls.

    Returns:
        Concatenated DataFrame of all chunks.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_chunks: list[pd.DataFrame] = []

    current = start
    chunk_num = 0

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        logger.info(
            "Fetching chunk %d: %s to %s",
            chunk_num + 1, chunk_start_str, chunk_end_str,
        )

        try:
            chunk_df = fetch_combined_history(chunk_start_str, chunk_end_str)
            all_chunks.append(chunk_df)
            chunk_num += 1
        except Exception as exc:
            logger.error(
                "Failed to fetch chunk %s to %s: %s — skipping",
                chunk_start_str, chunk_end_str, exc,
            )

        current = chunk_end + timedelta(days=1)

        if current < end and sleep_between > 0:
            time.sleep(sleep_between)

    if not all_chunks:
        raise RuntimeError("No data chunks were successfully fetched")

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined = combined.reset_index(drop=True)

    logger.info(
        "Historical fetch complete: %d total rows across %d chunks",
        len(combined), chunk_num,
    )
    return combined
