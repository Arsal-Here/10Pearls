"""
Centralized configuration for the Karachi AQI Prediction Service.

Contains all constants, API endpoints, feature definitions, EPA breakpoints,
and AQI category thresholds used across all pipeline stages.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Hopsworks Configuration
# =============================================================================
def _get_clean_env(key: str) -> str | None:
    val = os.getenv(key)
    if val is not None:
        val = val.strip()
        if val == "":
            return None
    return val

HOPSWORKS_API_KEY = _get_clean_env("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = _get_clean_env("HOPSWORKS_PROJECT") or _get_clean_env("HOPSWORKS_PROJECT_NAME")


FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "karachi_aqi_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "karachi_aqi_model"

# =============================================================================
# Karachi Geolocation
# =============================================================================
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
CITY_NAME = "Karachi, Pakistan"

# =============================================================================
# Open-Meteo API Endpoints
# =============================================================================
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# =============================================================================
# API Variable Names
# =============================================================================
AIR_QUALITY_HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "carbon_monoxide",
]

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

# =============================================================================
# Feature Column Definitions
# =============================================================================
# Raw pollutant columns (as returned from API after renaming)
POLLUTANT_COLS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

# Raw weather columns (after renaming)
WEATHER_COLS = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure"]

# Time-based feature columns
TIME_FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

# Target columns (AQI at future horizons)
TARGET_COLS = ["aqi_target_24h", "aqi_target_48h", "aqi_target_72h"]

# Primary key for feature group (must be unique)
PRIMARY_KEY = ["timestamp"]
EVENT_TIME = "timestamp"

# =============================================================================
# Rolling / Lag Window Sizes
# =============================================================================
ROLLING_WINDOWS = [6, 12, 24]       # hours
LAG_STEPS = [1, 3, 6, 12, 24]       # hours
CHANGE_RATE_WINDOWS = [3, 6, 12]    # hours

# =============================================================================
# Backfill Configuration
# =============================================================================
BACKFILL_START_DATE = "2024-01-01"
BACKFILL_CHUNK_DAYS = 30  # Fetch in monthly chunks to avoid API limits

# =============================================================================
# EPA AQI Breakpoint Tables
# =============================================================================
# Format: list of (C_low, C_high, I_low, I_high)
# Reference: EPA Technical Assistance Document for AQI (EPA-454/B-18-007)

# PM2.5 (µg/m³, 24-hour average, truncated to 0.1)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

# PM10 (µg/m³, 24-hour average, truncated to integer)
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

# NO2 (µg/m³, 1-hour average — converted from ppb at standard conditions)
# Note: Open-Meteo returns µg/m³. EPA uses ppb. We convert using 1 ppb ≈ 1.88 µg/m³
NO2_BREAKPOINTS_UGM3 = [
    (0, 100, 0, 50),
    (101, 188, 51, 100),
    (189, 677, 101, 150),
    (678, 1221, 151, 200),
    (1222, 2349, 201, 300),
    (2350, 3102, 301, 400),
    (3103, 3853, 401, 500),
]

# SO2 (µg/m³, 1-hour average — 1 ppb ≈ 2.62 µg/m³)
SO2_BREAKPOINTS_UGM3 = [
    (0, 93, 0, 50),
    (94, 197, 51, 100),
    (198, 486, 101, 150),
    (487, 797, 151, 200),
    (798, 1583, 201, 300),
    (1584, 2107, 301, 400),
    (2108, 2631, 401, 500),
]

# O3 (µg/m³, 1-hour average — 1 ppb ≈ 1.96 µg/m³)
O3_BREAKPOINTS_UGM3 = [
    (0, 108, 0, 50),
    (109, 137, 51, 100),
    (138, 167, 101, 150),
    (168, 392, 151, 200),
    (393, 784, 201, 300),
    (785, 980, 301, 400),
    (981, 1176, 401, 500),
]

# CO (µg/m³, 8-hour average — 1 ppm ≈ 1145 µg/m³)
CO_BREAKPOINTS_UGM3 = [
    (0, 5152, 0, 50),
    (5153, 10944, 51, 100),
    (10945, 14198, 101, 150),
    (14199, 17452, 151, 200),
    (17453, 34904, 201, 300),
    (34905, 46538, 301, 400),
    (46539, 57590, 401, 500),
]

# =============================================================================
# AQI Category Thresholds & Display
# =============================================================================
AQI_CATEGORIES = [
    {"range": (0, 50), "label": "Good", "color": "#00E400", "emoji": "🟢",
     "message": "Air quality is satisfactory with little or no risk."},
    {"range": (51, 100), "label": "Moderate", "color": "#FFFF00", "emoji": "🟡",
     "message": "Acceptable; moderate health concern for sensitive individuals."},
    {"range": (101, 150), "label": "Unhealthy for Sensitive Groups", "color": "#FF7E00",
     "emoji": "🟠", "message": "Sensitive groups may experience health effects."},
    {"range": (151, 200), "label": "Unhealthy", "color": "#FF0000", "emoji": "🔴",
     "message": "Everyone may begin to experience health effects."},
    {"range": (201, 300), "label": "Very Unhealthy", "color": "#8F3F97", "emoji": "🟣",
     "message": "Health alert: everyone may experience serious effects."},
    {"range": (301, 500), "label": "Hazardous", "color": "#7E0023", "emoji": "🟤",
     "message": "Health warning of emergency conditions for entire population."},
]


def get_aqi_category(aqi_value: float) -> dict:
    """Return the AQI category dict for a given AQI value."""
    for cat in AQI_CATEGORIES:
        low, high = cat["range"]
        if low <= aqi_value <= high:
            return cat
    # Above 500 — beyond index
    return AQI_CATEGORIES[-1]
