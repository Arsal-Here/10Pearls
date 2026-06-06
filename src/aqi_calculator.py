"""
EPA Air Quality Index (AQI) calculator.

Implements the standard EPA linear interpolation formula to convert raw
pollutant concentrations (µg/m³) into AQI sub-indices, then computes
the overall AQI as the maximum sub-index.

Reference: EPA Technical Assistance Document (EPA-454/B-18-007)
"""

import logging
from typing import Optional

from .config import (
    PM25_BREAKPOINTS,
    PM10_BREAKPOINTS,
    NO2_BREAKPOINTS_UGM3,
    SO2_BREAKPOINTS_UGM3,
    O3_BREAKPOINTS_UGM3,
    CO_BREAKPOINTS_UGM3,
)

logger = logging.getLogger(__name__)


def _truncate_pm25(value: float) -> float:
    """Truncate PM2.5 to 1 decimal place per EPA rules."""
    return int(value * 10) / 10.0


def _truncate_integer(value: float) -> int:
    """Truncate to integer per EPA rules (PM10, NO2, SO2, etc.)."""
    return int(value)


def calculate_sub_aqi(
    concentration: float,
    breakpoints: list[tuple[float, float, int, int]],
) -> Optional[float]:
    """
    Calculate the AQI sub-index for a single pollutant using EPA linear
    interpolation.

    Args:
        concentration: Truncated pollutant concentration.
        breakpoints: List of (C_low, C_high, I_low, I_high) tuples.

    Returns:
        The sub-AQI value, or None if concentration is out of range.
    """
    if concentration < 0:
        return None

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)

    # Concentration exceeds the highest breakpoint — cap at 500
    if concentration > breakpoints[-1][1]:
        logger.warning(
            "Concentration %.1f exceeds max breakpoint %.1f — capping AQI at 500",
            concentration,
            breakpoints[-1][1],
        )
        return 500

    return None


def calculate_aqi(
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    no2: Optional[float] = None,
    so2: Optional[float] = None,
    o3: Optional[float] = None,
    co: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate the overall AQI from individual pollutant concentrations.

    The overall AQI is the **maximum** of all valid sub-indices.
    All concentrations should be in µg/m³.

    Args:
        pm25: PM2.5 concentration (µg/m³)
        pm10: PM10 concentration (µg/m³)
        no2:  Nitrogen dioxide concentration (µg/m³)
        so2:  Sulphur dioxide concentration (µg/m³)
        o3:   Ozone concentration (µg/m³)
        co:   Carbon monoxide concentration (µg/m³)

    Returns:
        The overall AQI value, or None if no valid sub-indices.
    """
    sub_indices = []

    pollutant_configs = [
        ("PM2.5", pm25, _truncate_pm25, PM25_BREAKPOINTS),
        ("PM10", pm10, _truncate_integer, PM10_BREAKPOINTS),
        ("NO2", no2, _truncate_integer, NO2_BREAKPOINTS_UGM3),
        ("SO2", so2, _truncate_integer, SO2_BREAKPOINTS_UGM3),
        ("O3", o3, _truncate_integer, O3_BREAKPOINTS_UGM3),
        ("CO", co, _truncate_integer, CO_BREAKPOINTS_UGM3),
    ]

    for name, value, truncate_fn, breakpoints in pollutant_configs:
        if value is not None and value >= 0:
            try:
                truncated = truncate_fn(value)
                sub_aqi = calculate_sub_aqi(truncated, breakpoints)
                if sub_aqi is not None:
                    sub_indices.append((name, sub_aqi))
            except Exception:
                logger.warning("Error computing sub-AQI for %s (value=%.2f)", name, value)

    if not sub_indices:
        return None

    # Overall AQI = max of all sub-indices
    dominant_pollutant, overall_aqi = max(sub_indices, key=lambda x: x[1])
    logger.debug(
        "AQI = %d (dominant: %s). Sub-indices: %s",
        overall_aqi,
        dominant_pollutant,
        sub_indices,
    )
    return overall_aqi


def calculate_all_sub_indices(
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    no2: Optional[float] = None,
    so2: Optional[float] = None,
    o3: Optional[float] = None,
    co: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """
    Calculate individual AQI sub-indices for all pollutants.

    Returns:
        Dictionary mapping pollutant name to its sub-AQI.
    """
    results = {}

    pollutant_configs = [
        ("PM2.5", pm25, _truncate_pm25, PM25_BREAKPOINTS),
        ("PM10", pm10, _truncate_integer, PM10_BREAKPOINTS),
        ("NO2", no2, _truncate_integer, NO2_BREAKPOINTS_UGM3),
        ("SO2", so2, _truncate_integer, SO2_BREAKPOINTS_UGM3),
        ("O3", o3, _truncate_integer, O3_BREAKPOINTS_UGM3),
        ("CO", co, _truncate_integer, CO_BREAKPOINTS_UGM3),
    ]

    for name, value, truncate_fn, breakpoints in pollutant_configs:
        if value is not None and value >= 0:
            try:
                truncated = truncate_fn(value)
                results[name] = calculate_sub_aqi(truncated, breakpoints)
            except Exception:
                results[name] = None
        else:
            results[name] = None

    return results
