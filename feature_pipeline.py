"""
Feature Pipeline — Karachi AQI Prediction Service

Fetches air quality and weather data from Open-Meteo, computes ML features,
and inserts them into the Hopsworks Feature Store.

Usage:
    # Hourly incremental ingestion (default)
    python feature_pipeline.py

    # Full 2-year historical backfill
    python feature_pipeline.py --backfill

    # Backfill a custom date range
    python feature_pipeline.py --backfill --start-date 2024-06-01 --end-date 2025-01-01
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

from src.config import BACKFILL_START_DATE, BACKFILL_CHUNK_DAYS
from src.data_fetcher import (
    fetch_combined_recent,
    fetch_historical_in_chunks,
)
from src.feature_engineering import build_feature_dataframe
from src.hopsworks_utils import (
    get_hopsworks_project,
    get_feature_store,
    get_or_create_feature_group,
    insert_features,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_pipeline")


# ---------------------------------------------------------------------------
# Pipeline Modes
# ---------------------------------------------------------------------------

def run_hourly_pipeline() -> None:
    """
    Hourly incremental pipeline:
      1. Fetch last 5 days of data (overlap for rolling features)
      2. Compute features
      3. Upsert into Hopsworks feature group
    """
    logger.info("=" * 60)
    logger.info("STARTING HOURLY FEATURE PIPELINE")
    logger.info("=" * 60)

    # Step 1: Fetch recent data
    logger.info("Step 1/3: Fetching recent air quality + weather data...")
    raw_df = fetch_combined_recent(past_days=5, forecast_days=0)
    logger.info("Fetched %d raw rows", len(raw_df))

    if raw_df.empty:
        logger.warning("No data returned from API — skipping pipeline run")
        return

    # Step 2: Compute features
    logger.info("Step 2/3: Computing features...")
    feature_df = build_feature_dataframe(raw_df)
    logger.info("Computed %d features × %d rows", len(feature_df.columns), len(feature_df))

    # Step 3: Insert into Hopsworks
    logger.info("Step 3/3: Connecting to Hopsworks and inserting features...")
    project = get_hopsworks_project()
    fs = get_feature_store(project)
    fg = get_or_create_feature_group(fs, feature_df)
    insert_features(fg, feature_df)

    logger.info("=" * 60)
    logger.info("HOURLY FEATURE PIPELINE COMPLETE — %d rows inserted", len(feature_df))
    logger.info("=" * 60)


def run_backfill_pipeline(start_date: str, end_date: str) -> None:
    """
    Historical backfill pipeline:
      1. Fetch data in monthly chunks from start_date to end_date
      2. Compute features for entire range
      3. Insert into Hopsworks feature group
    """
    logger.info("=" * 60)
    logger.info("STARTING BACKFILL PIPELINE: %s → %s", start_date, end_date)
    logger.info("=" * 60)

    # Step 1: Fetch historical data in chunks
    logger.info("Step 1/3: Fetching historical data in %d-day chunks...", BACKFILL_CHUNK_DAYS)
    raw_df = fetch_historical_in_chunks(
        start_date=start_date,
        end_date=end_date,
        chunk_days=BACKFILL_CHUNK_DAYS,
        sleep_between=1.0,
    )
    logger.info("Fetched %d total raw rows", len(raw_df))

    if raw_df.empty:
        logger.warning("No historical data fetched — aborting backfill")
        return

    # Step 2: Compute features
    logger.info("Step 2/3: Computing features for %d rows...", len(raw_df))
    feature_df = build_feature_dataframe(raw_df)
    logger.info("Computed %d features × %d rows", len(feature_df.columns), len(feature_df))

    # Step 3: Insert into Hopsworks (in batches if very large)
    logger.info("Step 3/3: Connecting to Hopsworks and inserting features...")
    project = get_hopsworks_project()
    fs = get_feature_store(project)
    fg = get_or_create_feature_group(fs, feature_df)

    # Insert in batches of 5000 to avoid memory issues on free tier
    batch_size = 5000
    total_rows = len(feature_df)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = feature_df.iloc[start_idx:end_idx]
        logger.info(
            "Inserting batch %d–%d of %d...",
            start_idx + 1, end_idx, total_rows,
        )
        insert_features(fg, batch)

    logger.info("=" * 60)
    logger.info("BACKFILL PIPELINE COMPLETE — %d total rows inserted", total_rows)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Karachi AQI Feature Pipeline — fetch data and store features in Hopsworks",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help="Run historical backfill instead of hourly incremental ingestion",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=BACKFILL_START_DATE,
        help=f"Backfill start date (YYYY-MM-DD). Default: {BACKFILL_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backfill end date (YYYY-MM-DD). Default: yesterday",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the feature pipeline."""
    args = parse_args()

    try:
        if args.backfill:
            end_date = args.end_date or (
                datetime.utcnow() - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            run_backfill_pipeline(args.start_date, end_date)
        else:
            run_hourly_pipeline()

    except Exception as exc:
        logger.exception("Feature pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Feature pipeline finished successfully")


if __name__ == "__main__":
    main()
