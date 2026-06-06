"""
MongoDB Feature Store utilities.

Provides helper functions for connecting to MongoDB, upserting feature rows,
and pulling historical data for model training.
"""

import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

from bson.binary import Binary
from datetime import datetime

from src.config import (
    MONGODB_DATABASE,
    MONGODB_FEATURE_COLLECTION,
    MONGODB_MODEL_COLLECTION,
    MONGODB_URI,
    PRIMARY_KEY,
)

logger = logging.getLogger(__name__)


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_mongodb_uri(prompt_if_missing: bool = False) -> str:
    """
    Resolve the MongoDB URI from env/.env.

    If prompt_if_missing=True and running in an interactive terminal, prompt
    the user for MONGODB_URI.
    """
    uri = _clean_value(os.getenv("MONGODB_URI")) or MONGODB_URI
    if uri:
        return uri

    if prompt_if_missing and sys.stdin and sys.stdin.isatty():
        entered = input("Enter MONGODB_URI for the MongoDB feature store: ").strip()
        if entered:
            os.environ["MONGODB_URI"] = entered
            return entered

    raise EnvironmentError(
        "MONGODB_URI is not set. Please set it in your .env file or environment variables."
    )


def get_mongo_client(prompt_if_missing: bool = False) -> MongoClient:
    """
    Create and return an authenticated MongoClient.
    """
    uri = resolve_mongodb_uri(prompt_if_missing=prompt_if_missing)
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000,
        appname="karachi-aqi-service",
    )
    client.admin.command("ping")
    logger.info("Successfully connected to MongoDB")
    return client


def get_feature_collection(client: MongoClient) -> Collection:
    """
    Return the configured MongoDB collection used as feature store.
    """
    database_name = _clean_value(os.getenv("MONGODB_DATABASE")) or MONGODB_DATABASE
    collection_name = (
        _clean_value(os.getenv("MONGODB_FEATURE_COLLECTION")) or MONGODB_FEATURE_COLLECTION
    )

    collection = client[database_name][collection_name]
    ensure_feature_indexes(collection)
    logger.info("Feature collection ready: %s.%s", database_name, collection_name)
    return collection


def get_model_collection(client: MongoClient) -> Collection:
    """
    Return the configured MongoDB collection used for model storage.
    """
    database_name = _clean_value(os.getenv("MONGODB_DATABASE")) or MONGODB_DATABASE
    collection_name = (
        _clean_value(os.getenv("MONGODB_MODEL_COLLECTION")) or MONGODB_MODEL_COLLECTION
    )

    collection = client[database_name][collection_name]
    collection.create_index([("filename", ASCENDING)], unique=True)
    logger.info("Model collection ready: %s.%s", database_name, collection_name)
    return collection


def upload_model_file(collection: Collection, filename: str, content_bytes: bytes) -> None:
    """
    Upload or update a model artifact in MongoDB.
    """
    collection.update_one(
        {"filename": filename},
        {
            "$set": {
                "filename": filename,
                "content": Binary(content_bytes),
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )
    logger.info("Successfully uploaded '%s' to MongoDB cloud", filename)


def download_model_file(collection: Collection, filename: str) -> bytes | None:
    """
    Download a model artifact from MongoDB.
    """
    doc = collection.find_one({"filename": filename})
    if doc and "content" in doc:
        logger.info("Successfully downloaded '%s' from MongoDB cloud", filename)
        return doc["content"]
    logger.warning("File '%s' not found in MongoDB model store", filename)
    return None


def ensure_feature_indexes(collection: Collection) -> None:
    """
    Ensure unique indexes for primary key columns used during upserts.
    """
    if not PRIMARY_KEY:
        return

    if len(PRIMARY_KEY) == 1:
        collection.create_index([(PRIMARY_KEY[0], ASCENDING)], unique=True)
    else:
        collection.create_index([(field, ASCENDING) for field in PRIMARY_KEY], unique=True)


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return int(value.value // 10**6)

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _sanitize_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize feature DataFrame before upserting into MongoDB.
    """
    if df.empty:
        return df.copy()

    clean_df = df.copy()
    clean_df.columns = [col.lower().replace(" ", "_") for col in clean_df.columns]

    if "timestamp" not in clean_df.columns:
        raise ValueError("Feature DataFrame must include a 'timestamp' column.")

    ts = clean_df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        clean_df["timestamp"] = pd.to_datetime(ts).astype("int64") // 10**6
    elif not pd.api.types.is_integer_dtype(ts):
        parsed = pd.to_datetime(ts, errors="coerce")
        if parsed.notna().all():
            clean_df["timestamp"] = parsed.astype("int64") // 10**6
        else:
            clean_df["timestamp"] = pd.to_numeric(ts, errors="coerce")

    clean_df = clean_df.dropna(subset=["timestamp"]).copy()
    clean_df["timestamp"] = clean_df["timestamp"].astype("int64")

    # Drop rows where all pollutant values are NaN
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
    available_pollutants = [c for c in pollutant_cols if c in clean_df.columns]
    if available_pollutants:
        clean_df = clean_df.dropna(subset=available_pollutants, how="all")

    # Deduplicate by primary key
    clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="last")
    return clean_df


def upsert_features(collection: Collection, df: pd.DataFrame, batch_size: int = 2000) -> int:
    """
    Upsert features into MongoDB using timestamp as primary key.

    Returns:
        Number of rows prepared for write.
    """
    clean_df = _sanitize_feature_dataframe(df)
    if clean_df.empty:
        logger.warning("No valid feature rows available for upsert")
        return 0

    operations: list[UpdateOne] = []
    upserted_total = 0
    modified_total = 0

    for record in clean_df.to_dict(orient="records"):
        doc = {k: _normalize_scalar(v) for k, v in record.items()}
        timestamp = doc.get("timestamp")
        if timestamp is None:
            continue

        operations.append(
            UpdateOne(
                {"timestamp": timestamp},
                {"$set": doc},
                upsert=True,
            )
        )

        if len(operations) >= batch_size:
            result = collection.bulk_write(operations, ordered=False)
            upserted_total += result.upserted_count
            modified_total += result.modified_count
            operations.clear()

    if operations:
        result = collection.bulk_write(operations, ordered=False)
        upserted_total += result.upserted_count
        modified_total += result.modified_count

    logger.info(
        "MongoDB upsert complete: rows=%d, inserted=%d, updated=%d",
        len(clean_df),
        upserted_total,
        modified_total,
    )
    return len(clean_df)


def get_training_data(collection: Collection) -> pd.DataFrame:
    """
    Pull all feature rows from MongoDB for model training.
    """
    rows = list(collection.find({}, {"_id": 0}))
    if not rows:
        raise ValueError("No feature rows found in MongoDB feature collection.")

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Retrieved %d training rows × %d columns", len(df), len(df.columns))
    return df
