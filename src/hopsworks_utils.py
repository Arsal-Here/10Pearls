"""
Hopsworks Feature Store and Model Registry utilities.

Provides helper functions for connecting to Hopsworks, managing feature
groups, creating feature views, and interacting with the model registry.
"""

import logging
from typing import Optional

import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_NAME,
    PRIMARY_KEY,
    EVENT_TIME,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connection
# =============================================================================

def get_hopsworks_project():
    """
    Authenticate and return the Hopsworks project handle.

    Uses HOPSWORKS_API_KEY and HOPSWORKS_PROJECT from environment.

    Returns:
        hopsworks.Project instance.
    """
    import hopsworks

    if not HOPSWORKS_API_KEY:
        raise EnvironmentError(
            "HOPSWORKS_API_KEY is not set. "
            "Please set it in your .env file or environment variables."
        )

    logger.info("Connecting to Hopsworks project: %s", HOPSWORKS_PROJECT or "(default)")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    logger.info("Successfully connected to Hopsworks")
    return project


def get_feature_store(project):
    """Get the Feature Store handle from a Hopsworks project."""
    fs = project.get_feature_store()
    logger.info("Feature Store accessed: %s", fs.name)
    return fs


# =============================================================================
# Feature Group Management
# =============================================================================

def get_or_create_feature_group(fs, df: pd.DataFrame):
    """
    Get or create the Karachi AQI feature group.

    Args:
        fs: Hopsworks FeatureStore instance.
        df: Sample DataFrame to infer schema (used only on creation).

    Returns:
        Hopsworks FeatureGroup instance.
    """
    logger.info(
        "Getting/creating feature group: %s (v%d)",
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    )

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        description="Hourly air quality and weather features for Karachi, Pakistan. "
                    "Includes raw pollutants, weather variables, time features, "
                    "rolling aggregates, lag features, AQI change rates, and "
                    "forward-looking AQI targets.",
        online_enabled=False,
    )

    logger.info("Feature group ready: %s", fg.name)
    return fg


def insert_features(fg, df: pd.DataFrame) -> None:
    """
    Insert (upsert) features into the feature group.

    Deduplication is handled by the primary key (timestamp).

    Args:
        fg: Hopsworks FeatureGroup instance.
        df: DataFrame of features to insert.
    """
    n_rows = len(df)
    logger.info("Inserting %d rows into feature group '%s'...", n_rows, fg.name)

    # Ensure all column names are lowercase and valid
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Drop rows where all pollutant values are NaN
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
    available_pollutants = [c for c in pollutant_cols if c in df.columns]
    if available_pollutants:
        df = df.dropna(subset=available_pollutants, how="all")

    fg.insert(df, write_options={"wait_for_job": False})
    logger.info("Successfully inserted %d rows", len(df))


# =============================================================================
# Feature View Management
# =============================================================================

def get_or_create_feature_view(fs, fg):
    """
    Get or create a Feature View for training/inference.

    The Feature View selects all columns from the feature group.

    Args:
        fs: Hopsworks FeatureStore instance.
        fg: Hopsworks FeatureGroup instance.

    Returns:
        Hopsworks FeatureView instance.
    """
    logger.info(
        "Getting/creating feature view: %s (v%d)",
        FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    )

    try:
        fv = fs.get_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
        )
        logger.info("Found existing feature view: %s", fv.name)
    except Exception:
        logger.info("Creating new feature view...")
        query = fg.select_all()
        fv = fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=query,
            description="Feature view for Karachi AQI prediction model training "
                        "and inference. Selects all features from the main "
                        "feature group.",
        )
        logger.info("Created feature view: %s", fv.name)

    return fv


def get_training_data(fv) -> pd.DataFrame:
    """
    Pull all training data from the Feature View.

    Returns:
        DataFrame with all features and targets.
    """
    logger.info("Pulling training data from feature view...")

    try:
        # Try getting data with read_options
        df, _ = fv.training_data(description="karachi_aqi_training")
    except Exception:
        # Fallback: get as DataFrame directly
        df = fv.get_batch_data()

    logger.info("Retrieved %d training rows × %d columns", len(df), len(df.columns))
    return df


# =============================================================================
# Model Registry
# =============================================================================

def get_model_registry(project):
    """Get the Model Registry handle from a Hopsworks project."""
    mr = project.get_model_registry()
    logger.info("Model Registry accessed")
    return mr


def register_model(
    mr,
    model_dir: str,
    metrics: dict,
    description: str = "Karachi AQI prediction model",
    model_version: Optional[int] = None,
):
    """
    Register a trained model in the Hopsworks Model Registry.

    Args:
        mr: Hopsworks ModelRegistry instance.
        model_dir: Local directory containing model artifacts.
        metrics: Dict of evaluation metrics (e.g., {"rmse": 5.2}).
        description: Human-readable description.
        model_version: Specific version number (None = auto-increment).

    Returns:
        Hopsworks Model instance.
    """
    logger.info("Registering model '%s' with metrics: %s", MODEL_NAME, metrics)

    model = mr.python.create_model(
        name=MODEL_NAME,
        version=model_version,
        metrics=metrics,
        description=description,
    )

    model.save(model_dir)
    logger.info("Model registered successfully: %s (v%s)", model.name, model.version)
    return model


def get_latest_model(mr):
    """
    Retrieve the latest version of the registered model.

    Returns:
        Hopsworks Model instance.
    """
    logger.info("Fetching latest model: %s", MODEL_NAME)

    model = mr.get_best_model(
        name=MODEL_NAME,
        metric="rmse",
        direction="min",
    )

    logger.info(
        "Retrieved model: %s v%s (metrics: %s)",
        model.name, model.version, model.training_metrics,
    )
    return model


def download_model(model, local_dir: str = "models") -> str:
    """
    Download model artifacts to a local directory.

    Args:
        model: Hopsworks Model instance.
        local_dir: Local directory to save to.

    Returns:
        Path to the downloaded model directory.
    """
    logger.info("Downloading model artifacts to '%s'...", local_dir)
    model_dir = model.download()
    logger.info("Model downloaded to: %s", model_dir)
    return model_dir
