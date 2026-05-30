"""
Training Pipeline — Karachi AQI Prediction Service

Pulls features from MongoDB Feature Store, trains multiple models,
evaluates them, and saves model artifacts locally.

Usage:
    python training_pipeline.py
"""

import logging
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import TARGET_COLS
from src.feature_engineering import get_feature_columns
from src.mongodb_utils import (
    get_feature_collection,
    get_mongo_client,
    get_training_data,
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
logger = logging.getLogger("training_pipeline")


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_model_candidates() -> dict:
    """Return a dict of model name → untrained estimator."""
    return {
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "ridge_regression": Ridge(
            alpha=1.0,
            fit_intercept=True,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics.

    Returns:
        Dict with rmse, mae, r2 keys.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Compute feature importance from a trained model.

    Uses built-in feature_importances_ for tree models,
    coefficient magnitudes for linear models.

    Returns:
        DataFrame with columns: feature, importance (sorted desc).
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model %s has no feature importance attribute", model_name)
            return pd.DataFrame(columns=["feature", "importance"])

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # Log top 10
        logger.info("Top 10 features for %s:", model_name)
        for _, row in fi_df.head(10).iterrows():
            logger.info("  %-35s %.4f", row["feature"], row["importance"])

        return fi_df

    except Exception as exc:
        logger.warning("Failed to compute feature importance for %s: %s", model_name, exc)
        return pd.DataFrame(columns=["feature", "importance"])


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Prepare training and validation sets from the feature DataFrame.

    Uses a time-ordered 80/20 split (NOT random) to respect temporal
    dependencies.

    Returns:
        (X_train, X_val, y_train, y_val, feature_names)
    """
    # Drop rows where target is NaN
    df_clean = df.dropna(subset=[target_col]).copy()
    logger.info(
        "Rows after dropping NaN target (%s): %d / %d",
        target_col, len(df_clean), len(df),
    )

    if len(df_clean) < 100:
        raise ValueError(
            f"Insufficient training data: only {len(df_clean)} rows with valid "
            f"target '{target_col}'. Need at least 100."
        )

    # Feature columns = everything except timestamp and target columns
    feature_names = get_feature_columns(df_clean)
    logger.info("Using %d feature columns", len(feature_names))

    # Sort by timestamp (should already be sorted)
    if "timestamp" in df_clean.columns:
        df_clean = df_clean.sort_values("timestamp")

    X = df_clean[feature_names].values
    y = df_clean[target_col].values

    # Time-ordered split: 80% train, 20% validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Replace any remaining NaN/Inf in features with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Data split: train=%d rows, validation=%d rows",
        len(X_train), len(X_val),
    )

    return X_train, X_val, y_train, y_val, feature_names


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_and_evaluate_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    target_name: str,
) -> tuple:
    """
    Train all candidate models, evaluate them, and return the best.

    Returns:
        (best_model, best_model_name, best_metrics, all_results, best_fi_df)
    """
    models = get_model_candidates()
    all_results = {}
    best_model = None
    best_model_name = ""
    best_metrics = {"rmse": float("inf")}
    best_fi_df = pd.DataFrame()

    logger.info("Training %d model candidates for target '%s'...", len(models), target_name)

    for name, model in models.items():
        logger.info("-" * 50)
        logger.info("Training: %s", name.upper())

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics = evaluate_model(y_val, y_pred)
            all_results[name] = {
                "model": model,
                "metrics": metrics,
            }

            logger.info(
                "  %s → RMSE=%.2f, MAE=%.2f, R²=%.4f",
                name, metrics["rmse"], metrics["mae"], metrics["r2"],
            )

            # Feature importance
            fi_df = compute_feature_importance(model, feature_names, name)

            # Track best
            if metrics["rmse"] < best_metrics["rmse"]:
                best_model = model
                best_model_name = name
                best_metrics = metrics
                best_fi_df = fi_df

        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)
            continue

    if best_model is None:
        raise RuntimeError("All model candidates failed to train")

    logger.info("=" * 50)
    logger.info(
        "BEST MODEL: %s (RMSE=%.2f, MAE=%.2f, R²=%.4f)",
        best_model_name,
        best_metrics["rmse"],
        best_metrics["mae"],
        best_metrics["r2"],
    )
    logger.info("=" * 50)

    return best_model, best_model_name, best_metrics, all_results, best_fi_df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Main training pipeline entry point."""
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)
    df = None

    # Check if local data cache is available
    local_cache_path = "data/karachi_aqi_features.csv"
    if os.path.exists(local_cache_path):
        logger.info("Local data cache found! Loading '%s'...", local_cache_path)
        try:
            df = pd.read_csv(local_cache_path)
            logger.info("Successfully loaded %d rows from local cache", len(df))
        except Exception as e:
            logger.error("Failed to load local cache: %s. Will try MongoDB instead.", e)

    if df is None:
        logger.info("Step 1/4: Connecting to MongoDB feature store...")
        client = None
        try:
            client = get_mongo_client(prompt_if_missing=True)
            collection = get_feature_collection(client)
            logger.info("Step 2/4: Pulling training data from MongoDB...")
            df = get_training_data(collection)
            logger.info("Retrieved %d rows × %d columns", len(df), len(df.columns))
        except Exception as exc:
            logger.error("Failed to connect/pull from MongoDB: %s", exc)
            raise RuntimeError(
                "Unable to get training data from MongoDB and no valid local cache exists."
            )
        finally:
            if client is not None:
                client.close()

    # Step 2: Train models for each target horizon
    all_best_models = {}
    all_best_metrics = {}
    combined_fi = pd.DataFrame()

    for target_col in TARGET_COLS:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING FOR TARGET: %s", target_col)
        logger.info("=" * 60)

        try:
            X_train, X_val, y_train, y_val, feature_names = prepare_data(df, target_col)
            best_model, best_name, best_metrics, _all_results, fi_df = (
                train_and_evaluate_models(
                    X_train, X_val, y_train, y_val, feature_names, target_col,
                )
            )

            all_best_models[target_col] = {
                "model": best_model,
                "name": best_name,
                "feature_names": feature_names,
            }
            all_best_metrics[target_col] = best_metrics

            if not fi_df.empty:
                fi_df["target"] = target_col
                combined_fi = pd.concat([combined_fi, fi_df], ignore_index=True)

        except Exception as exc:
            logger.error("Failed to train for target %s: %s", target_col, exc)
            continue

    if not all_best_models:
        logger.error("No models were successfully trained — aborting")
        sys.exit(1)

    # Step 3: Save model artifacts
    logger.info("\nStep 3/4: Saving model artifacts...")

    model_dir = tempfile.mkdtemp(prefix="karachi_aqi_model_")
    local_models_dir = "models"
    os.makedirs(local_models_dir, exist_ok=True)

    for target_col, model_info in all_best_models.items():
        model_path = os.path.join(model_dir, f"model_{target_col}.joblib")
        joblib.dump(model_info["model"], model_path)
        
        local_model_path = os.path.join(local_models_dir, f"model_{target_col}.joblib")
        joblib.dump(model_info["model"], local_model_path)
        logger.info("Saved %s → %s and %s", target_col, model_path, local_model_path)

        # Save feature names for inference
        features_path = os.path.join(model_dir, f"features_{target_col}.joblib")
        joblib.dump(model_info["feature_names"], features_path)
        
        local_features_path = os.path.join(local_models_dir, f"features_{target_col}.joblib")
        joblib.dump(model_info["feature_names"], local_features_path)

    # Save feature importance
    if not combined_fi.empty:
        fi_path = os.path.join(model_dir, "feature_importance.csv")
        combined_fi.to_csv(fi_path, index=False)
        
        local_fi_path = os.path.join(local_models_dir, "feature_importance.csv")
        combined_fi.to_csv(local_fi_path, index=False)
        logger.info("Saved feature importance → %s and %s", fi_path, local_fi_path)

    # Save model metadata
    n_features = len(next(iter(all_best_models.values()))["feature_names"])
    metadata = {
        "models": {
            target: {
                "type": info["name"],
                "metrics": all_best_metrics[target],
            }
            for target, info in all_best_models.items()
        },
        "n_features": n_features,
        "training_rows": len(df),
    }
    metadata_path = os.path.join(model_dir, "metadata.joblib")
    joblib.dump(metadata, metadata_path)
    
    local_metadata_path = os.path.join(local_models_dir, "metadata.joblib")
    joblib.dump(metadata, local_metadata_path)
    logger.info("Saved metadata → %s and %s", metadata_path, local_metadata_path)

    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    for target_col, metrics in all_best_metrics.items():
        model_type = all_best_models[target_col]["name"]
        logger.info(
            "  %s: %s → RMSE=%.2f, MAE=%.2f, R²=%.4f",
            target_col, model_type,
            metrics["rmse"], metrics["mae"], metrics["r2"],
        )
    logger.info("Model artifacts updated in local '%s' directory", local_models_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)
