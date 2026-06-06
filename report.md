# Karachi AQI Prediction Service - Project Report

**Repository:** [Arsal-Here/10Pearls](https://github.com/Arsal-Here/10Pearls)  
**Live dashboard:** [https://aqipredict10p.streamlit.app/](https://aqipredict10p.streamlit.app/)  
**Location:** Karachi, Pakistan

---

## What This Project Does

This is a serverless air quality forecasting system built for Karachi. It pulls in hourly pollutant and weather data, stores engineered features in MongoDB Atlas, retrains models on a daily schedule, saves those models to the cloud, and serves 24-hour, 48-hour, and 72-hour AQI predictions through a Streamlit dashboard.

There is no dedicated backend server running the whole time. GitHub Actions handles the scheduled data and training work, MongoDB Atlas holds the data and models, and Streamlit Cloud hosts the dashboard.

---

## Tech Stack and Resources

### Core stack

| Area | What we used |
|------|--------------|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, XGBoost, joblib |
| Database | MongoDB Atlas (pymongo, dnspython for SRV connections) |
| APIs | Open-Meteo (air quality, weather, historical archive) |
| Dashboard | Streamlit, Plotly |
| Automation | GitHub Actions |
| Deployment | Streamlit Community Cloud |
| Config | python-dotenv locally, GitHub Secrets + Streamlit Secrets in production |

### External data sources

- **Open-Meteo Air Quality API** for PM2.5, PM10, NO2, SO2, O3, and CO
- **Open-Meteo Weather API** for temperature, humidity, wind, and pressure
- **Open-Meteo Historical Archive** for the initial backfill
- **EPA AQI breakpoints** for calculating the Air Quality Index

### MongoDB Atlas setup

| Collection | Purpose |
|------------|---------|
| `karachi_aqi_features` | Feature store for training data |
| `karachi_aqi_models` | Trained model files, metadata, and feature importance |

Default database name: `karachi_aqi`

Cluster: `10pearlscluster.mongodb.net`

---

## How the System Works

### The big picture

```
Open-Meteo APIs
      |
      v
Feature Pipeline (hourly) ---> MongoDB: karachi_aqi_features
                                      |
                                      v
Training Pipeline (daily) ---> MongoDB: karachi_aqi_models
                         \--> local models/ folder (backup)
                                      |
                                      v
Streamlit Dashboard ---> loads models from cloud, falls back to local
```

### 1. Feature pipeline

**File:** `feature_pipeline.py`  
**Runs:** Every hour through GitHub Actions, or manually

What it does:

1. Fetches the last 5 days of air quality and weather data from Open-Meteo
2. Engineers features like rolling averages, lag values, time-based encodings, and pollutant-weather interactions
3. Calculates EPA AQI and forward-looking targets for 24h, 48h, and 72h horizons
4. Upserts everything into MongoDB using `timestamp` as the unique key

There is also a backfill mode (`python feature_pipeline.py --backfill`) that loads historical data from January 2024 in monthly chunks. That first run usually takes around 10 to 15 minutes.

### 2. Training pipeline

**File:** `training_pipeline.py`  
**Runs:** Daily at 2:00 AM UTC through GitHub Actions, or manually

What it does:

1. Pulls training data from the MongoDB feature store (or a local CSV cache if available)
2. Trains three model types for each forecast horizon:
   - XGBoost
   - Random Forest
   - Ridge Regression
3. Picks the best model per horizon based on RMSE
4. Saves artifacts locally in `models/`
5. Uploads all artifacts to the `karachi_aqi_models` collection in MongoDB

Files uploaded to the cloud:

- `model_aqi_target_24h.joblib`, `model_aqi_target_48h.joblib`, `model_aqi_target_72h.joblib`
- `features_aqi_target_24h.joblib`, `features_aqi_target_48h.joblib`, `features_aqi_target_72h.joblib`
- `metadata.joblib`
- `feature_importance.csv`

If the cloud upload fails because of a network issue, training still finishes. The error gets logged and local files remain available.

**Recent model performance:**

| Horizon | Best model | RMSE | MAE | R² |
|---------|------------|------|-----|-----|
| 24h | Ridge Regression | 24.86 | 18.57 | 0.33 |
| 48h | Ridge Regression | 26.46 | 20.29 | 0.24 |
| 72h | Ridge Regression | 27.03 | 20.85 | 0.21 |

A full local training run on ~21,000 rows takes around 20 minutes, mostly because of XGBoost and Random Forest.

### 3. Streamlit dashboard

**File:** `app.py`  
**Live URL:** [aqipredict10p.streamlit.app](https://aqipredict10p.streamlit.app/)

**Model loading:**

1. First tries MongoDB Atlas (`karachi_aqi_models`)
2. If that fails, falls back to the local `models/` folder
3. Shows a status badge in the sidebar: Cloud, Local Fallback, or Missing/Error

**What users see:**

- Current AQI with EPA color coding
- 3-day forecast cards
- Historical and forecast trend charts
- Pollutant breakdown
- Health alert banners for poor air quality
- Model performance metrics in the sidebar
- Feature importance charts
- An EDA tab for exploratory analysis

Live pollutant and weather data is fetched from Open-Meteo on each refresh (cached for 30 minutes).

### 4. Shared code (`src/`)

| Module | What it handles |
|--------|-----------------|
| `config.py` | API URLs, MongoDB settings, EPA breakpoints, environment variables |
| `mongodb_utils.py` | MongoDB connection, feature upserts, model upload/download |
| `data_fetcher.py` | Open-Meteo API client with retry logic |
| `feature_engineering.py` | Building the full feature matrix |
| `aqi_calculator.py` | EPA AQI formula |

---

## Workflows

### GitHub Actions

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `feature-pipeline-hourly.yml` | Every hour at minute 0 | Runs `feature_pipeline.py`, upserts features to MongoDB |
| `training-pipeline-daily.yml` | Daily at 2:00 AM UTC | Runs `training_pipeline.py`, trains models and uploads to MongoDB |

Both workflows:

- Run on `ubuntu-latest` with Python 3.11
- Install dependencies from `requirements.txt`
- Read `MONGODB_URI` from GitHub Secrets
- Can be triggered manually from the Actions tab

The feature workflow also supports a manual backfill option.

**Required GitHub Secrets:**

| Secret | Required? |
|--------|-----------|
| `MONGODB_URI` | Yes |
| `MONGODB_DATABASE` | Optional (defaults to `karachi_aqi`) |
| `MONGODB_FEATURE_COLLECTION` | Optional (defaults to `karachi_aqi_features`) |

The training pipeline also uploads to `karachi_aqi_models`, but that collection name has a sensible default in code and does not need its own secret unless you want to override it.

### Local development

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Set up .env with your MONGODB_URI

python feature_pipeline.py --backfill
python training_pipeline.py
.\venv\Scripts\streamlit run app.py
```

### Deployment

1. Push code to GitHub
2. Streamlit Cloud picks up the changes and redeploys
3. Set `MONGODB_URI` and related values in Streamlit Secrets (the `.env` file is gitignored and does not get deployed)
4. The dashboard loads models from MongoDB at runtime

---

## Problems Faced

### 1. Hopsworks jobs were not running reliably

The project originally used **Hopsworks** as the feature store, with `src/hopsworks_utils.py` handling reads and writes. In practice, the scheduled GitHub Actions jobs kept failing or not running properly against Hopsworks. We ran into issues like empty project name errors in CI, unreliable data access during training, and general friction getting the pipelines to complete end to end on the free tier setup.

As a temporary workaround, training data was saved to a local CSV (`data/karachi_aqi_features.csv`) so model training could still move forward.

**The fix:** We migrated the entire feature store to **MongoDB Atlas** in commit `2cf073d` ("Changed feature store to mongoDB"). This meant:

- Replacing `src/hopsworks_utils.py` with `src/mongodb_utils.py`
- Updating both GitHub Actions workflows to use MongoDB connection secrets
- Rewriting `feature_pipeline.py` and `training_pipeline.py` to upsert features and pull training data from MongoDB
- Removing the Hopsworks dependency from `requirements.txt`

MongoDB turned out to be a better fit for this project: simpler to configure in GitHub Actions, no extra managed ML platform dependency, and straightforward to extend later for cloud model storage too.

### 2. Moving model storage to MongoDB as well

After fixing the feature store, models were still only saved locally (and sometimes committed to the repo for Streamlit Cloud). That meant the deployed dashboard depended on files being present in the repository rather than always pulling the latest trained versions.

We extended the training pipeline to upload all 8 model artifacts to the `karachi_aqi_models` collection, and updated the Streamlit app to download and load them from MongoDB at runtime, with a local fallback if the database is unreachable.

### 3. Raw HTML showing in the sidebar

The MongoDB connection status section used nested HTML inside `st.markdown()`. On Streamlit, that rendered as literal HTML code on screen instead of a styled badge.

We replaced it with native Streamlit components (`st.success`, `st.warning`, `st.error`) so the status displays cleanly.

### 4. Streamlit Cloud import error on deploy

After deploying, the app crashed immediately with an `ImportError` in the `src.config` import chain. Streamlit Cloud redacts the full error message, but the root causes were:

- The project package is named `src`, which can conflict with Streamlit's `/mount/src/<repo>/` directory structure
- Absolute imports like `from src.config import ...` inside the `src` package are fragile in that environment
- The `.env` file does not exist on Streamlit Cloud, so credentials must come from Streamlit Secrets

**Fixes applied:**

- Added a `sys.path` bootstrap at the top of `app.py`
- Switched internal `src/` imports to relative imports (`from .config import ...`)
- Added `st.secrets` fallback in `config.py`
- Added `pymongo[srv]` and `dnspython` for Atlas connection strings
- Pinned Python 3.11 with a `.python-version` file

### 5. Git push rejected (diverged branches)

When pushing the deployment fixes, the remote already had a newer commit ("Added Dev Container Folder") that was not in the local branch. A regular push was rejected.

Resolved with `git pull --rebase origin main` followed by `git push`.

### 6. Documentation out of sync

The README and GitHub Actions workflow comments still describe saving models locally only. The code already uploads to and fetches from MongoDB, but the docs have not been fully updated to reflect that. This is a known gap, not a functional blocker.

### 7. Long training times

Training three model types across three horizons on ~21,000 rows takes about 20 minutes locally. The daily GitHub Actions workflow has a 60-minute timeout, which is enough headroom, but it is something to keep in mind if the dataset grows significantly.

---

## MongoDB Collections

### `karachi_aqi_features`

Stores engineered feature rows for model training. Each document is keyed by `timestamp`.

### `karachi_aqi_models`

Stores trained model artifacts as binary payloads. Each document looks roughly like this:

```json
{
  "filename": "model_aqi_target_24h.joblib",
  "content": "<binary data>",
  "updated_at": "2026-06-06T12:00:00Z"
}
```

---

## Configuration

### Local (`.env`)

```
MONGODB_URI=mongodb+srv://<user>:<password>@10pearlscluster.mongodb.net/...
MONGODB_DATABASE=karachi_aqi
MONGODB_FEATURE_COLLECTION=karachi_aqi_features
MONGODB_MODEL_COLLECTION=karachi_aqi_models
```

### GitHub Actions Secrets

Set these under **Settings > Secrets and variables > Actions**.

### Streamlit Cloud Secrets

Set the same variables in the Streamlit app settings. The dashboard will not work without `MONGODB_URI` configured here.

---

## Project Structure

```
10Pearls/
├── app.py                          # Streamlit dashboard
├── feature_pipeline.py             # Hourly data ingestion
├── training_pipeline.py            # Daily training + cloud upload
├── requirements.txt
├── .python-version
├── models/                         # Local model backup
├── data/                           # Local feature cache (optional)
├── src/
│   ├── config.py
│   ├── aqi_calculator.py
│   ├── data_fetcher.py
│   ├── feature_engineering.py
│   └── mongodb_utils.py
├── .github/workflows/
│   ├── feature-pipeline-hourly.yml
│   └── training-pipeline-daily.yml
└── .streamlit/config.toml
```

---

## Current Status

**Working:**

- Hourly feature ingestion into MongoDB
- Daily model training with cloud upload
- Streamlit dashboard with cloud-first model loading and local fallback
- Deployment fixes pushed to `main`
- Live site: [aqipredict10p.streamlit.app](https://aqipredict10p.streamlit.app/)

**Still worth doing:**

- Update the README architecture diagram to show the model collection
- Document `MONGODB_MODEL_COLLECTION` in the secrets table
- Add a CI step to confirm model upload succeeded after training
- Double-check Streamlit Secrets and MongoDB Atlas network access for the cloud deployment

---

*Report generated June 2026. Covers the project from initial Hopsworks setup through the MongoDB migration, cloud model storage, and Streamlit deployment.*
