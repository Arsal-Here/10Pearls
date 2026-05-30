# 🌬️ Karachi AQI Prediction Service

A **100% serverless** Air Quality Index (AQI) prediction service for Karachi, Pakistan, forecasting AQI levels for the next 3 days using a Feature/Training/Inference pipeline architecture.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![MongoDB](https://img.shields.io/badge/Feature%20Store-MongoDB-47A248?style=flat-square)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions (CI/CD)                       │
│                                                                     │
│  ┌──────────────────────┐      ┌───────────────────────────────┐   │
│  │ Feature Pipeline      │      │ Training Pipeline              │   │
│  │ (Hourly Cron)         │      │ (Daily Cron)                   │   │
│  │                       │      │                                │   │
│  │ Open-Meteo API ──────►│      │ MongoDB FS ──► Train Models   │   │
│  │ Compute Features      │      │ Evaluate (RMSE/MAE/R²)        │   │
│  │ Upsert → MongoDB FS   │      │ Save → local model artifacts  │   │
│  └──────────────────────┘      └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MongoDB Feature Store                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Collection: karachi_aqi_features                             │   │
│  │ Unique key: timestamp (upserted hourly/backfill)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)                      │
│                                                                     │
│  • Current AQI with EPA color coding                                │
│  • 3-Day Forecast Cards                                             │
│  • Historical + Forecast Trend Chart                                │
│  • Pollutant Breakdown                                              │
│  • Health Alert Notifications                                       │
│  • Model Performance Metrics                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- A MongoDB deployment (Atlas or self-hosted)
- A GitHub repository (for automated pipelines)

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/karachi-aqi-service.git
cd karachi-aqi-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your credentials
# MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/<db>?retryWrites=true&w=majority

# MONGODB_DATABASE=karachi_aqi
# MONGODB_FEATURE_COLLECTION=karachi_aqi_features
```

If `MONGODB_URI` is not set, the CLI pipelines will prompt for it at runtime.

### 4. Run Historical Backfill

This populates the Feature Store with ~2 years of training data:

```bash
python feature_pipeline.py --backfill
```

> ⏳ This takes ~10-15 minutes (fetches data in monthly chunks).

### 5. Train the Model

```bash
python training_pipeline.py
```

This will:
- Pull data from the Feature Store
- Train XGBoost, Random Forest, and Ridge Regression
- Refresh local model artifacts in `models/`

### 6. Launch the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) to view.

---

## 🔧 GitHub Actions Setup

### Required Repository Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
|---|---|
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DATABASE` *(optional)* | Database name (default: `karachi_aqi`) |
| `MONGODB_FEATURE_COLLECTION` *(optional)* | Feature collection (default: `karachi_aqi_features`) |

### Automated Workflows

| Workflow | Schedule | Description |
|---|---|---|
| `feature-pipeline-hourly.yml` | Every hour | Fetches latest data, computes features, stores in MongoDB |
| `training-pipeline-daily.yml` | Daily at 2 AM UTC | Retrains models using MongoDB feature data and refreshes model artifacts |

Both workflows also support **manual triggering** via the "Run workflow" button in GitHub Actions.

---

## 📁 Project Structure

```
├── .env.example                    # Environment variables template
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration
│   ├── aqi_calculator.py           # EPA AQI formula implementation
│   ├── data_fetcher.py             # Open-Meteo API client
│   ├── feature_engineering.py      # Feature computation
│   └── mongodb_utils.py            # MongoDB feature-store helpers
│
├── feature_pipeline.py             # Data ingestion pipeline
├── training_pipeline.py            # Model training pipeline
├── app.py                          # Streamlit dashboard
│
├── .github/workflows/
│   ├── feature-pipeline-hourly.yml
│   └── training-pipeline-daily.yml
│
└── .streamlit/
    └── config.toml                 # Streamlit theme
```

---

## 🧪 Features Engineering

| Feature Category | Features |
|---|---|
| **Raw Pollutants** | PM2.5, PM10, NO2, SO2, O3, CO |
| **Weather** | Temperature, Humidity, Wind Speed, Wind Direction, Pressure |
| **Time** | Hour, Day of Week, Month, Is Weekend, Hour Sin/Cos, Month Sin/Cos |
| **Rolling** | 6h/12h/24h rolling mean & std for PM2.5, PM10, AQI |
| **Lag** | t-1, t-3, t-6, t-12, t-24 lag values |
| **Change Rate** | AQI change rate over 3h, 6h, 12h windows |
| **Targets** | AQI at t+24h, t+48h, t+72h |

---

## 🤖 Models

Three models are trained and compared:

| Model | Description |
|---|---|
| **XGBoost** | Gradient boosted trees — typically best for tabular data |
| **Random Forest** | Ensemble bagging baseline |
| **Ridge Regression** | Regularized linear baseline |

**Evaluation Metrics:** RMSE, MAE, R² Score

The best-performing models (lowest RMSE per target horizon) are saved as local artifacts in `models/`.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repo
4. Set the main file path to `app.py`
5. Ensure the `models/` artifacts are available in your deployment.

---

## 📊 Data Sources

- **Air Quality Data:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (CAMS)
- **Weather Data:** [Open-Meteo Weather API](https://open-meteo.com/en/docs) & [Historical Archive](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- **AQI Calculation:** [EPA AQI Technical Assistance Document](https://www.airnow.gov/aqi/aqi-basics/)

---

## 📝 License

This project is for educational and non-commercial use. Open-Meteo data is provided free for non-commercial purposes.
