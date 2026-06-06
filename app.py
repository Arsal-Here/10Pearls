"""
Streamlit Dashboard — Karachi AQI Prediction Service

Interactive web application that displays real-time and forecasted
Air Quality Index (AQI) for Karachi, Pakistan over the next 3 days.

Usage:
    streamlit run app.py
"""

import io
import logging
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.aqi_calculator import calculate_aqi, calculate_all_sub_indices
from src.config import (
    AQI_CATEGORIES,
    TARGET_COLS,
    get_aqi_category,
)
from src.data_fetcher import fetch_air_quality_recent, fetch_weather_recent
from src.feature_engineering import build_feature_dataframe
from src.mongodb_utils import (
    download_model_file,
    get_model_collection,
    get_mongo_client,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aqi_dashboard")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark glassmorphism design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f5d4, #00bbf9, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header .subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }

    /* AQI metric card */
    .aqi-card {
        text-align: center;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .aqi-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
    }
    .aqi-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .aqi-label {
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .aqi-category {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Forecast day card */
    .forecast-card {
        text-align: center;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
    }
    .forecast-card:hover {
        border-color: rgba(255,255,255,0.2);
    }
    .forecast-day {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .forecast-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0.5rem 0;
    }
    .forecast-category {
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        display: inline-block;
    }

    /* Alert banner */
    .alert-banner {
        background: linear-gradient(135deg, #ff0000aa, #7e0023cc);
        border: 1px solid #ff000066;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        animation: pulse-alert 2s ease-in-out infinite;
    }
    @keyframes pulse-alert {
        0%, 100% { box-shadow: 0 0 15px rgba(255,0,0,0.3); }
        50% { box-shadow: 0 0 30px rgba(255,0,0,0.6); }
    }
    .alert-banner h3 {
        color: #fff;
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    .alert-banner p {
        color: rgba(255,255,255,0.85);
        margin: 0;
        font-size: 0.95rem;
    }

    /* Pollutant bar */
    .pollutant-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .pollutant-name {
        font-weight: 600;
        width: 80px;
        color: rgba(255,255,255,0.8);
    }
    .pollutant-bar {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        margin: 0 1rem;
        overflow: hidden;
    }
    .pollutant-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .pollutant-value {
        font-weight: 500;
        min-width: 60px;
        text-align: right;
        color: rgba(255,255,255,0.6);
    }

    /* Sidebar styling */
    .sidebar-metric {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-metric .label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .sidebar-metric .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00f5d4;
    }

    /* Hide Streamlit footer branding only */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching & Data Loading
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=3600, show_spinner="Loading trained models from MongoDB Atlas...")
def load_models():
    """
    Load trained models and associated artifacts.
    First tries to retrieve from MongoDB Atlas Cloud, falling back to local files if offline.
    
    Returns:
        (models, metadata, fi_df, load_source)
    """
    # ── 1. Try Loading from MongoDB Atlas Cloud ──
    try:
        logger.info("Attempting to load models from MongoDB Atlas Cloud...")
        client = get_mongo_client(prompt_if_missing=False)
        model_collection = get_model_collection(client)
        
        models = {}
        # Load individual models
        for target_col in TARGET_COLS:
            model_bytes = download_model_file(model_collection, f"model_{target_col}.joblib")
            features_bytes = download_model_file(model_collection, f"features_{target_col}.joblib")
            
            if model_bytes is not None:
                models[target_col] = {
                    "model": joblib.load(io.BytesIO(model_bytes)),
                    "features": joblib.load(io.BytesIO(features_bytes)) if features_bytes is not None else None,
                }
                
        # Load metadata
        metadata_bytes = download_model_file(model_collection, "metadata.joblib")
        metadata = joblib.load(io.BytesIO(metadata_bytes)) if metadata_bytes is not None else {}
        
        # Load feature importance
        fi_bytes = download_model_file(model_collection, "feature_importance.csv")
        fi_df = pd.read_csv(io.BytesIO(fi_bytes)) if fi_bytes is not None else pd.DataFrame()
        
        client.close()
        
        if models:
            logger.info("Successfully loaded all models from MongoDB Atlas Cloud!")
            return models, metadata, fi_df, "cloud"
        else:
            raise FileNotFoundError("No models found in MongoDB Atlas cloud collection.")
            
    except Exception as exc:
        logger.warning("Failed to load models from MongoDB Atlas Cloud (%s). Falling back to local files...", exc)

    # ── 2. Fallback: Load from local file system ──
    model_dir = "models"
    try:
        models = {}
        for target_col in TARGET_COLS:
            model_path = os.path.join(model_dir, f"model_{target_col}.joblib")
            features_path = os.path.join(model_dir, f"features_{target_col}.joblib")

            if os.path.exists(model_path):
                models[target_col] = {
                    "model": joblib.load(model_path),
                    "features": joblib.load(features_path) if os.path.exists(features_path) else None,
                }

        metadata_path = os.path.join(model_dir, "metadata.joblib")
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

        fi_path = os.path.join(model_dir, "feature_importance.csv")
        fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

        if not models:
            raise FileNotFoundError("No trained models found in local models/ folder.")

        logger.info("Successfully loaded models from local fallback storage.")
        return models, metadata, fi_df, "local"

    except Exception as local_exc:
        logger.error("Failed to load local model artifacts: %s", local_exc)
        return None, None, pd.DataFrame(), "none"


@st.cache_data(ttl=1800, show_spinner="Fetching live data from Open-Meteo...")
def load_live_data():
    """Fetch recent air quality + weather data from Open-Meteo."""
    try:
        aq_df = fetch_air_quality_recent(past_days=7, forecast_days=0)
        weather_df = fetch_weather_recent(past_days=7, forecast_days=0)

        merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
        return merged
    except Exception as exc:
        logger.error("Failed to fetch live data: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------

def make_predictions(models: dict, live_df: pd.DataFrame) -> dict:
    """
    Run model inference to predict AQI for +24h, +48h, +72h.

    Returns:
        Dict mapping target name → predicted AQI value.
    """
    if not models or live_df.empty:
        return {}

    # Build features from live data
    feature_df = build_feature_dataframe(live_df)

    # Use the latest row for prediction
    latest = feature_df.iloc[[-1]].copy()

    predictions = {}
    for target_col, model_info in models.items():
        try:
            model = model_info["model"]
            feature_names = model_info.get("features")

            if feature_names:
                # Ensure all required features exist
                for col in feature_names:
                    if col not in latest.columns:
                        latest[col] = 0
                X = latest[feature_names].values
            else:
                exclude = {"timestamp", "aqi_target_24h", "aqi_target_48h", "aqi_target_72h"}
                cols = [c for c in latest.columns if c not in exclude]
                X = latest[cols].values

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            pred = model.predict(X)[0]
            predictions[target_col] = max(0, float(pred))  # AQI can't be negative

        except Exception as exc:
            logger.error("Prediction failed for %s: %s", target_col, exc)

    return predictions


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def create_aqi_trend_chart(
    historical_df: pd.DataFrame,
    predictions: dict,
) -> go.Figure:
    """Create a Plotly line chart with historical AQI + 3-day forecast."""
    fig = go.Figure()

    # Compute historical AQI
    if not historical_df.empty:
        from src.aqi_calculator import calculate_aqi as calc_aqi

        hist_aqi = []
        for _, row in historical_df.iterrows():
            aqi = calc_aqi(
                pm25=row.get("pm2_5"),
                pm10=row.get("pm10"),
                no2=row.get("no2"),
                so2=row.get("so2"),
                o3=row.get("o3"),
                co=row.get("co"),
            )
            hist_aqi.append(aqi)

        historical_df = historical_df.copy()
        historical_df["aqi"] = hist_aqi

        # Historical trace
        fig.add_trace(go.Scatter(
            x=historical_df["timestamp"],
            y=historical_df["aqi"],
            mode="lines",
            name="Historical AQI",
            line=dict(color="#00bbf9", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,187,249,0.08)",
        ))

    # Forecast points
    if predictions:
        last_time = historical_df["timestamp"].max() if not historical_df.empty else datetime.utcnow()
        forecast_times = [
            last_time + timedelta(hours=24),
            last_time + timedelta(hours=48),
            last_time + timedelta(hours=72),
        ]
        forecast_values = [
            predictions.get("aqi_target_24h", None),
            predictions.get("aqi_target_48h", None),
            predictions.get("aqi_target_72h", None),
        ]

        # Filter out None values
        valid = [(t, v) for t, v in zip(forecast_times, forecast_values) if v is not None]
        if valid:
            f_times, f_values = zip(*valid)

            # Connecting line from last historical to first forecast
            if not historical_df.empty and hist_aqi:
                connect_times = [last_time] + list(f_times)
                last_valid_aqi = [a for a in hist_aqi if a is not None]
                connect_values = [last_valid_aqi[-1] if last_valid_aqi else 0] + list(f_values)

                fig.add_trace(go.Scatter(
                    x=connect_times,
                    y=connect_values,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="#9b5de5", width=3, dash="dash"),
                ))

            # Forecast markers with color coding
            colors = [get_aqi_category(v)["color"] for v in f_values]
            fig.add_trace(go.Scatter(
                x=list(f_times),
                y=list(f_values),
                mode="markers+text",
                name="Predicted AQI",
                marker=dict(
                    size=18,
                    color=colors,
                    line=dict(width=2, color="white"),
                    symbol="diamond",
                ),
                text=[f"AQI: {int(v)}" for v in f_values],
                textposition="top center",
                textfont=dict(size=12, color="white"),
            ))

    # AQI threshold lines
    thresholds = [
        (50, "Good", "#00E400"),
        (100, "Moderate", "#FFFF00"),
        (150, "USG", "#FF7E00"),
        (200, "Unhealthy", "#FF0000"),
        (300, "Very Unhealthy", "#8F3F97"),
    ]
    for value, label, color in thresholds:
        fig.add_hline(
            y=value,
            line_dash="dot",
            line_color=color,
            opacity=0.3,
            annotation_text=label,
            annotation_position="left",
            annotation_font_size=10,
            annotation_font_color=color,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        xaxis=dict(
            title="",
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    return fig


def create_pollutant_chart(sub_indices: dict) -> go.Figure:
    """Create a horizontal bar chart of individual pollutant AQI contributions."""
    pollutants = list(sub_indices.keys())
    values = [sub_indices[p] if sub_indices[p] is not None else 0 for p in pollutants]

    colors = []
    for v in values:
        cat = get_aqi_category(v)
        colors.append(cat["color"])

    fig = go.Figure(go.Bar(
        x=values,
        y=pollutants,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{int(v)}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color="white"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=20, b=10),
        height=280,
        xaxis=dict(
            title="Sub-AQI",
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
        ),
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# EDA Data Loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading historical dataset for EDA...")
def load_eda_data():
    """Load the full historical dataset for EDA from local CSV cache."""
    csv_path = "data/karachi_aqi_features.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert timestamp from epoch ms back to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Dashboard Tab
# ---------------------------------------------------------------------------

def render_dashboard(live_df, models, metadata, fi_df):
    """Render the main forecast dashboard tab."""

    # ── Current AQI computation ──
    latest_row = live_df.iloc[-1]
    current_aqi = calculate_aqi(
        pm25=latest_row.get("pm2_5"),
        pm10=latest_row.get("pm10"),
        no2=latest_row.get("no2"),
        so2=latest_row.get("so2"),
        o3=latest_row.get("o3"),
        co=latest_row.get("co"),
    )
    current_aqi = current_aqi if current_aqi is not None else 0
    current_category = get_aqi_category(current_aqi)

    sub_indices = calculate_all_sub_indices(
        pm25=latest_row.get("pm2_5"),
        pm10=latest_row.get("pm10"),
        no2=latest_row.get("no2"),
        so2=latest_row.get("so2"),
        o3=latest_row.get("o3"),
        co=latest_row.get("co"),
    )

    # ── Predictions ──
    predictions = {}
    if models:
        predictions = make_predictions(models, live_df)

    # ── Alert Banner ──
    alert_days = []
    for target, label in [("aqi_target_24h", "Day 1"), ("aqi_target_48h", "Day 2"), ("aqi_target_72h", "Day 3")]:
        pred_val = predictions.get(target)
        if pred_val and pred_val > 150:
            cat = get_aqi_category(pred_val)
            alert_days.append((label, int(pred_val), cat["label"]))

    if alert_days:
        alert_lines = "".join(
            f"<br>• <strong>{d[0]}</strong>: AQI {d[1]} ({d[2]})" for d in alert_days
        )
        st.markdown(f"""
        <div class="alert-banner">
            <h3>⚠️ Air Quality Alert</h3>
            <p>Forecasted AQI levels are expected to reach <strong>unhealthy or worse</strong> ranges:
            {alert_lines}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Current AQI + 3-Day Forecast Row ──
    col_current, col_d1, col_d2, col_d3 = st.columns([1.5, 1, 1, 1])

    with col_current:
        st.markdown(f"""
        <div class="aqi-card glass-card" style="background: linear-gradient(135deg, {current_category['color']}22, {current_category['color']}08);">
            <div class="aqi-label" style="color: {current_category['color']};">Current AQI</div>
            <div class="aqi-value" style="color: {current_category['color']};">{int(current_aqi)}</div>
            <div class="aqi-category" style="background: {current_category['color']}33; color: {current_category['color']};">
                {current_category['emoji']} {current_category['label']}
            </div>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.8rem;">
                {current_category['message']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Forecast cards
    forecast_targets = [
        ("aqi_target_24h", "Day 1", "Tomorrow"),
        ("aqi_target_48h", "Day 2", "In 2 days"),
        ("aqi_target_72h", "Day 3", "In 3 days"),
    ]
    forecast_cols = [col_d1, col_d2, col_d3]

    for col, (target, day_label, day_desc) in zip(forecast_cols, forecast_targets):
        pred_val = predictions.get(target)
        with col:
            if pred_val is not None:
                cat = get_aqi_category(pred_val)
                st.markdown(f"""
                <div class="forecast-card glass-card" style="background: linear-gradient(180deg, {cat['color']}15, transparent);">
                    <div class="forecast-day">{day_desc}</div>
                    <div class="forecast-value" style="color: {cat['color']};">{int(pred_val)}</div>
                    <div class="forecast-category" style="background: {cat['color']}33; color: {cat['color']};">
                        {cat['emoji']} {cat['label']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="forecast-card glass-card">
                    <div class="forecast-day">{day_desc}</div>
                    <div class="forecast-value" style="color: rgba(255,255,255,0.3);">—</div>
                    <div class="forecast-category" style="background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.4);">
                        No Model Loaded
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AQI Trend Chart + Pollutant Breakdown ──
    chart_col, pollutant_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📈 AQI Trend — Historical & Forecast")
        fig = create_aqi_trend_chart(live_df, predictions)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with pollutant_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🧪 Pollutant Breakdown (Current)")
        fig_pollutant = create_pollutant_chart(sub_indices)
        st.plotly_chart(fig_pollutant, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Weather Conditions Row ──
    st.markdown("##### 🌡️ Current Weather Conditions")
    w1, w2, w3, w4, w5 = st.columns(5)

    weather_items = [
        (w1, "🌡️ Temperature", f"{latest_row.get('temperature', 'N/A'):.1f}°C" if pd.notna(latest_row.get('temperature')) else "N/A"),
        (w2, "💧 Humidity", f"{latest_row.get('humidity', 'N/A'):.0f}%" if pd.notna(latest_row.get('humidity')) else "N/A"),
        (w3, "💨 Wind Speed", f"{latest_row.get('wind_speed', 'N/A'):.1f} km/h" if pd.notna(latest_row.get('wind_speed')) else "N/A"),
        (w4, "🧭 Wind Dir", f"{latest_row.get('wind_direction', 'N/A'):.0f}°" if pd.notna(latest_row.get('wind_direction')) else "N/A"),
        (w5, "📊 Pressure", f"{latest_row.get('pressure', 'N/A'):.0f} hPa" if pd.notna(latest_row.get('pressure')) else "N/A"),
    ]

    for col, label, value in weather_items:
        with col:
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── AQI Legend ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📋 EPA AQI Scale")
    legend_cols = st.columns(6)
    for col, cat in zip(legend_cols, AQI_CATEGORIES):
        with col:
            low, high = cat["range"]
            st.markdown(f"""
            <div style="
                background: {cat['color']}22;
                border-left: 4px solid {cat['color']};
                border-radius: 8px;
                padding: 0.6rem 0.8rem;
                font-size: 0.8rem;
            ">
                <strong style="color: {cat['color']};">{cat['emoji']} {cat['label']}</strong><br>
                <span style="color: rgba(255,255,255,0.5);">{low}–{high}</span>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# EDA Tab
# ---------------------------------------------------------------------------

def render_eda():
    """Render the Exploratory Data Analysis tab."""

    st.markdown("""
    <div class="glass-card" style="margin-bottom: 1.5rem;">
        <h3 style="color: #00f5d4; margin: 0;">📊 Exploratory Data Analysis</h3>
        <p style="color: rgba(255,255,255,0.5); margin-top: 0.3rem;">
            Interactive exploration of the Karachi AQI historical dataset — distributions,
            correlations, seasonal patterns, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    eda_df = load_eda_data()

    if eda_df.empty:
        st.warning(
            "⚠️ No historical dataset found at `data/karachi_aqi_features.csv`. "
            "Run the backfill pipeline first: `python feature_pipeline.py --backfill`"
        )
        return

    # ── EDA Sub-sections ──
    eda_sections = st.tabs([
        "📋 Overview",
        "📊 Distributions",
        "🔗 Correlations",
        "📈 Time Series",
        "🏷️ AQI Categories",
        "🌡️ Seasonal Patterns",
        "🧪 Pollutant Contributions",
        "🔍 Outlier Detection",
    ])

    # Columns for analysis
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
    weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure"]
    available_pollutants = [c for c in pollutant_cols if c in eda_df.columns]
    available_weather = [c for c in weather_cols if c in eda_df.columns]

    # ── 1. Dataset Overview ──
    with eda_sections[0]:
        st.markdown("#### Dataset Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{len(eda_df):,}")
        m2.metric("Total Columns", f"{len(eda_df.columns)}")

        if "timestamp" in eda_df.columns and eda_df["timestamp"].notna().any():
            min_date = eda_df["timestamp"].min()
            max_date = eda_df["timestamp"].max()
            m3.metric("Date Range Start", min_date.strftime("%Y-%m-%d") if hasattr(min_date, "strftime") else str(min_date)[:10])
            m4.metric("Date Range End", max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)[:10])

        # Missing values heatmap
        st.markdown("##### Missing Values")
        core_cols = available_pollutants + available_weather + (["aqi"] if "aqi" in eda_df.columns else [])
        if core_cols:
            missing_pct = (eda_df[core_cols].isnull().sum() / len(eda_df) * 100).round(2)
            missing_df = pd.DataFrame({"Column": missing_pct.index, "Missing %": missing_pct.values})

            fig_missing = go.Figure(go.Bar(
                x=missing_df["Column"],
                y=missing_df["Missing %"],
                marker=dict(
                    color=["#ff6b6b" if v > 5 else "#00f5d4" for v in missing_df["Missing %"]],
                ),
                text=[f"{v:.1f}%" for v in missing_df["Missing %"]],
                textposition="outside",
                textfont=dict(color="white", size=11),
            ))
            fig_missing.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
                xaxis=dict(title=""),
                yaxis=dict(title="Missing %", gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False,
            )
            st.plotly_chart(fig_missing, use_container_width=True, config={"displayModeBar": False})

        # Descriptive statistics
        st.markdown("##### Descriptive Statistics")
        stats_cols = available_pollutants + available_weather + (["aqi"] if "aqi" in eda_df.columns else [])
        if stats_cols:
            st.dataframe(
                eda_df[stats_cols].describe().round(2).T.style.format("{:.2f}"),
                use_container_width=True,
            )

    # ── 2. Distributions ──
    with eda_sections[1]:
        st.markdown("#### Pollutant & Weather Distributions")

        dist_col_select = st.multiselect(
            "Select variables to plot",
            options=available_pollutants + available_weather + (["aqi"] if "aqi" in eda_df.columns else []),
            default=available_pollutants[:4] + (["aqi"] if "aqi" in eda_df.columns else []),
            key="eda_dist_select",
        )

        if dist_col_select:
            n_cols = min(3, len(dist_col_select))
            from plotly.subplots import make_subplots
            n_rows = (len(dist_col_select) + n_cols - 1) // n_cols

            fig_dist = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=dist_col_select,
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            colors = ["#00f5d4", "#00bbf9", "#9b5de5", "#f15bb5", "#fee440", "#ff6b6b"]

            for i, col_name in enumerate(dist_col_select):
                row = i // n_cols + 1
                col = i % n_cols + 1
                data = eda_df[col_name].dropna()

                fig_dist.add_trace(
                    go.Histogram(
                        x=data,
                        nbinsx=50,
                        marker=dict(color=colors[i % len(colors)], opacity=0.7),
                        name=col_name,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )

            fig_dist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=300 * n_rows,
                showlegend=False,
            )
            fig_dist.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_dist.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    # ── 3. Correlations ──
    with eda_sections[2]:
        st.markdown("#### Correlation Matrix")

        corr_cols = available_pollutants + available_weather + (["aqi"] if "aqi" in eda_df.columns else [])
        if len(corr_cols) >= 2:
            corr_matrix = eda_df[corr_cols].corr().round(3)

            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[
                    [0.0, "#7e0023"],
                    [0.25, "#ff6b6b"],
                    [0.5, "#1a1f2e"],
                    [0.75, "#00bbf9"],
                    [1.0, "#00f5d4"],
                ],
                zmin=-1, zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=10, color="white"),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            ))

            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=500,
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

            # Top correlations with AQI
            if "aqi" in corr_cols:
                st.markdown("##### Strongest Correlations with AQI")
                aqi_corrs = corr_matrix["aqi"].drop("aqi").abs().sort_values(ascending=False).head(10)
                aqi_corrs_df = pd.DataFrame({
                    "Feature": aqi_corrs.index,
                    "Abs Correlation": aqi_corrs.values,
                    "Direction": [
                        "Positive ↑" if corr_matrix["aqi"][f] > 0 else "Negative ↓"
                        for f in aqi_corrs.index
                    ],
                })

                fig_aqi_corr = go.Figure(go.Bar(
                    x=aqi_corrs_df["Abs Correlation"],
                    y=aqi_corrs_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=[
                            "#00f5d4" if corr_matrix["aqi"][f] > 0 else "#ff6b6b"
                            for f in aqi_corrs_df["Feature"]
                        ],
                    ),
                    text=[f"{v:.3f}" for v in aqi_corrs_df["Abs Correlation"]],
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                ))
                fig_aqi_corr.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=60, t=20, b=20),
                    height=350,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(title="|Correlation|", gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_aqi_corr, use_container_width=True, config={"displayModeBar": False})

    # ── 4. Time Series ──
    with eda_sections[3]:
        st.markdown("#### Time Series Trends")

        if "timestamp" in eda_df.columns:
            ts_vars = st.multiselect(
                "Select variables to plot over time",
                options=available_pollutants + (["aqi"] if "aqi" in eda_df.columns else []) + available_weather,
                default=["aqi"] if "aqi" in eda_df.columns else available_pollutants[:2],
                key="eda_ts_select",
            )

            agg_option = st.radio(
                "Aggregation", ["Hourly (raw)", "Daily Mean", "Weekly Mean", "Monthly Mean"],
                horizontal=True, key="eda_ts_agg",
            )

            if ts_vars:
                ts_data = eda_df[["timestamp"] + ts_vars].copy()
                ts_data = ts_data.dropna(subset=["timestamp"])
                ts_data = ts_data.sort_values("timestamp")

                if agg_option == "Daily Mean":
                    ts_data = ts_data.set_index("timestamp").resample("D").mean().reset_index()
                elif agg_option == "Weekly Mean":
                    ts_data = ts_data.set_index("timestamp").resample("W").mean().reset_index()
                elif agg_option == "Monthly Mean":
                    ts_data = ts_data.set_index("timestamp").resample("ME").mean().reset_index()

                colors_ts = ["#00f5d4", "#00bbf9", "#9b5de5", "#f15bb5", "#fee440", "#ff6b6b"]
                fig_ts = go.Figure()

                for i, var in enumerate(ts_vars):
                    fig_ts.add_trace(go.Scatter(
                        x=ts_data["timestamp"],
                        y=ts_data[var],
                        mode="lines",
                        name=var,
                        line=dict(color=colors_ts[i % len(colors_ts)], width=1.5),
                    ))

                fig_ts.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=450,
                    xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Value", gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    # ── 5. AQI Category Distribution ──
    with eda_sections[4]:
        st.markdown("#### AQI Category Distribution")

        if "aqi" in eda_df.columns:
            aqi_data = eda_df["aqi"].dropna()

            # Assign categories
            cat_labels = []
            cat_colors_list = []
            for val in aqi_data:
                cat = get_aqi_category(val)
                cat_labels.append(cat["label"])
                cat_colors_list.append(cat["color"])

            cat_counts = pd.Series(cat_labels).value_counts()

            # Pie + Bar side by side
            pie_col, bar_col = st.columns(2)

            with pie_col:
                cat_color_map = {cat["label"]: cat["color"] for cat in AQI_CATEGORIES}
                fig_pie = go.Figure(go.Pie(
                    labels=cat_counts.index,
                    values=cat_counts.values,
                    marker=dict(colors=[cat_color_map.get(c, "#666") for c in cat_counts.index]),
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
                ))
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400,
                    showlegend=False,
                    title=dict(text="Category Breakdown", font=dict(size=14, color="rgba(255,255,255,0.7)")),
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with bar_col:
                fig_bar_cat = go.Figure(go.Bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    marker=dict(
                        color=[cat_color_map.get(c, "#666") for c in cat_counts.index],
                    ),
                    text=[f"{v:,}" for v in cat_counts.values],
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                ))
                fig_bar_cat.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400,
                    xaxis=dict(title=""),
                    yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False,
                    title=dict(text="Category Counts", font=dict(size=14, color="rgba(255,255,255,0.7)")),
                )
                st.plotly_chart(fig_bar_cat, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("AQI column not found in dataset.")

    # ── 6. Seasonal Patterns ──
    with eda_sections[5]:
        st.markdown("#### Seasonal & Temporal Patterns")

        if "aqi" in eda_df.columns:
            season_col1, season_col2 = st.columns(2)

            # AQI by Hour of Day
            with season_col1:
                if "hour" in eda_df.columns:
                    hourly_stats = eda_df.groupby("hour")["aqi"].agg(["mean", "median", "std"]).reset_index()

                    fig_hour = go.Figure()
                    fig_hour.add_trace(go.Scatter(
                        x=hourly_stats["hour"],
                        y=hourly_stats["mean"],
                        mode="lines+markers",
                        name="Mean AQI",
                        line=dict(color="#00f5d4", width=2.5),
                        marker=dict(size=8),
                    ))
                    fig_hour.add_trace(go.Scatter(
                        x=hourly_stats["hour"],
                        y=hourly_stats["median"],
                        mode="lines+markers",
                        name="Median AQI",
                        line=dict(color="#00bbf9", width=2, dash="dash"),
                        marker=dict(size=6),
                    ))
                    # Std deviation band
                    fig_hour.add_trace(go.Scatter(
                        x=list(hourly_stats["hour"]) + list(hourly_stats["hour"][::-1]),
                        y=list(hourly_stats["mean"] + hourly_stats["std"]) + list((hourly_stats["mean"] - hourly_stats["std"])[::-1]),
                        fill="toself",
                        fillcolor="rgba(0,245,212,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                    fig_hour.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=380,
                        title=dict(text="AQI by Hour of Day", font=dict(size=14, color="rgba(255,255,255,0.7)")),
                        xaxis=dict(title="Hour", dtick=2, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    )
                    st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})

            # AQI by Month
            with season_col2:
                if "month" in eda_df.columns:
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    monthly_stats = eda_df.groupby("month")["aqi"].agg(["mean", "median", "std"]).reset_index()

                    fig_month = go.Figure()
                    fig_month.add_trace(go.Bar(
                        x=[month_names[int(m)-1] if 1 <= m <= 12 else str(m) for m in monthly_stats["month"]],
                        y=monthly_stats["mean"],
                        marker=dict(
                            color=monthly_stats["mean"],
                            colorscale=[[0, "#00f5d4"], [0.5, "#fee440"], [1, "#ff6b6b"]],
                        ),
                        text=[f"{v:.0f}" for v in monthly_stats["mean"]],
                        textposition="outside",
                        textfont=dict(color="white", size=10),
                        error_y=dict(
                            type="data",
                            array=monthly_stats["std"].fillna(0).values,
                            visible=True,
                            color="rgba(255,255,255,0.3)",
                        ),
                    ))

                    fig_month.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=380,
                        title=dict(text="Mean AQI by Month", font=dict(size=14, color="rgba(255,255,255,0.7)")),
                        xaxis=dict(title=""),
                        yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})

            # AQI by Day of Week
            if "day_of_week" in eda_df.columns:
                st.markdown("##### AQI by Day of Week")
                dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                dow_data = eda_df.groupby("day_of_week")["aqi"].agg(["mean", "std"]).reset_index()

                fig_dow = go.Figure(go.Bar(
                    x=[dow_names[int(d)] if 0 <= d <= 6 else str(d) for d in dow_data["day_of_week"]],
                    y=dow_data["mean"],
                    marker=dict(
                        color=["#9b5de5" if d >= 5 else "#00bbf9" for d in dow_data["day_of_week"]],
                    ),
                    text=[f"{v:.0f}" for v in dow_data["mean"]],
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                    error_y=dict(
                        type="data",
                        array=dow_data["std"].fillna(0).values,
                        visible=True,
                        color="rgba(255,255,255,0.3)",
                    ),
                ))
                fig_dow.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=320,
                    xaxis=dict(title=""),
                    yaxis=dict(title="Mean AQI", gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})

    # ── 7. Pollutant Contributions ──
    with eda_sections[6]:
        st.markdown("#### Pollutant Concentration Trends")

        if "timestamp" in eda_df.columns and available_pollutants:
            # Stacked area chart of pollutant concentrations over time (daily mean)
            poll_ts = eda_df[["timestamp"] + available_pollutants].copy()
            poll_ts = poll_ts.dropna(subset=["timestamp"])
            poll_ts = poll_ts.set_index("timestamp").resample("D").mean().reset_index()

            colors_poll = ["#00f5d4", "#00bbf9", "#9b5de5", "#f15bb5", "#fee440", "#ff6b6b"]
            # rgba fill equivalents (hex "40" suffix ≈ 0.25 opacity) — rgba() required for older Plotly
            colors_poll_fill = [
                "rgba(0,245,212,0.25)",
                "rgba(0,187,249,0.25)",
                "rgba(155,93,229,0.25)",
                "rgba(241,91,181,0.25)",
                "rgba(254,228,64,0.25)",
                "rgba(255,107,107,0.25)",
            ]
            fig_stack = go.Figure()

            for i, pol in enumerate(available_pollutants):
                fig_stack.add_trace(go.Scatter(
                    x=poll_ts["timestamp"],
                    y=poll_ts[pol],
                    mode="lines",
                    name=pol.upper(),
                    line=dict(width=0.5, color=colors_poll[i % len(colors_poll)]),
                    stackgroup="one",
                    fillcolor=colors_poll_fill[i % len(colors_poll_fill)],
                ))

            fig_stack.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=450,
                xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Concentration (µg/m³)", gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                title=dict(text="Daily Mean Pollutant Concentrations (Stacked)", font=dict(size=14, color="rgba(255,255,255,0.7)")),
            )
            st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

            # Individual pollutant box plots
            st.markdown("##### Pollutant Concentration Box Plots")
            from plotly.subplots import make_subplots

            fig_box_poll = make_subplots(
                rows=1, cols=len(available_pollutants),
                subplot_titles=[p.upper() for p in available_pollutants],
                horizontal_spacing=0.05,
            )

            for i, pol in enumerate(available_pollutants):
                fig_box_poll.add_trace(
                    go.Box(
                        y=eda_df[pol].dropna(),
                        marker=dict(color=colors_poll[i % len(colors_poll)]),
                        boxmean="sd",
                        name=pol.upper(),
                        showlegend=False,
                    ),
                    row=1, col=i + 1,
                )

            fig_box_poll.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
            )
            fig_box_poll.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_box_poll, use_container_width=True, config={"displayModeBar": False})

    # ── 8. Outlier Detection ──
    with eda_sections[7]:
        st.markdown("#### Outlier Detection (IQR Method)")

        outlier_var = st.selectbox(
            "Select variable for outlier analysis",
            options=available_pollutants + (["aqi"] if "aqi" in eda_df.columns else []) + available_weather,
            key="eda_outlier_select",
        )

        if outlier_var:
            data = eda_df[outlier_var].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]

            # Metrics
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Total Values", f"{len(data):,}")
            o2.metric("Outliers Found", f"{len(outliers):,}")
            o3.metric("Outlier %", f"{len(outliers)/len(data)*100:.2f}%")
            o4.metric("IQR", f"{IQR:.2f}")

            # Box plot + scatter of outliers
            out_col1, out_col2 = st.columns([1, 2])

            with out_col1:
                fig_box = go.Figure(go.Box(
                    y=data,
                    marker=dict(color="#00f5d4"),
                    boxmean="sd",
                    name=outlier_var,
                ))
                fig_box.add_hline(y=upper_bound, line_dash="dash", line_color="#ff6b6b", opacity=0.6,
                                  annotation_text=f"Upper: {upper_bound:.1f}", annotation_font_color="#ff6b6b")
                fig_box.add_hline(y=lower_bound, line_dash="dash", line_color="#ff6b6b", opacity=0.6,
                                  annotation_text=f"Lower: {lower_bound:.1f}", annotation_font_color="#ff6b6b")
                fig_box.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400,
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

            with out_col2:
                if "timestamp" in eda_df.columns:
                    outlier_df = eda_df[["timestamp", outlier_var]].dropna()
                    is_outlier = (outlier_df[outlier_var] < lower_bound) | (outlier_df[outlier_var] > upper_bound)

                    fig_scatter = go.Figure()
                    # Normal points
                    normal = outlier_df[~is_outlier]
                    fig_scatter.add_trace(go.Scatter(
                        x=normal["timestamp"],
                        y=normal[outlier_var],
                        mode="markers",
                        name="Normal",
                        marker=dict(color="rgba(0,187,249,0.3)", size=3),
                    ))
                    # Outlier points
                    outlier_pts = outlier_df[is_outlier]
                    fig_scatter.add_trace(go.Scatter(
                        x=outlier_pts["timestamp"],
                        y=outlier_pts[outlier_var],
                        mode="markers",
                        name="Outlier",
                        marker=dict(color="#ff6b6b", size=5, symbol="x"),
                    ))
                    fig_scatter.add_hline(y=upper_bound, line_dash="dot", line_color="#ff6b6b", opacity=0.4)
                    fig_scatter.add_hline(y=lower_bound, line_dash="dot", line_color="#ff6b6b", opacity=0.4)

                    fig_scatter.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=400,
                        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title=outlier_var, gridcolor="rgba(255,255,255,0.05)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Sidebar (shared across tabs)
# ---------------------------------------------------------------------------

def render_sidebar(metadata, fi_df, load_source: str):
    """Render the sidebar with model info, cloud status, and controls."""
    with st.sidebar:
        # ── Cloud Storage Indicator ──
        st.markdown("### ☁️ Cloud Storage")
        st.caption("MongoDB Atlas")

        if load_source == "cloud":
            st.success("Models: ☁️ Cloud (MongoDB Atlas)")
        elif load_source == "local":
            st.warning("Models: 📂 Local Fallback")
        else:
            st.error("Models: Missing / Error")

        st.markdown("## 🤖 Model Information")

        if metadata:
            model_info = metadata.get("models", {})
            for target, info in model_info.items():
                horizon = target.replace("aqi_target_", "").replace("h", "H")
                st.markdown(f"""
                <div class="sidebar-metric">
                    <div class="label">⏱️ {horizon} Model</div>
                    <div class="value">{info.get('type', 'unknown').title()}</div>
                </div>
                """, unsafe_allow_html=True)

                metrics = info.get("metrics", {})
                if metrics:
                    m1, m2 = st.columns(2)
                    m1.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    m2.metric("R²", f"{metrics.get('r2', 0):.4f}")

            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="label">📊 Training Rows</div>
                <div class="value">{metadata.get('training_rows', 'N/A'):,}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="label">🔢 Features Used</div>
                <div class="value">{metadata.get('n_features', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("No model loaded. Run the training pipeline first.")

        # Feature Importance
        if not fi_df.empty:
            st.markdown("---")
            st.markdown("## 📊 Top Features")
            target_filter = st.selectbox(
                "Target Horizon",
                options=fi_df["target"].unique() if "target" in fi_df.columns else [],
            )
            filtered_fi = fi_df[fi_df["target"] == target_filter].head(10) if "target" in fi_df.columns else fi_df.head(10)

            if not filtered_fi.empty:
                fig_fi = go.Figure(go.Bar(
                    x=filtered_fi["importance"],
                    y=filtered_fi["feature"],
                    orientation="h",
                    marker=dict(
                        color="rgba(0,245,212,0.7)",
                    ),
                ))
                fig_fi.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        # Last updated
        st.markdown(f"""
        <div style="text-align: center; color: rgba(255,255,255,0.3); font-size: 0.75rem; margin-top: 1rem;">
            Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------

def main():
    """Render the Streamlit dashboard with tabbed navigation."""

    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌬️ Karachi AQI Forecast</h1>
        <p class="subtitle">Real-time Air Quality Monitoring & 3-Day Prediction for Karachi, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    with st.spinner("Loading latest data..."):
        live_df = load_live_data()
        models, metadata, fi_df, load_source = load_models()

    # ── Sidebar (always visible) ──
    render_sidebar(metadata, fi_df, load_source)

    if live_df.empty:
        st.error("⚠️ Unable to fetch live data from Open-Meteo. Please try again later.")
        return

    # ── Tabbed Navigation ──
    tab_dashboard, tab_eda = st.tabs(["🏠 Dashboard", "📊 Exploratory Data Analysis"])

    with tab_dashboard:
        render_dashboard(live_df, models, metadata, fi_df)

    with tab_eda:
        render_eda()


if __name__ == "__main__":
    main()

