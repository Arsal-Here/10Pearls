"""
Streamlit Dashboard — Karachi AQI Prediction Service

Interactive web application that displays real-time and forecasted
Air Quality Index (AQI) for Karachi, Pakistan over the next 3 days.

Usage:
    streamlit run app.py
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.aqi_calculator import calculate_aqi, calculate_all_sub_indices
from src.config import (
    AQI_CATEGORIES,
    CITY_NAME,
    MODEL_NAME,
    TARGET_COLS,
    get_aqi_category,
)
from src.data_fetcher import fetch_air_quality_recent, fetch_weather_recent
from src.feature_engineering import build_feature_dataframe, get_feature_columns

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

@st.cache_resource(ttl=3600, show_spinner="Loading trained models...")
def load_model_from_registry():
    """Load the trained model from Hopsworks Model Registry, with local fallback."""
    model_dir = "models"
    metrics = {}
    
    try:
        from src.hopsworks_utils import (
            get_hopsworks_project,
            get_model_registry,
            get_latest_model,
            download_model,
        )

        project = get_hopsworks_project()
        mr = get_model_registry(project)
        model = get_latest_model(mr)
        model_dir = download_model(model)
        metrics = model.training_metrics
        logger.info("Successfully loaded models from Hopsworks registry.")
    except Exception as exc:
        logger.warning("Failed to load model from Hopsworks: %s. Falling back to local models/ folder...", exc)

    try:
        # Load individual models for each target
        models = {}
        for target_col in TARGET_COLS:
            model_path = os.path.join(model_dir, f"model_{target_col}.joblib")
            features_path = os.path.join(model_dir, f"features_{target_col}.joblib")

            if os.path.exists(model_path):
                models[target_col] = {
                    "model": joblib.load(model_path),
                    "features": joblib.load(features_path) if os.path.exists(features_path) else None,
                }

        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.joblib")
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

        # Load feature importance
        fi_path = os.path.join(model_dir, "feature_importance.csv")
        fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

        if not models:
            raise FileNotFoundError("No trained models found in either Hopsworks registry or local models/ folder.")

        return models, metadata, fi_df, metrics

    except Exception as exc:
        logger.error("Failed to load model from both Hopsworks and local fallback: %s", exc)
        return None, None, pd.DataFrame(), {}


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
            cornerradius=6,
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
# Main Dashboard
# ---------------------------------------------------------------------------

def main():
    """Render the Streamlit dashboard."""

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
        models, metadata, fi_df, model_metrics = load_model_from_registry()

    if live_df.empty:
        st.error("⚠️ Unable to fetch live data from Open-Meteo. Please try again later.")
        return

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

    # ── Sidebar — Model Info ──
    with st.sidebar:
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

        elif model_metrics:
            for key, val in model_metrics.items():
                st.markdown(f"""
                <div class="sidebar-metric">
                    <div class="label">{key.upper()}</div>
                    <div class="value">{val}</div>
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
                        cornerradius=4,
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


if __name__ == "__main__":
    main()
