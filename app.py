import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AquaGenesis Intelligence", layout="wide")

# ================= SEASON CONFIG =================
SEASON_COLORS = {
    "Winter (Dec–Feb)": "#3B82F6",
    "Summer (Mar–May)": "#F97316",
    "Monsoon (Jun–Sep)": "#10B981",
    "Post-Monsoon (Oct–Nov)": "#8B5CF6"
}

SEASON_ORDER = [
    "Winter (Dec–Feb)",
    "Summer (Mar–May)",
    "Monsoon (Jun–Sep)",
    "Post-Monsoon (Oct–Nov)"
]

# ================= SIDEBAR =================
st.sidebar.title("🌊 AquaGenesis")
st.sidebar.markdown("Hybrid AI Atmospheric Water Intelligence")

STATES = {
    "Andhra Pradesh (Amaravati)": (16.5730, 80.3575),
    "Arunachal Pradesh (Itanagar)": (27.0844, 93.6053),
    "Assam (Dispur)": (26.1408, 91.7900),
    "Bihar (Patna)": (25.5941, 85.1376),
    "Chhattisgarh (Raipur)": (21.2514, 81.6296),
    "Goa (Panaji)": (15.4909, 73.8278),
    "Gujarat (Gandhinagar)": (23.2156, 72.6369),
    "Haryana (Chandigarh)": (30.7333, 76.7794),
    "Himachal Pradesh (Shimla)": (31.1048, 77.1734),
    "Jharkhand (Ranchi)": (23.3441, 85.3096),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Kerala (Thiruvananthapuram)": (8.5241, 76.9366),
    "Madhya Pradesh (Bhopal)": (23.2599, 77.4126),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Manipur (Imphal)": (24.8170, 93.9368),
    "Meghalaya (Shillong)": (25.5788, 91.8933),
    "Mizoram (Aizawl)": (23.7271, 92.7176),
    "Nagaland (Kohima)": (25.6751, 94.1086),
    "Odisha (Bhubaneswar)": (20.2961, 85.8245),
    "Punjab (Chandigarh)": (30.7333, 76.7794),
    "Rajasthan (Jaipur)": (26.9124, 75.7873),
    "Sikkim (Gangtok)": (27.3389, 88.6065),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tripura (Agartala)": (23.8315, 91.2868),
    "Uttar Pradesh (Lucknow)": (26.8467, 80.9462),
    "Uttarakhand (Dehradun)": (30.3165, 78.0322),
    "West Bengal (Kolkata)": (22.5726, 88.3639)
}

state = st.sidebar.selectbox("Select State", list(STATES.keys()))
run = st.sidebar.button("Run Full Analysis")

# ================= FETCH WEATHER =================
@st.cache_data(ttl=3600)
def fetch_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }

    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()

                if "hourly" not in data:
                    return None

                df = pd.DataFrame({
                    "time": pd.to_datetime(data["hourly"]["time"]),
                    "temperature": data["hourly"]["temperature_2m"],
                    "humidity": data["hourly"]["relative_humidity_2m"],
                    "dew_point": data["hourly"]["dewpoint_2m"],
                    "pressure": data["hourly"]["surface_pressure"]
                }).dropna()

                df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1
                df["month"] = df["time"].dt.month

                df["season"] = df["month"].apply(
                    lambda m: "Winter (Dec–Feb)" if m in [12,1,2] else
                    "Summer (Mar–May)" if m in [3,4,5] else
                    "Monsoon (Jun–Sep)" if m in [6,7,8,9] else
                    "Post-Monsoon (Oct–Nov)"
                )

                return df
        except:
            time.sleep(1)

    return None

# ================= TRAIN MODEL =================
@st.cache_resource
def train_models():
    all_data = []
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)

    for lat, lon in STATES.values():
        df = fetch_weather(lat, lon, start, end)
        if df is not None:
            all_data.append(df)
            time.sleep(0.5)

    if not all_data:
        st.error("API failed for all states")
        st.stop()

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    X_lstm, y_lstm = [], []
    for i in range(12, len(scaled)):
        X_lstm.append(scaled[i-12:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(12,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_lstm, y_lstm, epochs=2, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ================= MAIN =================
st.title("Atmospheric Water Intelligence Dashboard")

if run:

    lat, lon = STATES[state]

    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today()-timedelta(days=1))
    if past is None:
        st.error("Past data API failed")
        st.stop()

    st.metric("Current Water Yield", round(past["water_yield"].iloc[-1],3))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=past["time"], y=past["water_yield"]))
    st.plotly_chart(fig1)

    season_df = fetch_weather(lat, lon, date.today()-timedelta(days=365), date.today())
    if season_df is None:
        st.error("Seasonal API failed")
        st.stop()

    seasonal_avg = season_df.groupby("season")["water_yield"].mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=seasonal_avg.index, y=seasonal_avg.values))
    st.plotly_chart(fig2)

    # ===== FORECAST =====
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure"
    }

    f = None
    for _ in range(3):
        try:
            res = requests.get(forecast_url, params=params, timeout=10)
            if res.status_code == 200:
                f = res.json()
                break
        except:
            time.sleep(1)

    if f is None or "hourly" not in f:
        st.error("Forecast API failed")
        st.stop()

    future_df = pd.DataFrame({
        "temperature": f["hourly"]["temperature_2m"],
        "humidity": f["hourly"]["relative_humidity_2m"],
        "dew_point": f["hourly"]["dewpoint_2m"],
        "pressure": f["hourly"]["surface_pressure"]
    }).head(12)

    xgb_pred = xgb.predict(future_df)

    st.metric("Predicted Yield", round(np.mean(xgb_pred),3))
