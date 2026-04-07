import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AquaGenesis Intelligence", layout="wide")

# ================= SAFE API =================
def safe_api_call(url, params, retries=3):
    for i in range(retries):
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 200:
                return res.json()
        except:
            continue
    return None

# ================= STATES =================
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

# ================= FETCH =================
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

    r = safe_api_call(url, params)

    if r is None or "hourly" not in r:
        return None

    df = pd.DataFrame({
        "time": pd.to_datetime(r["hourly"]["time"]),
        "temperature": r["hourly"]["temperature_2m"],
        "humidity": r["hourly"]["relative_humidity_2m"],
        "dew_point": r["hourly"]["dewpoint_2m"],
        "pressure": r["hourly"]["surface_pressure"]
    }).dropna()

    df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1

    return df

# ================= TRAIN REAL MODEL =================
@st.cache_resource
def train_models():
    all_data = []

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)

    for state, (lat, lon) in STATES.items():
        df = fetch_weather(lat, lon, start, end)
        if df is not None:
            all_data.append(df)

    if not all_data:
        return None, None, None

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X, y)

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
    lstm.fit(X_lstm, y_lstm, epochs=3, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ================= UI =================
st.title("🌊 AquaGenesis – 100% Real Data Mode")

if xgb is None:
    st.error("❌ Unable to train model. API failed for all states.")
    st.stop()

state = st.selectbox("Select State", list(STATES.keys()))
lat, lon = STATES[state]

# ================= CURRENT =================
past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today())

if past is None:
    st.error("❌ Live data unavailable. Try again later.")
    st.stop()

st.metric("Current Water Yield", round(past["water_yield"].iloc[-1],3))

# ================= GRAPH =================
fig = go.Figure()
fig.add_trace(go.Scatter(x=past["time"], y=past["water_yield"]))
st.plotly_chart(fig)

# ================= FORECAST =================
forecast_url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure"
}

f = safe_api_call(forecast_url, params)

if f is None:
    st.error("❌ Forecast API failed.")
    st.stop()

future_df = pd.DataFrame({
    "temperature": f["hourly"]["temperature_2m"],
    "humidity": f["hourly"]["relative_humidity_2m"],
    "dew_point": f["hourly"]["dewpoint_2m"],
    "pressure": f["hourly"]["surface_pressure"]
}).head(12)

xgb_pred = xgb.predict(future_df)

st.metric("Predicted Yield", round(np.mean(xgb_pred),3))
