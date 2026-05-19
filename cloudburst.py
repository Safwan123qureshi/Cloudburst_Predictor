import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

# =========================
# LOAD ENV
# =========================
load_dotenv()
API_KEY = os.getenv("API_KEY")

# =========================
# LOAD ML DATASET
# =========================
data = pd.read_csv("synthetic_rainfall_data.csv")

data['Cloudburst'] = data['rainfall_mm'].apply(lambda x: 1 if x > 50 else 0)

X = data[
    ['temperature_c', 'humidity_pct', 'pressure_hpa', 'wind_speed_ms', 'cloud_cover_pct']
]
y = data['Cloudburst']

model = LogisticRegression()
model.fit(X, y)

# =========================
# LOAD CITY DATASET (CSV FIX)
# =========================
cities = pd.read_csv("worldcitiespop.csv")

# =========================
# UI
# =========================
st.title("🌧️ Cloudburst Predictor")

country = st.text_input("Country Name")
city = st.text_input("City Name")

# =========================
# COUNTRY CODES
# =========================
country_codes = {
    "pakistan": "PK",
    "india": "IN",
    "united states": "US",
    "china": "CN",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "uk": "GB",
    "bangladesh": "BD"
}

# =========================
# BUTTON
# =========================
if st.button("Check Weather & Predict"):

    if country == "" or city == "":
        st.warning("⚠️ Please enter Country and City")
        st.stop()

    user_country = country.lower()

    if user_country not in country_codes:
        st.error("❌ Country not supported")
        st.stop()

    country_code = country_codes[user_country]

    # =========================
    # CITY VALIDATION (CSV FIX)
    # =========================

    # IMPORTANT: adjust column names if needed
    filtered = cities[
        (cities["Country"].str.upper() == country_code) &
        (cities["City"].str.lower() == city.lower())
    ]

    if filtered.empty:
        st.error("❌ City not found in selected country")
        st.stop()

    lat = filtered.iloc[0]["Latitude"]
    lon = filtered.iloc[0]["Longitude"]

    # =========================
    # WEATHER API
    # =========================
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    weather = requests.get(weather_url).json()

    temp = weather['main']['temp']
    hum = weather['main']['humidity']
    press = weather['main']['pressure']
    wind = weather['wind']['speed']
    cloud = weather['clouds']['all']
    desc = weather['weather'][0]['description']

    # =========================
    # SHOW WEATHER
    # =========================
    st.subheader("🌍 Live Weather Info")

    st.write(f"📍 Location: {city}, {country}")
    st.write(f"🌡️ Temperature: {temp} °C")
    st.write(f"💧 Humidity: {hum}%")
    st.write(f"📉 Pressure: {press} hPa")
    st.write(f"💨 Wind Speed: {wind} m/s")
    st.write(f"☁️ Cloud Cover: {cloud}%")
    st.write(f"🌤️ Condition: {desc}")

    # =========================
    # PREDICTION
    # =========================
    input_data = pd.DataFrame([[temp, hum, press, wind, cloud]],
                              columns=[
                                  'temperature_c',
                                  'humidity_pct',
                                  'pressure_hpa',
                                  'wind_speed_ms',
                                  'cloud_cover_pct'
                              ])

    result = model.predict(input_data)

    st.subheader("🔍 Prediction Result")

    if result[0] == 1:
        st.error("⚠️ Cloudburst Possible!")
    else:
        st.success("✅ No Cloudburst Risk")
         
