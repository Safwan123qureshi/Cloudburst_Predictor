import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

# =====================================
# LOAD ENV
# =====================================

load_dotenv()

API_KEY = os.getenv("API_KEY")

# =====================================
# LOAD DATASET
# =====================================

data = pd.read_csv("synthetic_rainfall_data.csv")

# Create Label
data['Cloudburst'] = data['rainfall_mm'].apply(
    lambda x: 1 if x > 50 else 0
)

# =====================================
# FEATURES & TARGET
# =====================================

X = data[[
    'temperature_c',
    'humidity_pct',
    'pressure_hpa',
    'wind_speed_ms',
    'cloud_cover_pct'
]]

y = data['Cloudburst']

# =====================================
# TRAIN MODEL
# =====================================

model = LogisticRegression()
model.fit(X, y)

# =====================================
# STREAMLIT UI
# =====================================

st.title("🌧️ Worldwide Cloudburst Predictor")

country = st.text_input("Country Name")
city = st.text_input("City Name")

# =====================================
# COUNTRY CODES
# =====================================

country_codes = {
    "pakistan": "PK",
    "india": "IN",
    "united states": "US",
    "america": "US",
    "china": "CN",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "uk": "GB",
    "united kingdom": "GB"
}

# =====================================
# BUTTON
# =====================================

if st.button("Check Weather & Predict"):

    # Empty Check
    if country == "" or city == "":

        st.warning("⚠️ Please enter Country and City")

    else:

        try:

            # =====================================
            # COUNTRY VALIDATION
            # =====================================

            user_country = country.lower()

            if user_country not in country_codes:

                st.error("❌ Country not supported")
                st.stop()

            country_code = country_codes[user_country]

            # =====================================
            # GEO API
            # =====================================

            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={API_KEY}"

            geo_response = requests.get(geo_url)

            geo_data = geo_response.json()

            # Invalid Location
            if len(geo_data) == 0:

                st.error("❌ Invalid Location")

            else:

                # =====================================
                # STRICT COUNTRY VALIDATION
                # =====================================

                found = False

                for place in geo_data:

                    api_city = place['name'].lower()
                    api_country = place['country']

                    # Exact city + country match
                    if api_city == city.lower() and api_country == country_code:

                        found = True

                        lat = place['lat']
                        lon = place['lon']

                        break

                # Country mismatch
                if found == False:

                    st.error("❌ City does not belong to this country")
                    st.stop()

                # =====================================
                # WEATHER API
                # =====================================

                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

                weather_response = requests.get(weather_url)

                weather_data = weather_response.json()

                # =====================================
                # WEATHER DETAILS
                # =====================================

                temp = weather_data['main']['temp']
                hum = weather_data['main']['humidity']
                press = weather_data['main']['pressure']
                wind = weather_data['wind']['speed']
                cloud = weather_data['clouds']['all']
                description = weather_data['weather'][0]['description']

                # =====================================
                # SHOW WEATHER
                # =====================================

                st.subheader("🌍 Live Weather Information")

                st.write(f"📍 Location: {city}, {country}")
                st.write(f"🌡️ Temperature: {temp} °C")
                st.write(f"💧 Humidity: {hum}%")
                st.write(f"📉 Pressure: {press} hPa")
                st.write(f"💨 Wind Speed: {wind} m/s")
                st.write(f"☁️ Cloud Cover: {cloud}%")
                st.write(f"🌤️ Weather: {description}")

                # =====================================
                # PREDICTION INPUT
                # =====================================

                input_data = pd.DataFrame(
                    [[temp, hum, press, wind, cloud]],
                    columns=[
                        'temperature_c',
                        'humidity_pct',
                        'pressure_hpa',
                        'wind_speed_ms',
                        'cloud_cover_pct'
                    ]
                )

                # =====================================
                # PREDICTION
                # =====================================

                result = model.predict(input_data)

                # =====================================
                # RESULT
                # =====================================

                st.subheader("🔍 Prediction Result")

                if result[0] == 1:

                    st.error("⚠️ Cloudburst Possible!")

                else:

                    st.success("✅ No Cloudburst")

        except:

            st.error("❌ Something went wrong")
         
