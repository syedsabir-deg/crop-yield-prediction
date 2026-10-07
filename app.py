import streamlit as st
import joblib
from scipy.sparse import hstack

# Load the trained model and encoder
model = joblib.load("models/crop_yield_model.pkl")
encoder = joblib.load("models/encoder.pkl")

# Page title
st.title("🌾 Crop Yield Prediction")
st.write("Enter the agricultural and climate details to predict crop yield.")

# User inputs
area = st.selectbox("Country", encoder.categories_[0])

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2013,
    value=2000,
    step=1
)

rainfall = st.number_input(
    "Average Rainfall (mm/year)",
    min_value=0.0,
    value=1000.0
)

pesticides = st.number_input(
    "Pesticides Used (tonnes)",
    min_value=0.0,
    value=500.0
)

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=0.0,
    value=20.0
)

crop = st.selectbox("Crop", encoder.categories_[1])

# Prediction button
if st.button("Predict Crop Yield"):

    # Convert country and crop into encoded values
    encoded = encoder.transform([[area, crop]])

    # Numerical values
    numeric = [[
        year,
        rainfall,
        pesticides,
        temperature
    ]]

    # Combine all features
    X_new = hstack([numeric, encoded])

    # Make prediction
    prediction = model.predict(X_new)[0]

    st.success(f"Predicted Crop Yield: {prediction:.2f} hg/ha")
    st.info(f"Approximately {prediction / 10:.2f} kg/ha")
    