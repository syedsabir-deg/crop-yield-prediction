# 🌾 Crop Yield Prediction

A Machine Learning project that predicts crop yield based on agricultural and climate-related factors.

## 📌 Project Overview

This project uses historical crop data to predict crop yield in hectograms per hectare (hg/ha).

The model takes the following inputs:

- Country
- Year
- Average rainfall
- Pesticides used
- Average temperature
- Crop type

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib
- Streamlit

## 🤖 Machine Learning Model

A Random Forest Regressor was used to predict crop yield.

Categorical features such as country and crop type were converted into numerical features using One-Hot Encoding.

## 📊 Model Performance

R² Score: 0.9878

Mean Absolute Error (MAE): 3773.57 hg/ha

Root Mean Squared Error (RMSE): 9575.43 hg/ha

## 🌐 Web Application

The trained model is deployed locally using Streamlit.

Users can enter agricultural and climate information and receive a predicted crop yield.

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt