from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from scipy.sparse import hstack

# Create FastAPI app
app = FastAPI()

# Load trained model and encoder
model = joblib.load("models/crop_yield_model.pkl")
encoder = joblib.load("models/encoder.pkl")


# Input structure
class CropData(BaseModel):
    area: str
    year: int
    rainfall: float
    pesticides: float
    temperature: float
    crop: str


# Test endpoint
@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running!"}


# Prediction endpoint
@app.post("/predict")
def predict(data: CropData):

    # Encode country and crop
    encoded = encoder.transform([[data.area, data.crop]])

    # Numerical features
    numeric = [[
        data.year,
        data.rainfall,
        data.pesticides,
        data.temperature
    ]]

    # Combine features
    X_new = hstack([numeric, encoded])

    # Prediction
    prediction = model.predict(X_new)[0]

    return {
        "prediction_hg_ha": round(float(prediction), 2),
        "prediction_kg_ha": round(float(prediction / 10), 2)
    }
