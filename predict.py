import joblib
from scipy.sparse import hstack

# Load trained model and encoder
model = joblib.load("models/crop_yield_model.pkl")
encoder = joblib.load("models/encoder.pkl")

# Get user inputs
area = input("Enter country: ")
year = int(input("Enter year: "))
rainfall = float(input("Enter average rainfall (mm/year): "))
pesticides = float(input("Enter pesticides used (tonnes): "))
temperature = float(input("Enter average temperature (°C): "))
crop = input("Enter crop: ")

# Encode country and crop
encoded = encoder.transform([[area, crop]])

# Numerical features
numeric = [[
    year,
    rainfall,
    pesticides,
    temperature
]]

# Combine numerical + encoded features
X_new = hstack([numeric, encoded])

# Predict yield
prediction = model.predict(X_new)[0]

print("\nPredicted Crop Yield:", round(prediction, 2), "hg/ha")
print("Predicted Crop Yield:", round(prediction / 10, 2), "kg/ha")