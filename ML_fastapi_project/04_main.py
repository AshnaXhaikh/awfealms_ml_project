from fastapi import FastAPI 
from typing import Dict
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from sklearn.datasets import fetch_california_housing

import pickle
import numpy as np


# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API",
              description="API for predicting house prices using a machine learning model based on census data from California.",
              version="1.0.0")

# Load the model and scaler
try:
    with open('linear_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

except Exception as e:
    print("Error loading model or scaler:", e)

# Load dataset
housing = fetch_california_housing()

# Define feature names
FEATURE_NAMES = housing.feature_names

# Input data model 
class Housingfeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

# Define the output model
class PredictionResponse(BaseModel):
    estimated_value: str
    message: str 
    input_features: dict

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>California Housing Price Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f4f8;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    background-color: #ffffff;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    max-width: 600px;
                }
                h1 {
                    color: #2c7be5;
                    margin-bottom: 10px;
                }
                p {
                    font-size: 18px;
                    color: #555;
                }
                .footer {
                    margin-top: 30px;
                    font-size: 14px;
                    color: #777;
                }
                a {
                    color: #2c3e50;
                    text-decoration: none;
                    margin: 0 10px;
                }
                a:hover {
                    text-decoration: underline;
                }
                .swagger-link {
                    margin-top: 15px;
                    display: inline-block;
                    font-weight: bold;
                    background-color: #2c7be5;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 6px;
                    text-decoration: none;
                    transition: background-color 0.3s;
                }
                .swagger-link:hover {
                    background-color: #1a5fc8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏡 California Housing Price Prediction API</h1>
                <p>Welcome! Use the <code>/predict</code> endpoint to estimate house prices based on input features.</p>
                <a class="swagger-link" href="/docs" target="_blank">🔗 Go to Swagger UI</a>
                <div class="footer">
                    <p><strong>Final Machine Learning Project</strong> – Awfera LMS ML Course</p>
                    <p>Created by: <strong>Ashna Imtiaz</strong></p>
                    <p>
                        <a href="https://github.com/AshnaXhaikh/awfealms_ml_project" target="_blank">GitHub</a> |
                        <a href="https://www.linkedin.com/in/ashna-imtiaz-538335284/" target="_blank">LinkedIn</a>
                    </p>
                </div>
            </div>
        </body>
    </html>
    """
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(data: Housingfeatures):
    """
    Predict the house price based on the input features.

    Args:
        data (Housingfeatures): Input features for the prediction.
    
    Returns:
        PredictionResponse: Estimated house price and input features.
    """
    # convert to numpy array
    data = np.array([
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]).reshape(1, -1)

    # scale the input
    scaled_data = scaler.transform(data)

    # make predictions
    prediction = model.predict(scaled_data)[0]

    # Convert to USD for interpretable output
    estimated_value = round(prediction * 100000, 2)


    return PredictionResponse(
    message="Price predicted successfully",
    estimated_value=f"${estimated_value:,.2f}",
    input_features=dict(zip(FEATURE_NAMES, data[0].tolist()))
)
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8004, reload=True)
