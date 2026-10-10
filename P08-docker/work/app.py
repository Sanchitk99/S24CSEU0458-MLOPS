"The delivery-time prediction service."

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]

app = FastAPI(title="Delivery Time Predictor", version="1.0.0")
model = joblib.load(HERE / "model.joblib")


class Order(BaseModel):
    distance_km: float = Field(..., gt=0, le=50)
    prep_time_min: int = Field(..., ge=0, le=120)
    traffic_level: int = Field(..., ge=1, le=3)
    rain: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(order: Order):
    row = pd.DataFrame([order.model_dump()])[FEATURES]
    return {"delivery_min": round(float(model.predict(row)[0]), 1),
            "model_version": app.version}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLOps Lab 08 - Delivery Time Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 80px auto;
                padding: 20px;
                background: #f4f6f9;
                color: #222;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px #0001;
            }
            h1 { color: #2457a7; }
            a { color: #2457a7; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Delivery Time Predictor</h1>
            <h2>MLOps Lab 08</h2>
            <p><strong>Name:</strong> Sanchit Kukreja</p>
            <p><strong>Enrollment No.:</strong> S24CSEU0458</p>
            <p><strong>Batch:</strong> 42</p>
            <p><strong>Course:</strong> SCSE3040 - MLOps</p>
            <hr>
            <p>Predict delivery time using machine learning.</p>
            <a href="/docs">Open API Documentation</a>
        </div>
    </body>
    </html>
    """
