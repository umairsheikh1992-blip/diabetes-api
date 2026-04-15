# =============================================================
#  Diabetes Prediction API — FastAPI
#  Features: Glucose, BloodPressure, SkinThickness,
#             Insulin, BMI, Age
# =============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── Load model and scaler once at startup ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Feature order — must match exactly what the model was trained on
FEATURE_NAMES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Age",
]

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes risk from 6 clinical features",
    version="1.0.0",
)

# Allow requests from anywhere (needed for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ─────────────────────────────────────────────
class PatientData(BaseModel):
    Glucose:       float = Field(..., example=120.0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., example=70.0,  description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., example=20.0,  description="Triceps skinfold thickness (mm)")
    Insulin:       float = Field(..., example=80.0,  description="2-Hour serum insulin (µU/ml)")
    BMI:           float = Field(..., example=25.0,  description="Body mass index (kg/m²)")
    Age:           float = Field(..., example=30.0,  description="Age in years")

# ── Response model ────────────────────────────────────────────
class PredictionResult(BaseModel):
    diabetic:    bool
    probability: float
    risk_level:  str
    message:     str

# ── Health check endpoint ─────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Diabetes Prediction API is running",
        "endpoints": ["/predict", "/health", "/docs"],
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# ── Prediction endpoint ───────────────────────────────────────
@app.post("/predict", response_model=PredictionResult)
def predict(data: PatientData):
    # Build feature array in the correct order
    features = np.array([[
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.Age,
    ]])

    # Scale using the same scaler used during training
    features_scaled = scaler.transform(features)

    # Predict class (0 or 1) and probability
    prediction  = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    # Determine risk level for the app UI
    if probability < 0.30:
        risk_level = "Low"
        message = "Your results suggest a low risk of diabetes. Maintain a healthy lifestyle."
    elif probability < 0.60:
        risk_level = "Moderate"
        message = "Your results suggest a moderate risk. Please consult a doctor for a full check-up."
    else:
        risk_level = "High"
        message = "Your results suggest a high risk of diabetes. Please see a doctor as soon as possible."

    return PredictionResult(
        diabetic    = bool(prediction),
        probability = round(probability, 4),
        risk_level  = risk_level,
        message     = message,
    )
