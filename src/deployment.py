from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
import json
from datetime import datetime
from typing import List, Dict, Any
import uvicorn

ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / 'config.yaml'
# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Deployment API",
    description="Manual MLOps Project - Machine Failure Prediction",
    # version="1.0.0"
)

# Load model
model_dir = ROOT_DIR / 'models'
model = joblib.load(model_dir / 'model_v1.pkl')  # Manual version number!
label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
scaler = joblib.load(model_dir / 'scaler.pkl')

print("Loaded model_v1.pkl")

# Required fields mapping (ignores extras automatically)
REQUIRED_FIELDS = {
    'Air temperature [K]': float,
    'Process temperature [K]': float,
    'Rotational speed [rpm]': float,
    'Torque [Nm]': float,
    'Tool wear [min]': float,
    'Type': str
}

def validate_and_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate only required fields, ignore extras."""
    validated = {}
    for field, expected_type in REQUIRED_FIELDS.items():
        if field in data:
            try:
                validated[field] = expected_type(data[field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid type for {field}: expected {expected_type.__name__}")
        else:
            raise ValueError(f"Missing required field: {field}")
    return validated

@app.get("/")
def root():
    return {"status": "running", "model": "model_v1.pkl"}

@app.post("/predict")
def predict(data: Dict[str, Any]):
    # Validate and extract only required fields (ignores extras)
    input_data = validate_and_extract(data)
    
    # Create features
    features = {
        'Air temperature [K]': input_data['Air temperature [K]'],
        'Process temperature [K]': input_data['Process temperature [K]'],
        'Rotational speed [rpm]': input_data['Rotational speed [rpm]'],
        'Torque [Nm]': input_data['Torque [Nm]'],
        'Tool wear [min]': input_data['Tool wear [min]'],
        'Type_Encoded': label_encoder.transform([input_data['Type']])[0],
        'Temp_Diff': input_data['Process temperature [K]'] - input_data['Air temperature [K]'],
        'Power': input_data['Torque [Nm]'] * input_data['Rotational speed [rpm]'] / 9549
    }
    
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n⚠️  MANUAL STEP REQUIRED:")
    print("1. Open deployment_log.csv")
    print("2. Add a new row:")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},model_v1.pkl,deployed")
    print("\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)