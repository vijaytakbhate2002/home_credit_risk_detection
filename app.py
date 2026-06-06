import os
import sys
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modeling'))
from modeling.predictor import ModelPredictor

app = FastAPI(
    title="Home Credit Risk Modeling API",
    description="Production-grade API for serving default probability predictions using LightGBM",
    version="1.0.0"
)


predictor = None
model_metadata = None



@app.on_event("startup")
def startup_event():
    """Load model artifact on server startup to minimize inference latency."""
    global predictor, model_metadata
    
    modeling_dir = os.path.join(os.path.dirname(__file__), 'modeling')
    trained_models_dir = os.path.join(modeling_dir, 'model_artifacts')
    
    if not os.path.exists(trained_models_dir):
        raise FileNotFoundError(f"Model artifacts directory not found at: {trained_models_dir}")
    
    model_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No trained model pickle files found in: {trained_models_dir}")
    
    # Sort and pick latest model by modification time
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(trained_models_dir, x)), reverse=True)
    model_path = os.path.join(trained_models_dir, model_files[0])
    
    print(f"Loading latest trained model from: {model_path}")
    predictor = ModelPredictor(model_path)
    model_metadata = predictor.get_model_info()
    print("Model successfully loaded and ready for predictions.")



@app.get("/")
def get_health_and_info():
    """Health check and model metadata endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized")
    
    return {
        "status": "healthy",
        "app_name": "Home Credit Risk Modeling API",
        "model_details": {
            "model_name": model_metadata["model_name"],
            "created_at": model_metadata["created_at"],
            "number_of_features": model_metadata["num_features"],
            "target_column": model_metadata["target_column"],
            "metrics": model_metadata["metrics"]
        }
    }

@app.post("/predict/single")
def predict_single(features: Dict[str, Any], id_val:int=999999):
    """
    Real-time single-row prediction.
    Takes arbitrary dictionary of feature values. Unprovided features are automatically imputed with NaN
    and aligned by the preprocessor.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized")
    
    try:
        data = features.copy()
        id_col = predictor.preprocessor.id_col
        if id_col not in data:
            data[id_col] = id_val
        df = pd.DataFrame([data])
        predictions = predictor.predict(
            df,
            return_proba=True,
            include_id=True,
            id_col=id_col
        )
        result = predictions.iloc[0].to_dict()
        result["prediction_class"] = int(result["prediction_probability"] >= 0.5)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")



@app.post("/predict/batch/json")
def predict_batch_json(records: List[Dict[str, Any]]):
    """
    Batch predictions via JSON payload.
    Takes a list of feature dictionaries, pads/aligns, and returns predictions.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized")
    
    if not records:
        raise HTTPException(status_code=400, detail="Empty request payload")
        
    try:
        df = pd.DataFrame(records)
        id_col = predictor.preprocessor.id_col
        if id_col not in df.columns:
            df[id_col] = range(100000, 100000 + len(df))
        else:
            df[id_col] = df[id_col].fillna(pd.Series(range(100000, 100000 + len(df))))
        predictions = predictor.predict(
            df,
            return_proba=True,
            include_id=True,
            id_col=id_col
        )
        predictions["prediction_class"] = (predictions["prediction_probability"] >= 0.5).astype(int)
        return predictions.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
