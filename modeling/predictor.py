"""
Prediction module for Home Credit Risk Modeling
Loads trained model and makes predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Union, Tuple, List, Dict, Optional
from datetime import datetime
from preprocessor import DataPreprocessor, create_preprocessor_from_artifact


class ModelPredictor:
    """
    A comprehensive prediction class that handles:
    - Loading trained models
    - Preprocessing new data
    - Making predictions
    - Handling batch and single predictions
    """
    
    def __init__(self, model_path: str, preprocessor: Optional[DataPreprocessor] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model artifact
            preprocessor: Optional DataPreprocessor instance. If None, will extract from model artifact.
        """
        self.model_path = model_path
        self.model_artifact = None
        self.model = None
        self.metadata = None
        self.preprocessor = preprocessor
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model artifact."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model_artifact = joblib.load(self.model_path)
        self.model = self.model_artifact["model"]
        self.metadata = self.model_artifact["metadata"]
        
        # If preprocessor not provided, extract from artifact
        if self.preprocessor is None:
            self.preprocessor = create_preprocessor_from_artifact(self.model_path)
        
        print(f"Model loaded successfully from: {self.model_path}")
        print(f"Model: {self.metadata['model_name']}")
        print(f"Created at: {self.metadata['created_at']}")
        print(f"Number of features: {len(self.metadata['input_features'])}")
    
    def preprocess(self, df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
        """
        Preprocess input data using fitted preprocessor.
        
        Args:
            df: Input dataframe
            drop_id: Whether to drop ID column after preprocessing
            
        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        id_col = self.preprocessor.id_col
        target_col = self.metadata['target_column']
        
        # Drop TARGET if present (it should NOT be in test data, but just in case)
        if target_col in df_processed.columns:
            df_processed = df_processed.drop(target_col, axis=1)
        
        # DO NOT drop ID yet - the imputers were fitted WITH ID column
        # Apply preprocessing (imputation, encoding, scaling) which will include ID
        print(f"[DEBUG] Preprocessing data: input shape {df_processed.shape}")
        df_processed = self.preprocessor.transform(df_processed, align_columns=True)
        print(f"[DEBUG] After preprocessing: output shape {df_processed.shape}")
        
        # NOW drop ID if requested
        if drop_id and id_col in df_processed.columns:
            df_processed = df_processed.drop(id_col, axis=1)
            print(f"[DEBUG] Dropped ID column: output shape {df_processed.shape}")
        
        return df_processed
    
    def predict(self, df: pd.DataFrame, return_proba: bool = True, 
                include_id: bool = False, id_col: Optional[str] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            df: Input dataframe with features
            return_proba: Whether to return probability or class predictions
            include_id: Whether to include ID column in output
            id_col: Name of ID column (if different from preprocessor ID column)
            
        Returns:
            Predictions as dataframe or numpy array
        """
        id_col = id_col or self.preprocessor.id_col
        
        # Extract ID if needed
        ids = None
        if include_id and id_col in df.columns:
            ids = df[id_col].copy()
        
        print(f"[DEBUG] Input shape before preprocessing: {df.shape}")
        
        # Preprocess data
        df_processed = self.preprocess(df, drop_id=True)
        
        # Get expected features (excluding ID and TARGET)
        expected_features = self.metadata['input_features']
        expected_features = [f for f in expected_features if f not in [id_col, self.metadata['target_column']]]
        
        print(f"[DEBUG] Expected features: {len(expected_features)}")
        print(f"[DEBUG] Processed dataframe columns: {len(df_processed.columns)}")
        
        # Check for missing features
        missing = set(expected_features) - set(df_processed.columns)
        if missing:
            raise ValueError(f"Missing features after preprocessing: {missing}")
        
        # Select only expected features in the correct order
        df_processed = df_processed[expected_features]
        print(f"[DEBUG] Final input shape for model: {df_processed.shape}")
        
        # Make predictions
        if return_proba:
            predictions = self.model.predict_proba(df_processed, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            predictions = self.model.predict(df_processed, num_iteration=self.model.best_iteration_)
        
        print(f"[DEBUG] Predictions shape: {predictions.shape}")
        
        # Format output
        if include_id and ids is not None:
            result = pd.DataFrame({
                id_col: ids.values,
                'prediction_probability': predictions if return_proba else None,
                'prediction_class': predictions if not return_proba else None
            })
            # Remove None columns
            result = result.dropna(axis=1)
            return result
        
        return predictions
    
    def predict_batch(self, df: pd.DataFrame, batch_size: int = 1000, 
                     return_proba: bool = True) -> np.ndarray:
        """
        Make predictions on large datasets in batches.
        
        Args:
            df: Input dataframe
            batch_size: Number of samples per batch
            return_proba: Whether to return probability or class predictions
            
        Returns:
            Prediction array
        """
        all_predictions = []
        n_samples = len(df)
        
        for i in range(0, n_samples, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_pred = self.predict(batch_df, return_proba=return_proba, include_id=False)
            all_predictions.append(batch_pred)
            print(f"Processed batch: {min(i + batch_size, n_samples)}/{n_samples}")
        
        return np.concatenate(all_predictions)
    
    def get_model_info(self) -> Dict:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.metadata['model_name'],
            "created_at": self.metadata['created_at'],
            "target_column": self.metadata['target_column'],
            "num_features": len(self.metadata['input_features']),
            "input_features": self.metadata['input_features'],
            "metrics": self.metadata['model_metrics'],
            "library_versions": self.metadata['library_versions']
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of input feature names."""
        return self.metadata['input_features']
    
    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input dataframe against expected features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        expected_features = self.metadata['input_features']
        
        # Check for missing columns
        missing = set(expected_features) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
        
        # Check for null values
        null_cols = df[expected_features].columns[df[expected_features].isnull().any()].tolist()
        if null_cols:
            errors.append(f"Null values found in: {null_cols}")
        
        return len(errors) == 0, errors


def load_predictor(model_path: str, 
                   preprocessor_path: Optional[str] = None) -> ModelPredictor:
    """
    Utility function to load a predictor.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Optional path to saved preprocessor
        
    Returns:
        ModelPredictor instance
    """
    preprocessor = None
    if preprocessor_path and os.path.exists(preprocessor_path):
        preprocessor = DataPreprocessor.load(preprocessor_path)
    
    return ModelPredictor(model_path, preprocessor)


# Example usage
if __name__ == "__main__":
    # Initialize predictor with trained model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    trained_models = [f for f in os.listdir(os.path.join(model_dir, "trained_models")) 
                      if f.endswith(".pkl")]
    
    if trained_models:
        model_path = os.path.join(model_dir, "trained_models", trained_models[-1])
        
        predictor = ModelPredictor(model_path)
        
        # Display model info
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        info = predictor.get_model_info()
        for key, value in info.items():
            if key not in ['input_features', 'metrics', 'library_versions']:
                print(f"{key}: {value}")
        
        print("\nModel Metrics:")
        for metric, value in info['metrics'].items():
            print(f"  {metric}: {value:.6f}")
        
        print("\nLibrary Versions:")
        for lib, version in info['library_versions'].items():
            print(f"  {lib}: {version}")
        
        print(f"\nTotal Features: {info['num_features']}")
        print("="*60)
