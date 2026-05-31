"""
Data Preprocessing module for Home Credit Risk Modeling
Handles imputation, encoding, and scaling of features
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import joblib
import os
from typing import Tuple, List, Dict, Optional


class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles:
    - Imputation of missing values (separate for numeric and categorical)
    - Categorical encoding
    - Feature scaling
    """
    
    def __init__(self, id_col: str = "SK_ID_CURR", target_col: str = "TARGET"):
        """
        Initialize the preprocessor.
        
        Args:
            id_col: Name of the ID column
            target_col: Name of the target column
        """
        self.id_col = id_col
        self.target_col = target_col
        
        # Preprocessing components - separate imputers for numeric and categorical
        self.num_imputer = None  # For numeric columns only
        self.cat_imputer = None  # For categorical columns only
        self.encoder = None      # For encoding categorical columns
        self.scaler = None       # For scaling all features
        
        # Feature information - these specify which columns each imputer handles
        self.numeric_cols = None       # List of numeric columns
        self.categorical_cols = None   # List of categorical columns
        self.scale_cols = None         # All columns to scale (excluding ID and TARGET)
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training dataframe
            
        Returns:
            self: Fitted preprocessor instance
        """
        # Identify column types BEFORE dropping ID and TARGET
        all_cols = df.drop([self.id_col, self.target_col], axis=1, errors='ignore')
        self.categorical_cols = all_cols.select_dtypes(include='object').columns.tolist()
        self.numeric_cols = all_cols.select_dtypes(exclude='object').columns.tolist()
        
        # Columns to scale (all except ID and TARGET)
        self.scale_cols = [
            col for col in df.columns 
            if col not in [self.id_col, self.target_col]
        ]
        
        print(f"[DEBUG] Found {len(self.numeric_cols)} numeric columns")
        print(f"[DEBUG] Found {len(self.categorical_cols)} categorical columns")
        
        # Fit imputers on their respective columns
        if self.numeric_cols:
            self.num_imputer = SimpleImputer(strategy='median')
            self.num_imputer.fit(df[self.numeric_cols])
            print(f"[DEBUG] Fitted num_imputer on columns: {self.numeric_cols[:3]}... (showing first 3)")
        
        if self.categorical_cols:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            self.cat_imputer.fit(df[self.categorical_cols])
            print(f"[DEBUG] Fitted cat_imputer on columns: {self.categorical_cols[:3]}... (showing first 3)")
        
        # Fit encoder on categorical columns
        if self.categorical_cols:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.encoder.fit(df[self.categorical_cols])
            print(f"[DEBUG] Fitted encoder on categorical columns")
        
        # Fit scaler on all feature columns
        if self.scale_cols:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.scale_cols])
            print(f"[DEBUG] Fitted scaler on {len(self.scale_cols)} columns")
        
        self.is_fitted = True
        return self
    
    def _align_and_prepare_columns(self, df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        """
        Helper to select, reorder, and dynamically pad missing columns as NaN.
        
        Args:
            df: Input dataframe
            expected_cols: List of column names expected by a component
            
        Returns:
            Aligned and padded dataframe matching expected_cols exactly
        """
        df_aligned = df.copy()
        missing_cols = [col for col in expected_cols if col not in df_aligned.columns]
        if missing_cols:
            print(f"[WARNING] Adding {len(missing_cols)} missing columns as NaN: {missing_cols[:5]}...")
            for col in missing_cols:
                df_aligned[col] = np.nan
        return df_aligned[expected_cols]

    def transform(self, df: pd.DataFrame, align_columns: bool = True) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        Applies imputation, encoding, and scaling in sequence.
        
        Args:
            df: Dataframe to transform
            align_columns: Whether to align columns with fitted features
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        df_processed = df.copy()
        
        print(f"[DEBUG] Input shape: {df_processed.shape}")
        
        # Step 1: Imputation - Align & apply separately to numeric/categorical columns
        if self.numeric_cols and self.num_imputer is not None:
            print(f"[DEBUG] Applying num_imputer to {len(self.numeric_cols)} numeric columns...")
            df_num = self._align_and_prepare_columns(df_processed, self.numeric_cols)
            df_processed[self.numeric_cols] = self.num_imputer.transform(df_num)
        
        if self.categorical_cols and self.cat_imputer is not None:
            print(f"[DEBUG] Applying cat_imputer to {len(self.categorical_cols)} categorical columns...")
            df_cat = self._align_and_prepare_columns(df_processed, self.categorical_cols)
            df_processed[self.categorical_cols] = self.cat_imputer.transform(df_cat)
        
        # Step 2: Encoding - Align & apply to categorical columns
        if self.categorical_cols and self.encoder is not None:
            print(f"[DEBUG] Applying encoder to {len(self.categorical_cols)} categorical columns...")
            df_cat = self._align_and_prepare_columns(df_processed, self.categorical_cols)
            df_processed[self.categorical_cols] = self.encoder.transform(df_cat)
        
        # Step 3: Scaling - Align & apply to scale columns
        if self.scale_cols and self.scaler is not None:
            print(f"[DEBUG] Applying scaler to {len(self.scale_cols)} columns...")
            df_scale = self._align_and_prepare_columns(df_processed, self.scale_cols)
            df_processed[self.scale_cols] = self.scaler.transform(df_scale)
        
        # Step 4: Construct final output schema
        output_cols = []
        if self.id_col in df.columns:
            output_cols.append(self.id_col)
        output_cols.extend(self.scale_cols)
        
        df_processed = df_processed[output_cols]
        print(f"[DEBUG] Output shape: {df_processed.shape}")
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Transformed dataframe
        """
        return self.fit(df).transform(df)
    
    def save(self, path: str) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        joblib.dump(self, path, compress=3)
        print(f"Preprocessor saved to: {path}")
    
    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            path: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor instance
        """
        preprocessor = joblib.load(path)
        print(f"Preprocessor loaded from: {path}")
        return preprocessor
    
    def get_feature_info(self) -> Dict:
        """
        Get information about features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "scale_columns": self.scale_cols,
            "id_column": self.id_col,
            "target_column": self.target_col,
            "is_fitted": self.is_fitted
        }


def create_preprocessor_from_artifact(artifact_path: str) -> DataPreprocessor:
    """
    Create a DataPreprocessor from a trained model artifact.
    Extracts imputers, encoder, and scaler from the saved artifact.
    
    Args:
        artifact_path: Path to the model artifact
        
    Returns:
        DataPreprocessor instance with fitted components
    """
    artifact = joblib.load(artifact_path)
    
    preprocessor = DataPreprocessor()
    
    # Load preprocessing components from artifact
    preprocessing = artifact["preprocessing"]
    preprocessor.num_imputer = preprocessing["num_imputer"]
    preprocessor.cat_imputer = preprocessing["cat_imputer"]
    preprocessor.encoder = preprocessing["encoder"]
    preprocessor.scaler = preprocessing["scaler"]
    
    # Extract metadata
    metadata = artifact["metadata"]
    
    # IMPORTANT: The input_features in metadata are the columns AFTER X.columns was taken
    # which is AFTER dropping ID and TARGET from app_train_fe
    # But the imputers were fitted on data BEFORE dropping ID and TARGET
    # So we need to use the imputer's n_features_in to get the actual columns
    
    # Get the exact features the imputer was fitted on
    preprocessor.numeric_cols = list(preprocessor.num_imputer.feature_names_in_) if hasattr(preprocessor.num_imputer, 'feature_names_in_') else []
    preprocessor.categorical_cols = list(preprocessor.cat_imputer.feature_names_in_) if hasattr(preprocessor.cat_imputer, 'feature_names_in_') else []
    preprocessor.scale_cols = list(preprocessor.scaler.feature_names_in_) if hasattr(preprocessor.scaler, 'feature_names_in_') else metadata["input_features"]
    
    print(f"[DEBUG] Loaded preprocessor from artifact:")
    print(f"[DEBUG] - Numeric cols (from num_imputer): {len(preprocessor.numeric_cols)}")
    print(f"[DEBUG] - Categorical cols (from cat_imputer): {len(preprocessor.categorical_cols)}")
    print(f"[DEBUG] - Scale cols (from scaler): {len(preprocessor.scale_cols)}")
    print(f"[DEBUG] - First numeric cols: {preprocessor.numeric_cols[:3]}")
    print(f"[DEBUG] - First categorical cols: {preprocessor.categorical_cols[:3]}")
    print(f"[DEBUG] - First scale cols: {preprocessor.scale_cols[:3]}")
    
    preprocessor.is_fitted = True
    return preprocessor
