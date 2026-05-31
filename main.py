"""
Main script for Home Credit Risk Modeling - Prediction Pipeline
Tests predictor and preprocessor on test dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add modeling directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modeling'))

from modeling.predictor import ModelPredictor, load_predictor
from modeling.preprocessor import DataPreprocessor, create_preprocessor_from_artifact


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("HOME CREDIT RISK MODELING - PREDICTION PIPELINE")
    print("="*70)
    
    # Configuration
    modeling_dir = os.path.join(os.path.dirname(__file__), 'modeling')
    test_data_path = os.path.join(modeling_dir, 'input_data', 'row_fs_data', 'app_test_fs.csv')
    trained_models_dir = os.path.join(modeling_dir, 'model_artifacts')
    output_dir = os.path.join(os.path.dirname(__file__), 'test_predictions')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory ready: {output_dir}")
    
    # Step 1: Load test data
    print("\n" + "-"*70)
    print("STEP 1: Loading Test Data")
    print("-"*70)
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    df_test = pd.read_csv(test_data_path)
    print(f"✓ Test data loaded: {df_test.shape}")
    print(f"  Columns: {list(df_test.columns)[:5]}... (showing first 5)")
    print(f"  Null values: {df_test.isnull().sum().sum()}")
    
    # Step 2: Load trained model
    print("\n" + "-"*70)
    print("STEP 2: Loading Trained Model")
    print("-"*70)
    
    # Get latest model from trained_models directory
    if not os.path.exists(trained_models_dir):
        raise FileNotFoundError(f"Trained models directory not found: {trained_models_dir}")
    
    model_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No trained models found in: {trained_models_dir}")
    
    # Sort by modification time and get latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(trained_models_dir, x)), reverse=True)
    model_path = os.path.join(trained_models_dir, model_files[0])
    
    print(f"✓ Latest model selected: {model_files[0]}")
    
    # Initialize predictor
    predictor = ModelPredictor(model_path)
    
    # Display model information
    print("\n" + "-"*70)
    print("MODEL INFORMATION")
    print("-"*70)
    model_info = predictor.get_model_info()
    print(f"Model Name: {model_info['model_name']}")
    print(f"Created At: {model_info['created_at']}")
    print(f"Number of Features: {model_info['num_features']}")
    print(f"Expected Target Column: {model_info['target_column']}")
    
    print("\nModel Metrics:")
    for metric, value in model_info['metrics'].items():
        print(f"  {metric}: {value:.6f}")
    
    # Step 3: Validate input data
    print("\n" + "-"*70)
    print("STEP 3: Validating Input Data")
    print("-"*70)
    
    is_valid, errors = predictor.validate_input(df_test)
    if not is_valid:
        print(f"✗ Validation failed with the following errors:")
        for error in errors:
            print(f"  - {error}")
        # Continue anyway, as nulls can be handled by preprocessor
        print("\n⚠ Continuing with preprocessing...")
    else:
        print("✓ Input data validation passed")
    
    # Step 4: Make predictions
    print("\n" + "-"*70)
    print("STEP 4: Making Predictions")
    print("-"*70)
    
    predictions = predictor.predict(
        df_test,
        return_proba=True,
        include_id=True,
        id_col='SK_ID_CURR'
    )
    
    print(f"✓ Predictions generated: {predictions.shape}")
    print(f"\nPrediction Statistics:")
    print(f"  Min Probability: {predictions['prediction_probability'].min():.6f}")
    print(f"  Max Probability: {predictions['prediction_probability'].max():.6f}")
    print(f"  Mean Probability: {predictions['prediction_probability'].mean():.6f}")
    print(f"  Median Probability: {predictions['prediction_probability'].median():.6f}")
    
    # Step 5: Save predictions
    print("\n" + "-"*70)
    print("STEP 5: Saving Predictions")
    print("-"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    
    predictions.to_csv(output_file, index=False)
    print(f"✓ Predictions saved: {output_file}")
    
    # Summary statistics
    print("\n" + "-"*70)
    print("PREDICTION SUMMARY")
    print("-"*70)
    print(f"Total Predictions: {len(predictions)}")
    print(f"Output File: predictions_{timestamp}.csv")
    print(f"Output Directory: {output_dir}")
    
    # Display sample predictions
    print("\nSample Predictions (First 10 rows):")
    print(predictions.head(10).to_string())
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return predictions


if __name__ == "__main__":
    try:
        predictions = main()
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
