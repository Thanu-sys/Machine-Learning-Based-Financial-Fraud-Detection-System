import joblib
import json
import os
import numpy as np
from models.models import get_model
from utils.data_utils import load_and_preprocess_data
from evaluation.metrics import calculate_metrics

def save_optimized_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/creditcard.csv')
    
    # Get the optimized XGBoost model
    model = get_model('xgboost')
    
    # Train the model
    print("Training optimized XGBoost model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Save the model
    model_path = 'models/best_xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = 'metrics/xgboost_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Update model version
    version_file = 'model_versions.json'
    versions = {}
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            versions = json.load(f)
    
    versions['xgboost'] = {
        'version': '1.0.0',
        'metrics': metrics,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=4)
    print(f"Version information updated in {version_file}")
    
    # Print performance metrics
    print("\nOptimized XGBoost Model Performance:")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("="*50)

if __name__ == "__main__":
    save_optimized_model() 