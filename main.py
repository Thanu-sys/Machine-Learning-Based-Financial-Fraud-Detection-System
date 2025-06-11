import os
import numpy as np
from utils.data_utils import load_and_preprocess_data, create_federated_data, get_class_distribution
from models.models import get_model
from resampling.resampling import ResamplingTechnique
from federated.federated_learning import FederatedClient, FederatedServer
from evaluation.metrics import calculate_metrics, plot_precision_recall_curve, plot_roc_curve, plot_class_distribution
import time
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.base import BaseEstimator
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_local_models(clients, X_test, y_test):
    """
    Evaluate each client's local model performance.
    
    Args:
        clients: List of FederatedClient objects
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of metrics for each client
    """
    local_results = {}
    for i, client in enumerate(clients):
        try:
            y_pred = client.model.predict(X_test)
            y_prob = client.model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            local_results[f'Client_{i+1}'] = metrics
        except Exception as e:
            print(f"Warning: Could not evaluate client {i+1}: {str(e)}")
    return local_results

def plot_resampling_comparison(results, metric_name, save_path):
    """
    Plot comparison of resampling techniques for a specific metric.
    
    Args:
        results (dict): Dictionary containing results for all models and resampling methods
        metric_name (str): Name of the metric to plot
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    models = list(results.keys())
    resampling_methods = ['smote', 'adasyn', 'random']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, method in enumerate(resampling_methods):
        values = [results[model][method]['final_metrics'][metric_name] 
                 for model in models if results[model][method]]
        plt.bar(x + i*width, values, width, label=method.upper())
    
    plt.xlabel('Models')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Comparison of {metric_name.capitalize()} Across Resampling Techniques')
    plt.xticks(x + width, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_resampling_summary(results):
    """
    Generate summary statistics for resampling techniques.
    
    Args:
        results (dict): Dictionary containing results for all models and resampling methods
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    summary_data = []
    
    for model_name in results:
        for resampling_method in results[model_name]:
            if results[model_name][resampling_method]:
                metrics = results[model_name][resampling_method]['final_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Resampling': resampling_method,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC AUC': metrics['roc_auc']
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate mean and std for each resampling method
    resampling_stats = summary_df.groupby('Resampling').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1-Score': ['mean', 'std'],
        'ROC AUC': ['mean', 'std']
    })
    
    return summary_df, resampling_stats

def train_client_parallel(client: FederatedClient, n_epochs: int) -> Dict[str, Any]:
    """
    Train a single client in parallel.
    
    Args:
        client: FederatedClient object
        n_epochs: Number of epochs to train
        
    Returns:
        Dict containing training results
    """
    try:
        for _ in range(n_epochs):
            client.train_epoch()
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error training client: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def print_round_metrics(round_num: int, metrics: Dict[str, float], training_time: float):
    """
    Print detailed performance metrics for a round.
    """
    print("\n" + "="*50)
    print(f"Round {round_num} Performance Metrics:")
    print("="*50)
    print(f"{'Metric':<15} {'Value':<10} {'Interpretation':<30}")
    print("-"*50)
    
    # Accuracy
    print(f"{'Accuracy':<15} {metrics['accuracy']:.4f}    {'Higher is better'}")
    
    # Precision
    print(f"{'Precision':<15} {metrics['precision']:.4f}    {'Higher is better'}")
    
    # Recall
    print(f"{'Recall':<15} {metrics['recall']:.4f}    {'Higher is better'}")
    
    # F1 Score
    print(f"{'F1-Score':<15} {metrics['f1']:.4f}    {'Higher is better'}")
    
    # ROC AUC
    print(f"{'ROC AUC':<15} {metrics['roc_auc']:.4f}    {'Higher is better'}")
    
    print("-"*50)
    print(f"Training Time: {training_time:.2f} seconds")
    print("="*50 + "\n")

def print_model_comparison(comparison_df: pd.DataFrame, best_config: pd.Series):
    """
    Print comparison of models with the best configuration.
    """
    print("\n=== Best Performing Model Configuration ===")
    print("="*50)
    print(f"Model: {best_config['Model']}")
    print(f"Resampling Method: {best_config['Resampling']}")
    print("\nPerformance Metrics:")
    print(f"Accuracy: {best_config['Accuracy']:.4f}")
    print(f"Precision: {best_config['Precision']:.4f}")
    print(f"Recall: {best_config['Recall']:.4f}")
    print(f"F1-Score: {best_config['F1-Score']:.4f}")
    print(f"ROC AUC: {best_config['ROC AUC']:.4f}")
    print(f"Overall Score: {best_config['Overall Score']:.4f}")
    print(f"Training Time: {best_config['Training Time']:.2f} seconds")
    
    print("\nPerformance Comparison with Other Models:")
    print("="*50)
    print(f"{'Model':<20} {'Resampling':<10} {'Overall Score':<15} {'Improvement':<15}")
    print("-"*50)
    
    for _, row in comparison_df.iterrows():
        if row['Model'] != best_config['Model'] or row['Resampling'] != best_config['Resampling']:
            improvement = ((best_config['Overall Score'] - row['Overall Score']) / row['Overall Score']) * 100
            print(f"{row['Model']:<20} {row['Resampling']:<10} {row['Overall Score']:.4f}        {improvement:+.2f}%")

def get_model(model_name: str) -> BaseEstimator:
    """
    Get model with optimized hyperparameters and overfitting prevention.
    """
    if model_name == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=8,  # Reduced from 10
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=4,  # Increased from 2
            class_weight='balanced',
            random_state=42
        )
    elif model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=10,  # Reduced from 15
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=4,  # Increased from 2
            class_weight='balanced',
            max_features='sqrt',  # Added to prevent overfitting
            bootstrap=True,  # Added for better generalization
            n_jobs=-1,
            random_state=42
        )
    elif model_name == 'logistic_regression':
        return LogisticRegression(
            C=0.1,  # Reduced from 1.0 for stronger regularization
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
    elif model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=5,  # Reduced from 7
            learning_rate=0.05,  # Reduced from 0.1
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,  # Added to prevent overfitting
            gamma=0.1,  # Added for regularization
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            scale_pos_weight=1,
            n_jobs=-1,
            random_state=42
        )
    elif model_name == 'lightgbm':
        return LGBMClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=5,  # Reduced from 7
            learning_rate=0.05,  # Reduced from 0.1
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,  # Added to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_evaluate_model(
    model_name: str,
    resampling_method: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 2,
    n_rounds: int = 2,
    n_epochs: int = 2,
    max_workers: int = 4,
    early_stopping_patience: int = 3
) -> Dict[str, Any]:
    """
    Train and evaluate a single model using federated learning with overfitting prevention.
    """
    start_time = time.time()
    logger.info(f"Starting training of {model_name.upper()} with {resampling_method.upper()} resampling")
    
    # Split training data into train and validation sets
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create federated data
    logger.info("Creating federated data...")
    client_data = create_federated_data(X_train_main, y_train_main, n_clients)
    
    # Initialize models and clients
    logger.info("Initializing models and clients...")
    global_model = get_model(model_name)
    server = FederatedServer(global_model, n_clients)
    
    # Initialize clients with resampling
    for i, (X_client, y_client) in enumerate(client_data):
        logger.info(f"Setting up client {i+1}/{n_clients}...")
        resampler = ResamplingTechnique(resampling_method)
        X_resampled, y_resampled = resampler.resample(X_client, y_client)
        
        client_model = get_model(model_name)
        client = FederatedClient(client_model, X_resampled, y_resampled)
        server.add_client(client)
    
    # Store results for each round
    round_results = []
    best_metric = float('-inf')
    no_improvement_count = 0
    
    # Train federated model with early stopping and validation
    logger.info("Starting federated training...")
    for round in range(n_rounds):
        round_start = time.time()
        logger.info(f"\nRound {round + 1}/{n_rounds}")
        
        # Train clients in parallel with gradient accumulation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            train_func = partial(train_client_parallel, n_epochs=n_epochs)
            future_to_client = {executor.submit(train_func, client): client for client in server.clients}
            
            for future in as_completed(future_to_client):
                result = future.result()
                if result['status'] == 'error':
                    logger.error(f"Client training failed: {result['error']}")
        
        # Aggregate weights
        server.aggregate_weights()
        
        # Evaluate on validation set first
        try:
            y_val_pred = server.model.predict(X_val)
            y_val_prob = server.model.predict_proba(X_val)[:, 1]
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            # Check for overfitting
            if round > 0:
                prev_val_metrics = round_results[-1]['val_metrics']
                if val_metrics['f1'] < prev_val_metrics['f1'] * 0.95:  # 5% degradation threshold
                    logger.warning("Potential overfitting detected on validation set")
                    # Reduce learning rate or stop training
                    break
            
            # Evaluate on test set
            y_test_pred = server.model.predict(X_test)
            y_test_prob = server.model.predict_proba(X_test)[:, 1]
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            # Store results
            round_result = {
                'round': round + 1,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': time.time() - round_start
            }
            round_results.append(round_result)
            
            # Print detailed metrics for this round
            print_round_metrics(round + 1, test_metrics, round_result['training_time'])
            
            # Early stopping check with multiple metrics
            current_metric = (test_metrics['f1'] + test_metrics['precision']) / 2
            if current_metric > best_metric:
                best_metric = current_metric
                no_improvement_count = 0
                # Save best model
                joblib.dump(server.model, f'best_{model_name}_{resampling_method}_model.joblib')
                logger.info(f"New best model saved with combined metric: {best_metric:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} rounds")
                
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {round + 1} rounds")
                break
                
        except Exception as e:
            logger.error(f"Error in round {round + 1}: {str(e)}")
            continue
    
    # Final evaluation and visualization
    logger.info("Performing final evaluation...")
    try:
        # Load best model
        best_model = joblib.load(f'best_{model_name}_{resampling_method}_model.joblib')
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate final metrics
        final_metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Print final metrics
        print_round_metrics("Final", final_metrics, time.time() - start_time)
        
        # Plot curves
        plot_precision_recall_curve(y_test, y_prob, f'{model_name.upper()} ({resampling_method.upper()}) Precision-Recall Curve')
        plot_roc_curve(y_test, y_prob, f'{model_name.upper()} ({resampling_method.upper()}) ROC Curve')
        
        # Save detailed results
        results_df = pd.DataFrame(round_results)
        results_df.to_csv(f'{model_name}_{resampling_method}_results.csv', index=False)
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
        return {
            'final_metrics': final_metrics,
            'round_results': round_results,
            'training_time': total_time
        }
    except Exception as e:
        logger.error(f"Error in final evaluation: {str(e)}")
        return None

def analyze_model_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Analyze and compare results across all models and resampling methods.
    
    Args:
        results: Dictionary containing results for all models and resampling methods
    """
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name in results:
        for resampling_method in results[model_name]:
            if results[model_name][resampling_method]:
                metrics = results[model_name][resampling_method]['final_metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Resampling': resampling_method,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC AUC': metrics['roc_auc'],
                    'Training Time': results[model_name][resampling_method]['training_time']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print overall comparison
    print("\n=== Overall Model Comparison ===")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    
    # Find best model for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    print("\n=== Best Models by Metric ===")
    print("=" * 100)
    for metric in metrics:
        best_model = comparison_df.loc[comparison_df[metric].idxmax()]
        print(f"\nBest {metric}:")
        print(f"Model: {best_model['Model']}")
        print(f"Resampling: {best_model['Resampling']}")
        print(f"Score: {best_model[metric]:.4f}")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model Comparison by Metric
    plt.subplot(2, 1, 1)
    comparison_df.groupby('Model')[metrics].mean().plot(kind='bar')
    plt.title('Average Performance by Model')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Resampling Method Comparison
    plt.subplot(2, 1, 2)
    comparison_df.groupby('Resampling')[metrics].mean().plot(kind='bar')
    plt.title('Average Performance by Resampling Method')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Save detailed comparison to CSV
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    
    return comparison_df

def analyze_best_performance(comparison_df: pd.DataFrame) -> None:
    """
    Analyze and report the best performing model configuration in detail.
    """
    # Calculate overall score (weighted average of all metrics)
    weights = {
        'Accuracy': 0.2,
        'Precision': 0.2,
        'Recall': 0.2,
        'F1-Score': 0.2,
        'ROC AUC': 0.2
    }
    
    comparison_df['Overall Score'] = sum(
        comparison_df[metric] * weight 
        for metric, weight in weights.items()
    )
    
    # Find best overall configuration
    best_config = comparison_df.loc[comparison_df['Overall Score'].idxmax()]
    
    # Print detailed comparison
    print_model_comparison(comparison_df, best_config)
    
    # Create detailed performance plot
    plt.figure(figsize=(12, 6))
    
    # Plot metrics for best configuration
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    values = [best_config[metric] for metric in metrics]
    
    plt.bar(metrics, values)
    plt.title(f'Performance Metrics for Best Model ({best_config["Model"]} with {best_config["Resampling"]})')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('best_model_performance.png')
    plt.close()
    
    # Save best configuration details
    best_config_dict = {
        'model': best_config['Model'],
        'resampling_method': best_config['Resampling'],
        'metrics': {
            'accuracy': float(best_config['Accuracy']),
            'precision': float(best_config['Precision']),
            'recall': float(best_config['Recall']),
            'f1_score': float(best_config['F1-Score']),
            'roc_auc': float(best_config['ROC AUC'])
        },
        'training_time': float(best_config['Training Time']),
        'overall_score': float(best_config['Overall Score'])
    }
    
    with open('best_model_config.json', 'w') as f:
        json.dump(best_config_dict, f, indent=4)
    
    return best_config_dict

def main():
    # Configuration
    DATA_PATH = 'data/creditcard.csv'
    N_CLIENTS = 2    # Optimal: 2 clients for better model diversity
    N_ROUNDS = 1     # Optimal: 1 round is sufficient
    N_EPOCHS = 1     # Optimal: 1 epoch is sufficient
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    
    print("\nOriginal class distribution:")
    plot_class_distribution(y_train, 'Original Training Data Distribution')
    
    # List of all models to train
    models = [
        'decision_tree',
        'random_forest',
        'logistic_regression',
        'xgboost',
        'lightgbm'
    ]
    
    # All resampling methods for comparative analysis
    resampling_methods = ['smote', 'adasyn', 'random']
    
    # Initialize performance log with empty lists
    performance_log = {
        'model': [],
        'resampling': [],
        'training_time': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    # Train and evaluate each model with each resampling method
    results = {}
    total_start_time = time.time()
    
    for model_name in models:
        results[model_name] = {}
        for resampling_method in resampling_methods:
            print(f"\nTraining model {model_name} with {resampling_method} resampling")
            start_time = time.time()
            
            model_results = train_and_evaluate_model(
                model_name, resampling_method, X_train, X_test, y_train, y_test,
                n_clients=N_CLIENTS, n_rounds=N_ROUNDS, n_epochs=N_EPOCHS
            )
            
            if model_results:
                training_time = time.time() - start_time
                metrics = model_results['final_metrics']
                
                # Log all metrics
                performance_log['model'].append(model_name)
                performance_log['resampling'].append(resampling_method)
                performance_log['training_time'].append(training_time)
                performance_log['accuracy'].append(metrics['accuracy'])
                performance_log['precision'].append(metrics['precision'])
                performance_log['recall'].append(metrics['recall'])
                performance_log['f1_score'].append(metrics['f1'])
                performance_log['roc_auc'].append(metrics['roc_auc'])
                
                print(f"\nPerformance Summary for {model_name} with {resampling_method}:")
                print(f"Training Time: {training_time:.2f} seconds")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
            results[model_name][resampling_method] = model_results
    
    total_time = time.time() - total_start_time
    print(f"\nTotal training time for all models and resampling methods: {total_time:.2f} seconds")
    
    # Create performance DataFrame
    performance_df = pd.DataFrame(performance_log)
    
    # Save performance log
    performance_df.to_csv('performance_log.csv', index=False)
    
    # Analyze and compare results
    comparison_df = analyze_model_comparison(results)
    
    # Analyze best performance
    best_config = analyze_best_performance(comparison_df)
    
    print("\nTraining completed. Results saved in:")
    print("- performance_log.csv")
    print("- best_model_performance.png")
    print("- model_comparison_results.csv")
    print("- best_model_config.json")

if __name__ == '__main__':
    main() 