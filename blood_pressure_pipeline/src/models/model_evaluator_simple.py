#!/usr/bin/env python3
"""
Simple Model Evaluator for M3 Pro Pipeline
Evaluates models with clinical validation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "BP") -> Dict[str, float]:
    """Calculate clinical validation metrics for blood pressure prediction."""
    
    # Basic regression metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Clinical accuracy metrics
    errors = np.abs(y_pred - y_true)
    
    # Percentage within thresholds
    within_5 = np.mean(errors <= 5) * 100
    within_10 = np.mean(errors <= 10) * 100
    within_15 = np.mean(errors <= 15) * 100
    
    # BHS (British Hypertension Society) Grade
    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        bhs_grade = "A"
    elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        bhs_grade = "B"
    elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        bhs_grade = "C"
    else:
        bhs_grade = "D"
    
    # AAMI (Association for the Advancement of Medical Instrumentation) Standard
    # Mean error should be ≤ 5 mmHg and std ≤ 8 mmHg
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    aami_pass = mean_error <= 5 and std_error <= 8
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_error': mean_error,
        'std_error': std_error,
        'within_5_mmhg': within_5,
        'within_10_mmhg': within_10,
        'within_15_mmhg': within_15,
        'bhs_grade': bhs_grade,
        'aami_pass': aami_pass
    }


def evaluate_single_model(model_result: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray, 
                         model_name: str, target: str) -> Dict[str, Any]:
    """Evaluate a single model."""
    
    if 'model' not in model_result:
        logger.warning(f"No model found in {model_name} results")
        return {}
    
    model = model_result['model']
    
    try:
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
        else:
            logger.warning(f"Model {model_name} doesn't have predict method")
            return {}
        
        # Calculate metrics
        metrics = calculate_clinical_metrics(y_val, y_pred, target.upper())
        
        # Add model-specific info
        metrics.update({
            'model_name': model_name,
            'target': target,
            'n_samples': len(y_val),
            'training_time': model_result.get('training_time', 0)
        })
        
        logger.info(f"{model_name} {target.upper()} - R²: {metrics['r2']:.4f}, "
                   f"RMSE: {metrics['rmse']:.2f}, BHS: {metrics['bhs_grade']}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return {}


def evaluate_models(training_results: Dict[str, Any], target: str = "sbp") -> Dict[str, Any]:
    """
    Evaluate models with clinical validation.
    
    Args:
        training_results: Results from train_enhanced_models_m3_pro
        target: 'sbp' or 'dbp'
    
    Returns:
        Dictionary containing evaluation results
    """
    
    logger.info(f"Evaluating {target.upper()} models with clinical validation...")
    
    # Get data splits
    if 'data_splits' not in training_results:
        logger.error("No data splits found in training results")
        return {'error': 'No data splits available'}
    
    if target not in training_results['data_splits']:
        logger.error(f"No data splits found for {target}")
        return {'error': f'No data splits for {target}'}
    
    splits = training_results['data_splits'][target]
    X_val = splits['X_val']
    y_val = splits['y_val']
    
    # Get models
    models_key = f'{target}_models'
    if models_key not in training_results:
        logger.error(f"No models found for {target}")
        return {'error': f'No models found for {target}'}
    
    models = training_results[models_key]
    
    # Evaluate each model
    evaluation_results = {}
    all_metrics = []
    
    for model_name, model_result in models.items():
        metrics = evaluate_single_model(model_result, X_val, y_val, model_name, target)
        if metrics:
            evaluation_results[model_name] = metrics
            all_metrics.append(metrics)
    
    # Create comparison table
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        # Sort by R² score (descending)
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        
        # Create summary report
        best_model = comparison_df.iloc[0]
        
        report = f"""
{target.upper()} Model Evaluation Report
={'=' * 40}

Best Model: {best_model['model_name']}
Performance Metrics:
- R² Score: {best_model['r2']:.4f}
- RMSE: {best_model['rmse']:.2f} mmHg
- MAE: {best_model['mae']:.2f} mmHg
- Mean Error: {best_model['mean_error']:.2f} mmHg
- Std Error: {best_model['std_error']:.2f} mmHg

Clinical Validation:
- Within ±5 mmHg: {best_model['within_5_mmhg']:.1f}%
- Within ±10 mmHg: {best_model['within_10_mmhg']:.1f}%
- Within ±15 mmHg: {best_model['within_15_mmhg']:.1f}%
- BHS Grade: {best_model['bhs_grade']}
- AAMI Standard: {'PASS' if best_model['aami_pass'] else 'FAIL'}

Training Time: {best_model['training_time']:.1f} seconds
Validation Samples: {best_model['n_samples']}

All Models Performance:
"""
        
        for _, row in comparison_df.iterrows():
            report += f"\n{row['model_name']}:"
            report += f"\n  R²: {row['r2']:.4f}, RMSE: {row['rmse']:.2f}, BHS: {row['bhs_grade']}, AAMI: {'PASS' if row['aami_pass'] else 'FAIL'}"
        
        evaluation_results['_summary'] = {
            'comparison_table': comparison_df,
            'report': report,
            'best_model': best_model['model_name'],
            'best_r2': best_model['r2'],
            'best_rmse': best_model['rmse'],
            'best_bhs_grade': best_model['bhs_grade']
        }
        
        logger.info(f"✅ {target.upper()} evaluation completed. Best model: {best_model['model_name']} "
                   f"(R²: {best_model['r2']:.4f}, BHS: {best_model['bhs_grade']})")
    else:
        logger.warning(f"No models could be evaluated for {target}")
        evaluation_results['_summary'] = {
            'comparison_table': pd.DataFrame(),
            'report': f"No models could be evaluated for {target}",
            'best_model': None,
            'best_r2': 0,
            'best_rmse': float('inf'),
            'best_bhs_grade': 'D'
        }
    
    return evaluation_results


if __name__ == "__main__":
    # Test the evaluator
    print("🧪 Testing Simple Model Evaluator...")
    
    # Create synthetic evaluation data
    np.random.seed(42)
    n_samples = 200
    y_true = 120 + 15 * np.random.randn(n_samples)
    y_pred = y_true + 5 * np.random.randn(n_samples)  # Add some noise
    
    metrics = calculate_clinical_metrics(y_true, y_pred, "SBP")
    
    print(f"✅ Clinical metrics calculated:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f} mmHg")
    print(f"  BHS Grade: {metrics['bhs_grade']}")
    print(f"  AAMI Pass: {metrics['aami_pass']}")
    print(f"  Within ±5 mmHg: {metrics['within_5_mmhg']:.1f}%")
