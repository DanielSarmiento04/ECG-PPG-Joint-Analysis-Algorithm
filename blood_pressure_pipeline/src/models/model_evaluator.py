"""
Enhanced model evaluation with clinical validation metrics and comprehensive reporting.
Addresses the need for medical-standard evaluation criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ClinicalMetrics:
    """Medical-standard evaluation metrics for blood pressure prediction."""
    
    def __init__(self):
        # Clinical thresholds based on medical standards
        self.bhs_thresholds = {
            'grade_a': {'within_5': 60, 'within_10': 85, 'within_15': 95},
            'grade_b': {'within_5': 50, 'within_10': 75, 'within_15': 90},
            'grade_c': {'within_5': 40, 'within_10': 65, 'within_15': 85}
        }
        
        self.aami_limits = {
            'mean_error': 5,     # Mean error ≤ ±5 mmHg
            'std_error': 8       # Standard deviation ≤ 8 mmHg
        }
    
    def calculate_error_percentiles(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate percentage of predictions within error thresholds."""
        errors = np.abs(y_pred - y_true)
        
        return {
            'within_5': np.mean(errors <= 5) * 100,
            'within_10': np.mean(errors <= 10) * 100,
            'within_15': np.mean(errors <= 15) * 100,
            'within_20': np.mean(errors <= 20) * 100
        }
    
    def british_hypertension_society_grade(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Calculate BHS grading (A, B, C, or D)."""
        percentiles = self.calculate_error_percentiles(y_true, y_pred)
        
        # Check for Grade A
        if (percentiles['within_5'] >= self.bhs_thresholds['grade_a']['within_5'] and
            percentiles['within_10'] >= self.bhs_thresholds['grade_a']['within_10'] and
            percentiles['within_15'] >= self.bhs_thresholds['grade_a']['within_15']):
            return 'A'
        
        # Check for Grade B
        elif (percentiles['within_5'] >= self.bhs_thresholds['grade_b']['within_5'] and
              percentiles['within_10'] >= self.bhs_thresholds['grade_b']['within_10'] and
              percentiles['within_15'] >= self.bhs_thresholds['grade_b']['within_15']):
            return 'B'
        
        # Check for Grade C
        elif (percentiles['within_5'] >= self.bhs_thresholds['grade_c']['within_5'] and
              percentiles['within_10'] >= self.bhs_thresholds['grade_c']['within_10'] and
              percentiles['within_15'] >= self.bhs_thresholds['grade_c']['within_15']):
            return 'C'
        
        # Grade D (does not meet minimum requirements)
        else:
            return 'D'
    
    def aami_validation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """AAMI (Association for the Advancement of Medical Instrumentation) validation."""
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'mean_error_passes': abs(mean_error) <= self.aami_limits['mean_error'],
            'std_error_passes': std_error <= self.aami_limits['std_error'],
            'overall_passes': (abs(mean_error) <= self.aami_limits['mean_error'] and 
                             std_error <= self.aami_limits['std_error'])
        }
    
    def bland_altman_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Bland-Altman analysis for agreement assessment."""
        differences = y_pred - y_true
        means = (y_pred + y_true) / 2
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Limits of agreement (95% limits)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        # Calculate percentage within limits
        within_loa = np.mean((differences >= lower_loa) & (differences <= upper_loa)) * 100
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'upper_loa': upper_loa,
            'lower_loa': lower_loa,
            'within_loa_percent': within_loa,
            'loa_width': upper_loa - lower_loa
        }


class ModelEvaluator:
    """Comprehensive model evaluation with clinical validation."""
    
    def __init__(self):
        self.clinical_metrics = ClinicalMetrics()
    
    def evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive evaluation of a single model."""
        
        # Basic regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation
        correlation, p_value = pearsonr(y_true, y_pred)
        
        # Clinical metrics
        error_percentiles = self.clinical_metrics.calculate_error_percentiles(y_true, y_pred)
        bhs_grade = self.clinical_metrics.british_hypertension_society_grade(y_true, y_pred)
        aami_results = self.clinical_metrics.aami_validation(y_true, y_pred)
        bland_altman = self.clinical_metrics.bland_altman_analysis(y_true, y_pred)
        
        # Additional metrics
        errors = y_pred - y_true
        bias = np.mean(errors)
        precision = np.std(errors)
        
        return {
            'model_name': model_name,
            'basic_metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'correlation_p_value': p_value
            },
            'clinical_metrics': {
                'error_percentiles': error_percentiles,
                'bhs_grade': bhs_grade,
                'aami_validation': aami_results,
                'bland_altman': bland_altman
            },
            'error_analysis': {
                'bias': bias,
                'precision': precision,
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'error_range': np.ptp(errors)
            }
        }
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]], 
                      target_name: str = "BP") -> pd.DataFrame:
        """Compare multiple models and create ranking table."""
        
        comparison_data = []
        
        for model_name, results in results_dict.items():
            basic = results['basic_metrics']
            clinical = results['clinical_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'RMSE': basic['rmse'],
                'MAE': basic['mae'],
                'R²': basic['r2'],
                'Correlation': basic['correlation'],
                'Within ±5 mmHg (%)': clinical['error_percentiles']['within_5'],
                'Within ±10 mmHg (%)': clinical['error_percentiles']['within_10'],
                'Within ±15 mmHg (%)': clinical['error_percentiles']['within_15'],
                'BHS Grade': clinical['bhs_grade'],
                'AAMI Pass': clinical['aami_validation']['overall_passes'],
                'Bias': results['error_analysis']['bias'],
                'Precision': results['error_analysis']['precision']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by R² (descending) and then by RMSE (ascending)
        df = df.sort_values(['R²', 'RMSE'], ascending=[False, True])
        
        return df
    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str, target_name: str, 
                              save_path: Optional[str] = None) -> None:
        """Create comprehensive evaluation plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - {target_name} Prediction Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Prediction vs Actual scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_true, p(y_true), 'b-', alpha=0.8, label=f'Regression Line')
        
        ax1.set_xlabel(f'Actual {target_name} (mmHg)')
        ax1.set_ylabel(f'Predicted {target_name} (mmHg)')
        ax1.set_title('Prediction vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and display metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\\nRMSE = {rmse:.2f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Bland-Altman plot
        ax2 = axes[0, 1]
        differences = y_pred - y_true
        means = (y_pred + y_true) / 2
        
        ax2.scatter(means, differences, alpha=0.6, s=30)
        
        # Calculate limits of agreement
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff
        
        # Add horizontal lines
        ax2.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean Diff: {mean_diff:.2f}')
        ax2.axhline(upper_loa, color='red', linestyle='--', linewidth=1, label=f'Upper LoA: {upper_loa:.2f}')
        ax2.axhline(lower_loa, color='red', linestyle='--', linewidth=1, label=f'Lower LoA: {lower_loa:.2f}')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel(f'Mean of Actual and Predicted {target_name} (mmHg)')
        ax2.set_ylabel(f'Difference (Predicted - Actual) {target_name} (mmHg)')
        ax2.set_title('Bland-Altman Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution histogram
        ax3 = axes[1, 0]
        errors = y_pred - y_true
        ax3.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax3.set_xlabel(f'Prediction Error {target_name} (mmHg)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Clinical accuracy plot
        ax4 = axes[1, 1]
        thresholds = [5, 10, 15, 20]
        percentages = [np.mean(np.abs(errors) <= t) * 100 for t in thresholds]
        
        bars = ax4.bar([f'±{t}' for t in thresholds], percentages, 
                      color=['green', 'orange', 'blue', 'red'], alpha=0.7)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Add clinical threshold lines
        ax4.axhline(60, color='green', linestyle='--', alpha=0.7, label='BHS Grade A (≥60% at ±5)')
        ax4.axhline(85, color='orange', linestyle='--', alpha=0.7, label='BHS Grade A (≥85% at ±10)')
        
        ax4.set_xlabel('Error Threshold (mmHg)')
        ax4.set_ylabel('Percentage of Predictions (%)')
        ax4.set_title('Clinical Accuracy')
        ax4.set_ylim(0, 105)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results_dict: Dict[str, Dict[str, Any]], 
                                 target_name: str = "BP") -> str:
        """Generate a comprehensive evaluation report."""
        
        report = f"""
=== {target_name.upper()} PREDICTION MODEL EVALUATION REPORT ===

Model Performance Summary:
{'-' * 50}
"""
        
        # Create comparison table
        comparison_df = self.compare_models(results_dict, target_name)
        
        # Add top performers
        report += f"\nTOP PERFORMING MODELS:\n"
        report += f"{'Rank':<5} {'Model':<20} {'R²':<8} {'RMSE':<8} {'±5mmHg%':<10} {'BHS Grade':<10}\n"
        report += f"{'-' * 70}\n"
        
        for i, (_, row) in enumerate(comparison_df.head(5).iterrows(), 1):
            report += f"{i:<5} {row['Model']:<20} {row['R²']:<8.3f} {row['RMSE']:<8.2f} {row['Within ±5 mmHg (%)']:<10.1f} {row['BHS Grade']:<10}\n"
        
        # Clinical validation summary
        report += f"\n\nCLINICAL VALIDATION SUMMARY:\n"
        report += f"{'-' * 50}\n"
        
        for model_name, results in results_dict.items():
            clinical = results['clinical_metrics']
            aami = clinical['aami_validation']
            
            report += f"\n{model_name}:\n"
            report += f"  • BHS Grade: {clinical['bhs_grade']}\n"
            report += f"  • AAMI Standard: {'PASS' if aami['overall_passes'] else 'FAIL'}\n"
            report += f"    - Mean Error: {aami['mean_error']:.2f} mmHg (limit: ±5)\n"
            report += f"    - Std Error: {aami['std_error']:.2f} mmHg (limit: ≤8)\n"
            report += f"  • Clinical Accuracy:\n"
            report += f"    - Within ±5 mmHg: {clinical['error_percentiles']['within_5']:.1f}%\n"
            report += f"    - Within ±10 mmHg: {clinical['error_percentiles']['within_10']:.1f}%\n"
            report += f"    - Within ±15 mmHg: {clinical['error_percentiles']['within_15']:.1f}%\n"
        
        # Recommendations
        report += f"\n\nRECOMMENDATIONS:\n"
        report += f"{'-' * 50}\n"
        
        best_model = comparison_df.iloc[0]
        
        if best_model['BHS Grade'] in ['A', 'B']:
            report += f"✅ {best_model['Model']} shows excellent clinical performance (BHS Grade {best_model['BHS Grade']})\n"
        elif best_model['BHS Grade'] == 'C':
            report += f"⚠️  {best_model['Model']} shows acceptable performance but could be improved\n"
        else:
            report += f"❌ {best_model['Model']} does not meet clinical standards (Grade D)\n"
        
        if best_model['AAMI Pass']:
            report += f"✅ {best_model['Model']} meets AAMI standards\n"
        else:
            report += f"❌ {best_model['Model']} does not meet AAMI standards\n"
        
        if best_model['R²'] >= 0.7:
            report += f"✅ Strong predictive performance (R² = {best_model['R²']:.3f})\n"
        elif best_model['R²'] >= 0.5:
            report += f"⚠️  Moderate predictive performance (R² = {best_model['R²']:.3f})\n"
        else:
            report += f"❌ Weak predictive performance (R² = {best_model['R²']:.3f})\n"
        
        return report


def evaluate_models(training_results: Dict[str, Any], target: str = "sbp") -> Dict[str, Any]:
    """
    Main entry point for model evaluation.
    
    Args:
        training_results: Results from model training
        target: Target variable ('sbp' or 'dbp')
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Extract test data
    data_splits = training_results['data_splits']
    X_test = data_splits['X_test']
    y_test = data_splits[f'y_test_{target}']
    
    # Get models
    models_key = f'{target}_models'
    ensemble_key = f'{target}_ensemble'
    
    evaluation_results = {}
    
    # Evaluate individual models
    if models_key in training_results:
        for model_name, model_result in training_results[models_key].items():
            model = model_result['model']
            y_pred = model.predict(X_test)
            
            eval_result = evaluator.evaluate_single_model(
                y_test, y_pred, model_name
            )
            evaluation_results[model_name] = eval_result
    
    # Evaluate ensemble models
    if ensemble_key in training_results:
        for ensemble_name, ensemble_model in training_results[ensemble_key].items():
            y_pred = ensemble_model.predict(X_test)
            
            eval_result = evaluator.evaluate_single_model(
                y_test, y_pred, f"{ensemble_name}_ensemble"
            )
            evaluation_results[f"{ensemble_name}_ensemble"] = eval_result
    
    # Generate comparison report
    if evaluation_results:
        comparison_df = evaluator.compare_models(evaluation_results, target.upper())
        report = evaluator.generate_evaluation_report(evaluation_results, target.upper())
        
        evaluation_results['_summary'] = {
            'comparison_table': comparison_df,
            'report': report
        }
    
    return evaluation_results
