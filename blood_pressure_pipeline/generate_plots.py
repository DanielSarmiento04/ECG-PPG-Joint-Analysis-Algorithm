#!/usr/bin/env python3
"""
M3 Pro Results Visualization Script
Generates comprehensive plots and visualizations from saved models and evaluation results.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from typing import Dict, Any, List, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set matplotlib style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_saved_models(models_dir: Union[str, Path] = "models") -> Dict[str, Any]:
    """Load all saved M3 Pro models."""
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return {}
    
    saved_models = {}
    
    # Find all M3 Pro model files
    model_files = list(models_dir.glob("*_m3_pro.pkl"))
    
    for model_file in model_files:
        try:
            model_data = joblib.load(model_file)
            model_name = model_file.stem
            saved_models[model_name] = model_data
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_file}: {e}")
    
    return saved_models

def load_evaluation_reports(reports_dir: Union[str, Path] = "reports") -> Dict[str, Any]:
    """Load evaluation reports."""
    reports_dir = Path(reports_dir)
    
    if not reports_dir.exists():
        logger.warning(f"Reports directory not found: {reports_dir}")
        return {}
    
    reports = {}
    
    # Load CSV comparison files
    csv_files = list(reports_dir.glob("*_m3_pro_comparison.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            target = csv_file.stem.split('_')[0]  # Extract 'sbp' or 'dbp'
            reports[target] = df
            logger.info(f"Loaded evaluation report for {target.upper()}")
        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")
    
    return reports

def generate_model_performance_plots(models: Dict[str, Any], output_dir: Union[str, Path] = "plots"):
    """Generate model performance comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract performance data
    sbp_models = {}
    dbp_models = {}
    
    for model_name, model_data in models.items():
        try:
            # Skip if not a dictionary
            if not isinstance(model_data, dict):
                logger.warning(f"Skipping {model_name}: not a dictionary")
                continue
                
            if 'performance' in model_data and isinstance(model_data['performance'], dict):
                perf = model_data['performance']
                
                if '_sbp_' in model_name:
                    clean_name = model_name.replace('_sbp_m3_pro', '')
                    sbp_models[clean_name] = perf
                elif '_dbp_' in model_name:
                    clean_name = model_name.replace('_dbp_m3_pro', '')
                    dbp_models[clean_name] = perf
            else:
                logger.warning(f"Skipping {model_name}: no performance data")
                
        except Exception as e:
            logger.warning(f"Error processing {model_name}: {e}")
            continue
    
    # Plot SBP model performance
    if sbp_models:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SBP Model Performance Comparison (M3 Pro Optimized)', fontsize=16, fontweight='bold')
        
        # R² scores
        models_names = list(sbp_models.keys())
        r2_scores = [sbp_models[m].get('val_r2', 0) for m in models_names]
        
        axes[0, 0].bar(models_names, r2_scores, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE scores
        rmse_scores = [sbp_models[m].get('val_rmse', 0) for m in models_names]
        
        axes[0, 1].bar(models_names, rmse_scores, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('RMSE (Lower is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE (mmHg)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(rmse_scores):
            axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        training_times = [sbp_models[m].get('training_time', 0) / 60 for m in models_names]  # Convert to minutes
        
        axes[1, 0].bar(models_names, training_times, color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Training Time (Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('Training Time (minutes)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(training_times):
            axes[1, 0].text(i, v + max(training_times) * 0.01, f'{v:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # Performance scatter plot
        axes[1, 1].scatter(rmse_scores, r2_scores, s=100, alpha=0.7, c=['red', 'blue', 'green'][:len(models_names)])
        for i, name in enumerate(models_names):
            axes[1, 1].annotate(name, (rmse_scores[i], r2_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=10)
        axes[1, 1].set_xlabel('RMSE (mmHg)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Performance Trade-off (Top-left is best)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sbp_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Generated SBP model performance plots")
    
    # Plot DBP model performance (similar structure)
    if dbp_models:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DBP Model Performance Comparison (M3 Pro Optimized)', fontsize=16, fontweight='bold')
        
        models_names = list(dbp_models.keys())
        r2_scores = [dbp_models[m].get('val_r2', 0) for m in models_names]
        rmse_scores = [dbp_models[m].get('val_rmse', 0) for m in models_names]
        training_times = [dbp_models[m].get('training_time', 0) / 60 for m in models_names]
        
        # R² scores
        axes[0, 0].bar(models_names, r2_scores, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE scores
        axes[0, 1].bar(models_names, rmse_scores, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('RMSE (Lower is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE (mmHg)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(rmse_scores):
            axes[0, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Training times
        axes[1, 0].bar(models_names, training_times, color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Training Time (Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('Training Time (minutes)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(training_times):
            axes[1, 0].text(i, v + max(training_times) * 0.01, f'{v:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # Performance scatter
        axes[1, 1].scatter(rmse_scores, r2_scores, s=100, alpha=0.7, c=['red', 'blue', 'green'][:len(models_names)])
        for i, name in enumerate(models_names):
            axes[1, 1].annotate(name, (rmse_scores[i], r2_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=10)
        axes[1, 1].set_xlabel('RMSE (mmHg)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Performance Trade-off (Top-left is best)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dbp_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Generated DBP model performance plots")

def generate_clinical_validation_plots(reports: Dict[str, Any], output_dir: Union[str, Path] = "plots"):
    """Generate clinical validation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not reports:
        logger.warning("No evaluation reports found for clinical validation plots")
        return
    
    # Clinical accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Clinical Validation Results (M3 Pro Optimized)', fontsize=16, fontweight='bold')
    
    targets = ['sbp', 'dbp']
    colors = ['skyblue', 'lightcoral']
    
    for idx, target in enumerate(targets):
        if target not in reports:
            continue
            
        df = reports[target]
        if df.empty:
            continue
        
        # BHS Grade distribution
        bhs_counts = df['bhs_grade'].value_counts()
        axes[0, idx].pie(bhs_counts.values, labels=bhs_counts.index, autopct='%1.1f%%', 
                        colors=['green', 'yellow', 'orange', 'red'][:len(bhs_counts)])
        axes[0, idx].set_title(f'{target.upper()} BHS Grade Distribution', fontweight='bold')
        
        # Clinical accuracy thresholds
        models = df['model_name']
        within_5 = df['within_5_mmhg']
        within_10 = df['within_10_mmhg']
        within_15 = df['within_15_mmhg']
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[1, idx].bar(x - width, within_5, width, label='±5 mmHg', alpha=0.8, color='green')
        axes[1, idx].bar(x, within_10, width, label='±10 mmHg', alpha=0.8, color='blue')
        axes[1, idx].bar(x + width, within_15, width, label='±15 mmHg', alpha=0.8, color='orange')
        
        axes[1, idx].set_xlabel('Models')
        axes[1, idx].set_ylabel('Accuracy (%)')
        axes[1, idx].set_title(f'{target.upper()} Clinical Accuracy', fontweight='bold')
        axes[1, idx].set_xticks(x)
        axes[1, idx].set_xticklabels(models, rotation=45, ha='right')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
        
        # Add BHS Grade thresholds
        axes[1, idx].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='BHS Grade A (±5)')
        axes[1, idx].axhline(y=85, color='red', linestyle='--', alpha=0.7, label='BHS Grade A (±10)')
        axes[1, idx].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='BHS Grade A (±15)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clinical_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Generated clinical validation plots")

def generate_feature_importance_plots(models: Dict[str, Any], output_dir: Union[str, Path] = "plots"):
    """Generate feature importance plots from saved models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract feature importance data
    for model_name, model_data in models.items():
        try:
            # Skip if no model data
            if not isinstance(model_data, dict):
                logger.warning(f"Skipping {model_name}: invalid model data structure")
                continue
            
            # Try to find the actual model object
            model = None
            if 'model' in model_data:
                model = model_data['model']
            elif hasattr(model_data, 'feature_importances_'):
                model = model_data
            else:
                logger.warning(f"Skipping {model_name}: no model object found")
                continue
            
            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # Create feature importance plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get top 20 features
                top_indices = np.argsort(importance)[-20:]
                top_importance = importance[top_indices]
                feature_names = [f'Feature_{i}' for i in top_indices]
                
                # Create horizontal bar plot
                bars = ax.barh(range(len(top_importance)), top_importance, color='skyblue', alpha=0.8)
                ax.set_yticks(range(len(top_importance)))
                ax.set_yticklabels(feature_names)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'Top 20 Feature Importances - {model_name}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Generated feature importance plot for {model_name}")
            else:
                logger.info(f"Skipping {model_name}: no feature_importances_ attribute")
                
        except Exception as e:
            logger.warning(f"Error generating feature importance plot for {model_name}: {e}")
            continue

def generate_hardware_utilization_plot(models: Dict[str, Any], output_dir: Union[str, Path] = "plots"):
    """Generate hardware utilization summary plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract hardware info from models
    hardware_info = None
    training_times = {}
    
    for model_name, model_data in models.items():
        try:
            # Skip if not a dictionary
            if not isinstance(model_data, dict):
                continue
                
            if 'hardware_info' in model_data:
                hardware_info = model_data['hardware_info']
            
            if 'performance' in model_data and isinstance(model_data['performance'], dict):
                perf_data = model_data['performance']
                if 'training_time' in perf_data:
                    training_times[model_name] = perf_data['training_time']
        except Exception as e:
            logger.warning(f"Error extracting data from {model_name}: {e}")
            continue
    
    if not hardware_info or not training_times:
        logger.warning("Insufficient hardware data for utilization plot")
        return
    
    # Create hardware utilization plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('M3 Pro Hardware Utilization Summary', fontsize=16, fontweight='bold')
    
    # CPU utilization
    cpu_used = hardware_info.get('recommended_n_jobs', 0)
    cpu_total = hardware_info.get('cpu_count', 0)
    cpu_unused = max(0, cpu_total - cpu_used)
    
    if cpu_total > 0:
        axes[0, 0].pie([cpu_used, cpu_unused], labels=['Used', 'Available'], 
                       autopct='%1.0f cores', colors=['lightcoral', 'lightgray'])
        axes[0, 0].set_title(f'CPU Core Utilization\n({cpu_used}/{cpu_total} cores)', fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'CPU info unavailable', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('CPU Core Utilization', fontweight='bold')
    
    # Memory info
    memory_gb = hardware_info.get('memory_gb', 0)
    if memory_gb > 0:
        axes[0, 1].bar(['Available Memory'], [memory_gb], color='lightblue', alpha=0.8)
        axes[0, 1].set_ylabel('Memory (GB)')
        axes[0, 1].set_title('Available System Memory', fontweight='bold')
        axes[0, 1].text(0, memory_gb + 0.5, f'{memory_gb:.1f} GB', ha='center', va='bottom', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Memory info unavailable', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Available System Memory', fontweight='bold')
    
    # Training time breakdown
    if training_times:
        model_names = list(training_times.keys())
        times_minutes = [training_times[m] / 60 for m in model_names]
        
        axes[1, 0].bar(range(len(model_names)), times_minutes, color='lightgreen', alpha=0.8)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Training Time (minutes)')
        axes[1, 0].set_title('Training Time per Model', fontweight='bold')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels([m.replace('_m3_pro', '') for m in model_names], rotation=45, ha='right')
        
        # Add time labels
        for i, v in enumerate(times_minutes):
            axes[1, 0].text(i, v + max(times_minutes) * 0.01, f'{v:.1f}m', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'Training time data unavailable', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Time per Model', fontweight='bold')
    
    # Platform info
    platform_info = hardware_info.get('platform', 'Unknown')
    chip_type = hardware_info.get('chip_type', 'Unknown')
    
    axes[1, 1].text(0.5, 0.7, f'Platform: {platform_info}', ha='center', va='center', 
                   transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
    axes[1, 1].text(0.5, 0.5, f'Chip: {chip_type}', ha='center', va='center', 
                   transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
    axes[1, 1].text(0.5, 0.3, f'Optimization: M3 Pro Enabled ✅', ha='center', va='center', 
                   transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', color='green')
    axes[1, 1].set_title('Hardware Configuration', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hardware_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Generated hardware utilization plot")

def generate_summary_report(models: Dict[str, Any], reports: Dict[str, Any], output_dir: Union[str, Path] = "plots"):
    """Generate a summary report with key metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract best performance metrics
    best_metrics = {}
    
    for target in ['sbp', 'dbp']:
        if target in reports and not reports[target].empty:
            df = reports[target]
            best_model = df.loc[df['r2'].idxmax()]
            
            best_metrics[target] = {
                'model': best_model['model_name'],
                'r2': best_model['r2'],
                'rmse': best_model['rmse'],
                'bhs_grade': best_model['bhs_grade'],
                'within_5': best_model['within_5_mmhg'],
                'within_10': best_model['within_10_mmhg'],
                'aami_pass': best_model['aami_pass']
            }
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('M3 Pro Blood Pressure Prediction - Performance Summary', fontsize=18, fontweight='bold')
    
    if best_metrics:
        # R² comparison
        targets = list(best_metrics.keys())
        r2_values = [best_metrics[t]['r2'] for t in targets]
        rmse_values = [best_metrics[t]['rmse'] for t in targets]
        
        bars1 = axes[0, 0].bar(targets, r2_values, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 0].set_title('Best Model R² Score by Target', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        
        for i, (bar, val) in enumerate(zip(bars1, r2_values)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE comparison
        bars2 = axes[0, 1].bar(targets, rmse_values, color=['lightgreen', 'orange'], alpha=0.8)
        axes[0, 1].set_title('Best Model RMSE by Target', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE (mmHg)')
        
        for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + max(rmse_values) * 0.01, 
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Clinical accuracy summary
        within_5_values = [best_metrics[t]['within_5'] for t in targets]
        within_10_values = [best_metrics[t]['within_10'] for t in targets]
        
        x = np.arange(len(targets))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, within_5_values, width, label='±5 mmHg', color='green', alpha=0.8)
        axes[1, 0].bar(x + width/2, within_10_values, width, label='±10 mmHg', color='blue', alpha=0.8)
        
        axes[1, 0].set_xlabel('Target')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Clinical Accuracy - Best Models', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([t.upper() for t in targets])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # BHS Grade summary
        bhs_grades = [best_metrics[t]['bhs_grade'] for t in targets]
        grade_colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red'}
        colors = [grade_colors.get(grade, 'gray') for grade in bhs_grades]
        
        bars3 = axes[1, 1].bar(targets, [1] * len(targets), color=colors, alpha=0.8)
        axes[1, 1].set_title('BHS Grade - Best Models', fontweight='bold')
        axes[1, 1].set_ylabel('BHS Grade')
        axes[1, 1].set_ylim(0, 1.2)
        
        for i, (bar, grade) in enumerate(zip(bars3, bhs_grades)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, 0.5, 
                           grade, ha='center', va='center', fontweight='bold', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Generated performance summary plot")

def main():
    """Main function to generate all plots."""
    print("🎨 M3 Pro Results Visualization Generator")
    print("=" * 50)
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    print("\n📂 Loading saved models...")
    models = load_saved_models()
    
    if not models:
        print("❌ No saved models found. Please run the training pipeline first.")
        return
    
    print(f"   ✅ Loaded {len(models)} models")
    
    print("\n📊 Loading evaluation reports...")
    reports = load_evaluation_reports()
    
    if reports:
        print(f"   ✅ Loaded evaluation reports for {list(reports.keys())}")
    else:
        print("   ⚠️  No evaluation reports found")
    
    print("\n🎨 Generating visualizations...")
    
    # Generate all plots
    try:
        print("   📈 Model performance plots...")
        generate_model_performance_plots(models, plots_dir)
        
        if reports:
            print("   🏥 Clinical validation plots...")
            generate_clinical_validation_plots(reports, plots_dir)
            
            print("   📋 Performance summary...")
            generate_summary_report(models, reports, plots_dir)
        
        print("   🔧 Feature importance plots...")
        generate_feature_importance_plots(models, plots_dir)
        
        print("   💻 Hardware utilization plot...")
        generate_hardware_utilization_plot(models, plots_dir)
        
        print(f"\n✅ All visualizations generated successfully!")
        print(f"📁 Plots saved to: {plots_dir.absolute()}")
        
        # List generated files
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            print(f"\n📸 Generated {len(plot_files)} plots:")
            for plot_file in sorted(plot_files):
                print(f"   • {plot_file.name}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        print(f"❌ Error generating plots: {e}")

if __name__ == "__main__":
    main()
