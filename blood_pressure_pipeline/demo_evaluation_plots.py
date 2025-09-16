#!/usr/bin/env python3
"""
Demo script showcasing comprehensive ML model evaluation plots.
Demonstrates the usage pattern requested in the prompt with sample data.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our plotting functions
from generate_plots import (
    evaluar_modelo, 
    bland_altman_plot, 
    generate_prediction_scatter_plots,
    generate_bland_altman_plots,
    generate_enhanced_feature_importance_plots
)

def create_sample_blood_pressure_data():
    """Create realistic blood pressure prediction data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic features (ECG/PPG-like characteristics)
    n_samples = 500
    n_features = 50
    
    # Create base features with more structured relationships
    X = np.random.randn(n_samples, n_features)
    
    # Add some correlated features to make the problem more learnable
    # Make sure we don't exceed array bounds
    for i in range(0, n_features-2, 3):  # Changed from n_features-1 to n_features-2
        if i+2 < n_features:  # Extra safety check
            X[:, i+1] = X[:, i] + 0.3 * np.random.randn(n_samples)  # Correlated feature
            X[:, i+2] = 0.5 * X[:, i] + 0.5 * X[:, i+1] + 0.2 * np.random.randn(n_samples)
    
    # Generate realistic blood pressure targets with stronger signal-to-noise ratio
    # SBP typically ranges 90-180 mmHg, DBP ranges 60-120 mmHg
    
    # Create stronger linear relationships with multiple features
    sbp_signal = (2.5 * X[:, 0] + 1.8 * X[:, 1] + 1.2 * X[:, 2] + 
                  0.8 * X[:, 3] + 0.5 * X[:, 4])
    dbp_signal = (2.0 * X[:, 1] + 1.5 * X[:, 2] + 1.0 * X[:, 5] + 
                  0.7 * X[:, 6] + 0.4 * X[:, 7])
    
    # Add some non-linear relationships
    sbp_signal += 0.5 * np.sin(X[:, 8]) + 0.3 * (X[:, 9] ** 2)
    dbp_signal += 0.4 * np.cos(X[:, 10]) + 0.2 * (X[:, 11] ** 2)
    
    # Scale and add realistic baseline + noise
    sbp_base = 130 + 15 * (sbp_signal / np.std(sbp_signal)) + np.random.normal(0, 5, n_samples)
    dbp_base = 80 + 10 * (dbp_signal / np.std(dbp_signal)) + np.random.normal(0, 3, n_samples)
    
    # Clip to realistic ranges
    y_sbp = np.clip(sbp_base, 90, 180)
    y_dbp = np.clip(dbp_base, 60, 120)
    
    # Add some correlation between SBP and DBP (realistic medical relationship)
    correlation_noise = 0.3 * (y_sbp - 130) / 50  # DBP tends to increase with SBP
    y_dbp = y_dbp + correlation_noise + np.random.normal(0, 2, n_samples)
    y_dbp = np.clip(y_dbp, 60, 120)
    
    return X, y_sbp, y_dbp

def train_demo_models():
    """Train sample models for demonstration."""
    print("\n====== 📊 CREATING SAMPLE DATA ======")
    
    X, y_sbp, y_dbp = create_sample_blood_pressure_data()
    
    # Split data
    X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(
        X, y_sbp, test_size=0.3, random_state=42)
    _, _, y_dbp_train, y_dbp_test = train_test_split(
        X, y_dbp, test_size=0.3, random_state=42)
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    
    models = {}
    predictions = {}
    
    # Train Random Forest models
    print("\n====== 📌 RANDOM FOREST RESULTS ======")
    
    # SBP Random Forest - improved parameters
    rf_sbp = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
                                  min_samples_leaf=2, random_state=42)
    rf_sbp.fit(X_train, y_sbp_train)
    y_pred_rf_sbp = rf_sbp.predict(X_test)
    
    rmse_rf_sbp, mae_rf_sbp, r2_rf_sbp = evaluar_modelo(y_sbp_test, y_pred_rf_sbp, "Random Forest SBP")
    
    models['random_forest_sbp_m3_pro'] = {
        'model': rf_sbp,
        'predictions': y_pred_rf_sbp,
        'actuals': y_sbp_test,
        'performance': {
            'val_rmse': rmse_rf_sbp,
            'val_mae': mae_rf_sbp,
            'val_r2': r2_rf_sbp,
            'training_time': 45.2  # Sample time in seconds
        }
    }
    
    # DBP Random Forest - improved parameters
    rf_dbp = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42)
    rf_dbp.fit(X_train, y_dbp_train)
    y_pred_rf_dbp = rf_dbp.predict(X_test)
    
    rmse_rf_dbp, mae_rf_dbp, r2_rf_dbp = evaluar_modelo(y_dbp_test, y_pred_rf_dbp, "Random Forest DBP")
    
    models['random_forest_dbp_m3_pro'] = {
        'model': rf_dbp,
        'predictions': y_pred_rf_dbp,
        'actuals': y_dbp_test,
        'performance': {
            'val_rmse': rmse_rf_dbp,
            'val_mae': mae_rf_dbp,
            'val_r2': r2_rf_dbp,
            'training_time': 42.8
        }
    }
    
    # Train XGBoost models
    print("\n====== 📌 XGBOOST RESULTS ======")
    
    # SBP XGBoost - improved parameters
    xgb_sbp = xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_sbp.fit(X_train, y_sbp_train)
    y_pred_xgb_sbp = xgb_sbp.predict(X_test)
    
    rmse_xgb_sbp, mae_xgb_sbp, r2_xgb_sbp = evaluar_modelo(y_sbp_test, y_pred_xgb_sbp, "XGBoost SBP")
    
    models['xgboost_m3_optimized_sbp_m3_pro'] = {
        'model': xgb_sbp,
        'predictions': y_pred_xgb_sbp,
        'actuals': y_sbp_test,
        'performance': {
            'val_rmse': rmse_xgb_sbp,
            'val_mae': mae_xgb_sbp,
            'val_r2': r2_xgb_sbp,
            'training_time': 38.5
        }
    }
    
    # DBP XGBoost - improved parameters
    xgb_dbp = xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_dbp.fit(X_train, y_dbp_train)
    y_pred_xgb_dbp = xgb_dbp.predict(X_test)
    
    rmse_xgb_dbp, mae_xgb_dbp, r2_xgb_dbp = evaluar_modelo(y_dbp_test, y_pred_xgb_dbp, "XGBoost DBP")
    
    models['xgboost_m3_optimized_dbp_m3_pro'] = {
        'model': xgb_dbp,
        'predictions': y_pred_xgb_dbp,
        'actuals': y_dbp_test,
        'performance': {
            'val_rmse': rmse_xgb_dbp,
            'val_mae': mae_xgb_dbp,
            'val_r2': r2_xgb_dbp,
            'training_time': 35.7
        }
    }
    
    return models

def generate_demo_evaluation_plots():
    """Generate comprehensive evaluation plots using the demo models."""
    print("\n🎨 Demo: Machine Learning Model Evaluation Plot Generation")
    print("=" * 70)
    
    # Train models
    models = train_demo_models()
    
    # Create demo plots directory
    plots_dir = Path("demo_evaluation_plots")
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n🎯 Generating comprehensive evaluation plots...")
    print(f"📁 Output directory: {plots_dir.absolute()}")
    
    try:
        # Generate all three types of plots
        print("\n📈 1. Prediction Scatter Plots (2x2 Grid)")
        generate_prediction_scatter_plots(models, output_dir=plots_dir)
        
        print("📊 2. Bland-Altman Agreement Plots (2x2 Grid)")  
        generate_bland_altman_plots(models, output_dir=plots_dir)
        
        print("🎯 3. Enhanced Feature Importance Visualizations")
        generate_enhanced_feature_importance_plots(models, output_dir=plots_dir, top_n=15)
        
        print(f"\n✅ Demo evaluation plots generated successfully!")
        
        # List generated files
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            print(f"\n📸 Generated {len(plot_files)} evaluation plots:")
            for plot_file in sorted(plot_files):
                print(f"   • {plot_file.name}")
            
            print(f"\n📋 Expected outputs:")
            print(f"   1. prediction_comparison.png - 2x2 scatter plots comparing models")
            print(f"   2. bland_altman_plots.png - 2x2 agreement analysis plots") 
            print(f"   3. feature_importances.png - Feature importance bar charts")
        
        # Display sample statistics
        print(f"\n📊 Sample Model Performance Summary:")
        print(f"{'Model':<30} {'Target':<6} {'RMSE':<8} {'R²':<8}")
        print(f"{'-'*60}")
        
        for model_name, model_data in models.items():
            if 'performance' in model_data:
                perf = model_data['performance']
                target = 'SBP' if '_sbp_' in model_name else 'DBP'
                clean_name = model_name.replace(f'_{target.lower()}_m3_pro', '').replace('_', ' ').title()
                
                print(f"{clean_name:<30} {target:<6} {perf['val_rmse']:<8.2f} {perf['val_r2']:<8.3f}")
        
    except Exception as e:
        print(f"❌ Error generating demo plots: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_manual_plotting():
    """Demonstrate manual plotting functions as requested in prompt."""
    print("\n🔧 Manual Plotting Function Demonstration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.normal(120, 20, 100)  # Sample SBP values
    y_pred = y_true + np.random.normal(0, 5, 100)  # Add some prediction error
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Prediction scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color='blue', s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ax1.set_xlabel('Actual SBP (mmHg)', fontweight='bold')
    ax1.set_ylabel('Predicted SBP (mmHg)', fontweight='bold')
    ax1.set_title(f'Prediction Accuracy\nRMSE: {rmse:.2f} | R²: {r2:.3f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Bland-Altman plot
    ax2 = axes[1]
    bland_altman_plot(ax2, y_true, y_pred, 'Agreement Analysis', color='green')
    
    plt.tight_layout()
    plt.savefig('demo_evaluation_plots/manual_demo_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Manual plotting demonstration completed")
    print("📁 Saved: manual_demo_plots.png")

if __name__ == "__main__":
    # Run the comprehensive demo
    generate_demo_evaluation_plots()
    
    # Run manual plotting demonstration
    demonstrate_manual_plotting()
    
    print(f"\n🎉 Demo completed! Check the 'demo_evaluation_plots' directory for results.")