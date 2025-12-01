import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def train_baseline_model(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter out any remaining NaNs
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    # Define features
    # We use the robust features we just fixed + the normalized ones
    feature_cols = [
        # Physiological Intervals (Normalized are best)
        'pat_ecg_ppg_norm', 'pat_to_peak_norm', 'pat_to_maxslope_norm',
        'ptt_peak_to_foot_norm', 'ptt_peak_to_peak_norm',
        
        # Morphology (Normalized)
        'amplitude_ratio_ra_norm', 'systolic_duration_tsd_norm', 
        'diastolic_duration_tfd_norm', 'systolic_area_ratio_norm',
        'reflection_index_norm',
        
        # Derivatives
        'max_upslope_norm', 'slope_ratio', 
        
        # Statistical (Robust Fallback)
        'stat_mean', 'stat_std', 'stat_skew', 'stat_kurtosis',
        'stat_power_ratio',
        
        # Heart Rate
        'hr_bpm_norm'
    ]
    
    # Check which features exist
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df[available_features]
    y = df['sbp_reference']
    groups = df['patient_id']
    
    # Split: Leave-One-Group-Out (or GroupKFold)
    # For 5 patients, we can do 5-fold CV
    gkf = GroupKFold(n_splits=min(5, df['patient_id'].nunique()))
    
    fold = 1
    metrics = []
    
    print("\n" + "="*40)
    print("STARTING CROSS-VALIDATION")
    print("="*40)
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Fold {fold}: MAE={mae:.2f} mmHg, RMSE={rmse:.2f} mmHg, R2={r2:.4f}")
        metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
        fold += 1
        
    # Summary
    avg_mae = np.mean([m['mae'] for m in metrics])
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    avg_r2 = np.mean([m['r2'] for m in metrics])
    
    print("\n" + "="*40)
    print(f"AVERAGE RESULTS: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, R2={avg_r2:.4f}")
    print("="*40)
    
    # Final Train on All Data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X, y)
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(final_model, max_num_features=15, height=0.8)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig('dataset/data/figures/xgb_feature_importance.png')
    print("Feature importance plot saved.")
    
    return final_model

if __name__ == "__main__":
    data_path = 'dataset/data/processed/bp_dataset_features.csv'
    if os.path.exists(data_path):
        train_baseline_model(data_path)
    else:
        print(f"File not found: {data_path}")
