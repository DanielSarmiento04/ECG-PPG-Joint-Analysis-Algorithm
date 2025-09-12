#!/usr/bin/env python3
"""
M3 Pro Model Trainer - Clean Working Version
Simplified trainer focused on Random Forest and XGBoost with M3 Pro optimizations.
"""

import os
import time
import logging
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Core ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# Optional imports with fallbacks
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class AppleSiliconOptimizer:
    """Hardware optimization utilities for Apple Silicon M3 Pro."""
    
    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Detect hardware configuration and optimize settings."""
        
        # Get basic system info
        cpu_count = os.cpu_count() or 8
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            memory_gb = 16.0  # Default assumption
        
        # Detect Apple Silicon
        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
        
        # Detect M3 Pro specifically
        chip_type = "Unknown"
        if is_apple_silicon:
            if cpu_count >= 11:  # M3 Pro has 11-12 cores
                chip_type = "M3 Pro (or similar)"
            elif cpu_count >= 8:
                chip_type = "M3 (or similar)"
            else:
                chip_type = "Apple Silicon"
        
        # Calculate optimal n_jobs for M3 Pro
        if is_apple_silicon and cpu_count >= 11:
            # M3 Pro: Use most cores but leave some for system
            recommended_n_jobs = min(10, cpu_count - 1)
        else:
            # Other systems: Use conservative approach
            recommended_n_jobs = min(6, cpu_count - 1)
        
        hardware_info = {
            'platform': platform.platform(),
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'is_apple_silicon': is_apple_silicon,
            'chip_type': chip_type,
            'recommended_n_jobs': recommended_n_jobs
        }
        
        logger.info(f"Hardware detected: {chip_type}, {cpu_count} cores, {memory_gb:.1f}GB RAM")
        return hardware_info
    
    @staticmethod
    def optimize_numpy():
        """Optimize NumPy for Apple Silicon."""
        try:
            # Set thread limits for M3 Pro optimization
            cpu_count = os.cpu_count() or 8
            os.environ['OPENBLAS_NUM_THREADS'] = str(min(8, cpu_count))
            os.environ['MKL_NUM_THREADS'] = str(min(8, cpu_count))
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, cpu_count))
            
            logger.info("NumPy optimized for Apple Silicon")
        except Exception as e:
            logger.warning(f"Could not optimize NumPy: {e}")


class M3ProModelTrainer:
    """Model trainer optimized for M3 Pro hardware."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the M3 Pro optimized trainer."""
        
        # Detect and optimize hardware
        self.hardware_info = AppleSiliconOptimizer.detect_hardware()
        AppleSiliconOptimizer.optimize_numpy()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        logger.info("M3 Pro Model Trainer initialized")
        logger.info(f"Hardware: {self.hardware_info['chip_type']}")
        logger.info(f"Recommended parallel jobs: {self.hardware_info['recommended_n_jobs']}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration with M3 Pro defaults."""
        
        # Default M3 Pro optimized configuration
        default_config = {
            'models': {
                'random_forest': {
                    'enabled': True,
                    'base_params': {
                        'n_estimators': 200,
                        'max_depth': 12,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'bootstrap': True,
                        'random_state': 42,
                        'n_jobs': self.hardware_info['recommended_n_jobs']
                    },
                    'param_grid': {
                        'n_estimators': [100, 200],  # Reduced from [200, 300]
                        'max_depth': [10, 12],       # Reduced from [10, 12, 15]
                        'min_samples_split': [5],    # Reduced grid
                        'min_samples_leaf': [2]      # Reduced grid
                    }
                },
                'xgboost_m3_optimized': {
                    'enabled': True,
                    'base_params': {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'tree_method': 'hist',  # Best for M3 Pro
                        'n_estimators': 200,        # Reduced from 300
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.9,
                        'colsample_bytree': 0.9,
                        'random_state': 42,
                        'n_jobs': self.hardware_info['recommended_n_jobs'],
                        'verbosity': 0
                    },
                    'param_grid': {
                        'learning_rate': [0.1],      # Reduced from [0.05, 0.1, 0.15]
                        'max_depth': [6, 8],         # Reduced from [4, 6, 8]
                        'n_estimators': [200],       # Reduced from [200, 300]
                        'subsample': [0.9],          # Reduced from [0.8, 0.9]
                        'colsample_bytree': [0.9]    # Reduced from [0.8, 0.9]
                    }
                }
            },
            'ensemble': {
                'enabled': True,
                'voting_weights': None
            },
            'cross_validation': {
                'n_splits': 5,
                'shuffle': True,
                'random_state': 42,
                'n_jobs': min(5, self.hardware_info['recommended_n_jobs'])
            },
            'feature_selection': {
                'enabled': True,
                'method': 'model_based',
                'max_features': 100
            }
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists() and YAML_AVAILABLE and yaml is not None:
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(file_config)
                    logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def optimize_xgboost_for_m3(self, base_params: Dict) -> Dict:
        """Optimize XGBoost parameters specifically for M3 Pro."""
        optimized_params = base_params.copy()
        
        # M3 Pro specific optimizations
        optimized_params.update({
            'tree_method': 'hist',  # Fastest on M3 Pro
            'n_jobs': self.hardware_info['recommended_n_jobs'],
            'max_bin': 256,  # Optimal for M3 Pro memory bandwidth
        })
        
        logger.info("XGBoost parameters optimized for M3 Pro")
        return optimized_params
    
    def train_random_forest_m3_optimized(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest with M3 Pro optimizations."""
        logger.info("Training Random Forest with M3 Pro optimizations...")
        start_time = time.time()
        
        config = self.config['models']['random_forest']
        base_params = config['base_params'].copy()
        base_params['n_jobs'] = self.hardware_info['recommended_n_jobs']
        
        # M3 Pro memory optimization
        if self.hardware_info['memory_gb'] < 32:
            base_params['max_samples'] = 0.7  # Reduce memory usage
        
        rf = RandomForestRegressor(**base_params)
        
        # Grid search with M3 Pro optimized CV (reduced folds for speed)
        cv_folds = GroupKFold(n_splits=2)  # Reduced from 3 to 2 for faster training
        grid_search = GridSearchCV(
            rf, config['param_grid'],
            cv=cv_folds,
            scoring='r2',
            n_jobs=min(2, self.hardware_info['recommended_n_jobs']),  # Reduced parallel jobs
            verbose=0
        )
        
        # Create dummy groups for GroupKFold
        groups = np.arange(len(X_train)) % 2  # Changed to match n_splits
        grid_search.fit(X_train, y_train, groups=groups)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        
        train_time = time.time() - start_time
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'feature_importance': best_model.feature_importances_,
            'training_time': train_time
        }
        
        logger.info(f"Random Forest M3 - Val R²: {results['val_r2']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.2f}, Time: {train_time:.1f}s")
        
        return results
    
    def train_xgboost_m3_optimized(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost with M3 Pro specific optimizations."""
        logger.info("Training XGBoost with M3 Pro optimizations...")
        start_time = time.time()
        
        config = self.config['models']['xgboost_m3_optimized']
        base_params = self.optimize_xgboost_for_m3(config['base_params'])
        
        # Create XGBoost regressor
        xgb_model = xgb.XGBRegressor(**base_params)
        
        # Grid search with M3 Pro optimizations (optimized for speed)
        cv_folds = GroupKFold(n_splits=2)  # Reduced from 3 to 2 for faster training
        grid_search = GridSearchCV(
            xgb_model, config['param_grid'],
            cv=cv_folds,
            scoring='r2',
            n_jobs=min(2, self.hardware_info['recommended_n_jobs']),  # Reduced parallel jobs
            verbose=0
        )
        
        # Create dummy groups
        groups = np.arange(len(X_train)) % 2  # Changed to match n_splits
        grid_search.fit(X_train, y_train, groups=groups)
        
        best_model = grid_search.best_estimator_
        
        # Make final predictions with the best model
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        
        train_time = time.time() - start_time
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'feature_importance': best_model.feature_importances_,
            'training_time': train_time
        }
        
        logger.info(f"XGBoost M3 - Val R²: {results['val_r2']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.2f}, Time: {train_time:.1f}s")
        
        return results
    
    def create_ensemble_model(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Create an ensemble model from trained models."""
        logger.info("Creating M3 Pro optimized ensemble...")
        start_time = time.time()
        
        if len(models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return {}
        
        # Simple averaging ensemble
        predictions_val = []
        predictions_train = []
        
        for model_name, model_result in models.items():
            if 'model' in model_result:
                model = model_result['model']
                pred_train = model.predict(X_train)
                pred_val = model.predict(X_val)
                
                predictions_train.append(pred_train)
                predictions_val.append(pred_val)
        
        if not predictions_train:
            return {}
        
        # Average predictions
        ensemble_pred_train = np.mean(predictions_train, axis=0)
        ensemble_pred_val = np.mean(predictions_val, axis=0)
        
        train_time = time.time() - start_time
        
        results = {
            'model': 'ensemble_average',
            'train_r2': r2_score(y_train, ensemble_pred_train),
            'val_r2': r2_score(y_val, ensemble_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred_val)),
            'training_time': train_time,
            'component_models': list(models.keys())
        }
        
        logger.info(f"Ensemble M3 - Val R²: {results['val_r2']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.2f}")
        
        return results
    
    def train_models(self, X: np.ndarray, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                    groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models with M3 Pro optimizations."""
        
        logger.info("Starting M3 Pro optimized model training...")
        
        # Feature selection
        if self.config['feature_selection']['enabled']:
            logger.info("Performing feature selection...")
            selector = SelectKBest(score_func=f_regression, 
                                 k=min(self.config['feature_selection']['max_features'], X.shape[1]))
            X_selected = selector.fit_transform(X, y_sbp)
            logger.info(f"Features reduced from {X.shape[1]} to {X_selected.shape[1]}")
        else:
            X_selected = X
            selector = None
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        results = {
            'scaler': scaler,
            'feature_selector': selector,
            'feature_names': list(range(X_scaled.shape[1]))
        }
        
        # Store data splits for evaluation
        data_splits = {}
        
        # Train for both targets
        for target, y_target in [('sbp', y_sbp), ('dbp', y_dbp)]:
            logger.info(f"\n=== Training models for {target.upper()} ===")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_target, test_size=0.2, random_state=42
            )
            
            # Store splits for evaluation
            data_splits[target] = {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val
            }
            
            target_results = {}
            
            # Random Forest with M3 Pro optimizations
            if self.config['models']['random_forest']['enabled']:
                rf_result = self.train_random_forest_m3_optimized(X_train, y_train, X_val, y_val)
                if rf_result:
                    target_results['random_forest_m3'] = rf_result
            
            # XGBoost with M3 Pro optimizations
            if self.config['models']['xgboost_m3_optimized']['enabled']:
                xgb_result = self.train_xgboost_m3_optimized(X_train, y_train, X_val, y_val)
                if xgb_result:
                    target_results['xgboost_m3_optimized'] = xgb_result
            
            # Create ensemble if enabled and we have multiple models
            if self.config['ensemble']['enabled'] and len(target_results) >= 2:
                ensemble_result = self.create_ensemble_model(target_results, X_train, y_train, X_val, y_val)
                if ensemble_result:
                    target_results['ensemble_m3'] = ensemble_result
            
            results[f'{target}_models'] = target_results
        
        # Add data splits to results for evaluation
        results['data_splits'] = data_splits
        
        logger.info("M3 Pro optimized training completed!")
        return results


def train_enhanced_models_m3_pro(X: np.ndarray, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                                groups: Optional[np.ndarray] = None, 
                                config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to train models with M3 Pro optimizations.
    
    Args:
        X: Feature matrix
        y_sbp: Systolic blood pressure targets
        y_dbp: Diastolic blood pressure targets
        groups: Patient groups for cross-validation
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing trained models and results
    """
    
    trainer = M3ProModelTrainer(config_path)
    return trainer.train_models(X, y_sbp, y_dbp, groups)


if __name__ == "__main__":
    # Test the trainer
    print("🧪 Testing M3 Pro Model Trainer...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X = np.random.randn(n_samples, n_features)
    y_sbp = 120 + 15 * np.random.randn(n_samples) + np.sum(X[:, :5] * 0.5, axis=1)
    y_dbp = 80 + 10 * np.random.randn(n_samples) + np.sum(X[:, :3] * 0.3, axis=1)
    
    # Train models
    results = train_enhanced_models_m3_pro(X, y_sbp, y_dbp)
    
    print("✅ M3 Pro Model Trainer test completed!")
    for target in ['sbp', 'dbp']:
        if f'{target}_models' in results:
            models = results[f'{target}_models']
            print(f"\n{target.upper()} Models:")
            for model_name, model_result in models.items():
                if 'val_r2' in model_result:
                    print(f"  • {model_name}: R²={model_result['val_r2']:.4f}, "
                          f"RMSE={model_result['val_rmse']:.2f}")
