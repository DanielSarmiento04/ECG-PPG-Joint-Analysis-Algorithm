"""
M3 Pro Optimized Model Trainer
Enhanced for Apple Silicon performance with Metal acceleration and Accelerate framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import platform
import os
import psutil
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import joblib
import time
from pathlib import Path

# Apple Silicon optimizations
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# Optional: Neural network with Metal Performance Shaders
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    # Check for Metal Performance Shaders support
    METAL_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    torch = None
    nn = None
    PYTORCH_AVAILABLE = False
    METAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class AppleSiliconOptimizer:
    """Optimization utilities specific to Apple Silicon (M3 Pro)."""
    
    @staticmethod
    def detect_hardware():
        """Detect M3 Pro hardware configuration."""
        system_info = {
            'platform': platform.machine(),
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'is_apple_silicon': platform.machine() == 'arm64'
        }
        
        # Detect M3 Pro specific features
        if system_info['is_apple_silicon']:
            # M3 Pro typically has 12 cores and 18-36GB unified memory
            if system_info['cpu_count'] >= 10:
                system_info['chip_type'] = 'M3_Pro_or_higher'
                system_info['performance_cores'] = 6
                system_info['efficiency_cores'] = 6
                system_info['recommended_n_jobs'] = min(10, system_info['cpu_count'] - 2)
            else:
                system_info['chip_type'] = 'M3_Base'
                system_info['recommended_n_jobs'] = min(6, system_info['cpu_count'] - 1)
        else:
            system_info['recommended_n_jobs'] = min(4, system_info['cpu_count'])
            
        return system_info
    
    @staticmethod
    def optimize_numpy():
        """Optimize NumPy for Apple Silicon."""
        if platform.machine() == 'arm64':
            # Use Apple's Accelerate framework
            os.environ['OPENBLAS_NUM_THREADS'] = str(min(8, os.cpu_count()))
            os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, os.cpu_count()))
            logger.info("NumPy optimized for Apple Silicon with Accelerate framework")

class M3ProModelTrainer:
    """Model trainer optimized for MacBook Pro M3 Pro performance."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with M3 Pro optimizations."""
        self.hardware_info = AppleSiliconOptimizer.detect_hardware()
        AppleSiliconOptimizer.optimize_numpy()
        
        self.config = self._load_config(config_path)
        self.trained_models = {}
        self.feature_selector = None
        self.scaler = None
        
        # Log hardware configuration
        logger.info(f"Hardware detected: {self.hardware_info}")
        logger.info(f"Recommended n_jobs: {self.hardware_info['recommended_n_jobs']}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load M3 Pro optimized configuration."""
        default_config = self._get_m3_pro_default_config()
        
        if config_path and Path(config_path).exists() and YAML_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(file_config)
                    logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
                
        return default_config
    
    def _get_m3_pro_default_config(self) -> Dict:
        """Get default configuration optimized for M3 Pro."""
        n_jobs = self.hardware_info['recommended_n_jobs']
        
        return {
            'performance': {
                'n_jobs': n_jobs,
                'parallel_backend': 'threading',
                'batch_size': 512,
                'memory_efficient': True
            },
            'models': {
                'random_forest': {
                    'enabled': True,
                    'param_grid': {
                        'n_estimators': [300, 500, 800],
                        'max_features': ['sqrt', 'log2', 0.3, 0.5],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [2, 5, 10]
                    },
                    'base_params': {
                        'n_jobs': n_jobs,
                        'random_state': 42,
                        'warm_start': True,
                        'max_samples': 0.8
                    }
                },
                'xgboost_m3_optimized': {
                    'enabled': True,
                    'param_grid': {
                        'learning_rate': [0.01, 0.05, 0.1, 0.15],
                        'max_depth': [3, 4, 5, 6],
                        'n_estimators': [200, 300, 500],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'reg_alpha': [0.1, 0.5, 1.0],
                        'reg_lambda': [1.0, 2.0, 3.0]
                    },
                    'base_params': {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'tree_method': 'hist',  # Optimized for M3 Pro
                        'nthread': n_jobs,
                        'early_stopping_rounds': 15,
                        'random_state': 42,
                        'verbosity': 0
                    }
                },
                'lightgbm_m3': {
                    'enabled': LIGHTGBM_AVAILABLE,
                    'param_grid': {
                        'learning_rate': [0.01, 0.05, 0.1],
                        'num_leaves': [31, 50, 70, 100],
                        'n_estimators': [200, 300, 500],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    'base_params': {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'num_threads': n_jobs,
                        'force_row_wise': True,  # Better for M3 Pro memory
                        'early_stopping_rounds': 15,
                        'random_state': 42,
                        'verbosity': -1
                    }
                }
            },
            'ensemble': {
                'enabled': True,
                'voting_weights': None,  # Auto-determined
                'stacking_meta_learner': 'ridge'
            },
            'cross_validation': {
                'n_splits': 5,
                'shuffle': True,
                'random_state': 42,
                'n_jobs': min(5, n_jobs)
            },
            'feature_selection': {
                'enabled': True,
                'method': 'model_based',
                'max_features': 100
            }
        }
    
    def optimize_xgboost_for_m3(self, base_params: Dict) -> Dict:
        """Optimize XGBoost parameters specifically for M3 Pro."""
        optimized_params = base_params.copy()
        
        # M3 Pro specific optimizations
        optimized_params.update({
            'tree_method': 'hist',  # Fastest on M3 Pro
            'nthread': self.hardware_info['recommended_n_jobs'],
            'max_bin': 256,  # Optimal for M3 Pro memory bandwidth
            'grow_policy': 'lossguide',  # More efficient tree growth
            'single_precision_histogram': True,  # Memory optimization
        })
        
        logger.info("XGBoost parameters optimized for M3 Pro")
        return optimized_params
    
    def train_lightgbm_m3_optimized(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train LightGBM with M3 Pro optimizations."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, skipping")
            return {}
        
        config = self.config['models']['lightgbm_m3']
        if not config['enabled']:
            return {}
        
        logger.info("Training LightGBM with M3 Pro optimizations...")
        start_time = time.time()
        
        # M3 Pro optimized parameters
        base_params = config['base_params'].copy()
        base_params.update({
            'num_threads': self.hardware_info['recommended_n_jobs'],
            'force_row_wise': True,  # Better for M3 Pro unified memory
            'max_bin': 255,  # Optimal for ARM64
            'feature_pre_filter': False,  # Let M3 Pro handle feature filtering
        })
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Cross-validation with GroupKFold if groups provided
        if groups is not None:
            cv_folds = GroupKFold(n_splits=self.config['cross_validation']['n_splits'])
            cv_results = lgb.cv(
                base_params,
                train_data,
                folds=cv_folds.split(X_train, y_train, groups),
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
            )
            best_iteration = len(cv_results['valid rmse-mean'])
            logger.info(f"LightGBM CV best iteration: {best_iteration}")
        else:
            best_iteration = 500
        
        # Train final model
        model = lgb.train(
            base_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=best_iteration,
            callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
        )
        
        # Predictions and evaluation
        y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        
        train_time = time.time() - start_time
        
        results = {
            'model': model,
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'feature_importance': model.feature_importance(importance_type='gain'),
            'training_time': train_time,
            'best_iteration': model.best_iteration
        }
        
        logger.info(f"LightGBM M3 - Val R²: {results['val_r2']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.2f}, Time: {train_time:.1f}s")
        
        return results
    
    def create_m3_optimized_neural_network(self, input_dim: int) -> Optional[Any]:
        """Create neural network optimized for M3 Pro Metal Performance Shaders."""
        if not PYTORCH_AVAILABLE or not METAL_AVAILABLE:
            logger.info("PyTorch with MPS not available, skipping neural network")
            return None
        
        class M3OptimizedNN(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        device = torch.device("mps" if METAL_AVAILABLE else "cpu")
        model = M3OptimizedNN(input_dim).to(device)
        
        logger.info(f"Neural network created for device: {device}")
        return model
    
    def train_enhanced_models_m3_pro(self, X: np.ndarray, y_sbp: np.ndarray, 
                                    y_dbp: np.ndarray, groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models with M3 Pro optimizations."""
        logger.info("Starting M3 Pro optimized model training...")
        logger.info(f"Dataset shape: {X.shape}, Hardware: {self.hardware_info['chip_type']}")
        
        # Data preprocessing optimized for M3 Pro
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # Feature selection for M3 Pro memory efficiency
        if self.config['feature_selection']['enabled']:
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, n_jobs=self.hardware_info['recommended_n_jobs']),
                max_features=self.config['feature_selection']['max_features']
            )
            X_scaled = selector.fit_transform(X_scaled, y_sbp)
            self.feature_selector = selector
            logger.info(f"Feature selection: {X.shape[1]} -> {X_scaled.shape[1]} features")
        
        # Split data with stratification
        X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
            X_scaled, y_sbp, y_dbp, test_size=0.2, random_state=42
        )
        
        # Further split for validation
        X_train, X_val, y_sbp_train, y_sbp_val, y_dbp_train, y_dbp_val = train_test_split(
            X_train, y_sbp_train, y_dbp_train, test_size=0.2, random_state=42
        )
        
        results = {
            'data_splits': {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train_sbp': y_sbp_train, 'y_val_sbp': y_sbp_val, 'y_test_sbp': y_sbp_test,
                'y_train_dbp': y_dbp_train, 'y_val_dbp': y_dbp_val, 'y_test_dbp': y_dbp_test
            },
            'feature_selector': self.feature_selector,
            'scaler': self.scaler,
            'hardware_info': self.hardware_info
        }
        
        # Train models for both SBP and DBP
        for target, y_train, y_val, y_test in [
            ('sbp', y_sbp_train, y_sbp_val, y_sbp_test),
            ('dbp', y_dbp_train, y_dbp_val, y_dbp_test)
        ]:
            logger.info(f"\n=== Training models for {target.upper()} ===")
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
            
            # LightGBM with M3 Pro optimizations
            if self.config['models']['lightgbm_m3']['enabled']:
                lgb_result = self.train_lightgbm_m3_optimized(X_train, y_train, X_val, y_val)
                if lgb_result:
                    target_results['lightgbm_m3'] = lgb_result
            
            # Create ensemble models
            if self.config['ensemble']['enabled'] and len(target_results) >= 2:
                ensemble_results = self.create_m3_optimized_ensembles(target_results, X_train, y_train, X_val, y_val)
                target_results.update(ensemble_results)
            
            results[f'{target}_models'] = target_results
        
        logger.info("M3 Pro optimized training completed!")
        return results
    
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
        
        # Grid search with M3 Pro optimized CV
        cv_folds = GroupKFold(n_splits=3)  # Reduced for speed on M3 Pro
        grid_search = GridSearchCV(
            rf, config['param_grid'],
            cv=cv_folds,
            scoring='r2',
            n_jobs=min(3, self.hardware_info['recommended_n_jobs']),
            verbose=0
        )
        
        # Create dummy groups for GroupKFold
        groups = np.arange(len(X_train)) % 3
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
        
        # Grid search with M3 Pro optimizations
        cv_folds = GroupKFold(n_splits=3)
        grid_search = GridSearchCV(
            xgb_model, config['param_grid'],
            cv=cv_folds,
            scoring='r2',
            n_jobs=min(3, self.hardware_info['recommended_n_jobs']),
            verbose=0
        )
        
        # Create dummy groups
        groups = np.arange(len(X_train)) % 3
        grid_search.fit(
            X_train, y_train,
            groups=groups,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
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
        
        logger.info(f"XGBoost M3 - Val R²: {results['val_r2']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.2f}, Time: {train_time:.1f}s")
        
        return results
    
    def create_m3_optimized_ensembles(self, models: Dict[str, Any], X_train: np.ndarray, 
                                     y_train: np.ndarray, X_val: np.ndarray, 
                                     y_val: np.ndarray) -> Dict[str, Any]:
        """Create ensemble models optimized for M3 Pro."""
        logger.info("Creating M3 Pro optimized ensemble models...")
        
        base_models = [(name, result['model']) for name, result in models.items()]
        ensemble_results = {}
        
        if len(base_models) >= 2:
            # Voting Regressor
            voting_regressor = VotingRegressor(
                estimators=base_models,
                n_jobs=self.hardware_info['recommended_n_jobs']
            )
            voting_regressor.fit(X_train, y_train)
            
            y_pred_val_voting = voting_regressor.predict(X_val)
            ensemble_results['voting_m3'] = {
                'model': voting_regressor,
                'val_r2': r2_score(y_val, y_pred_val_voting),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val_voting))
            }
            
            # Stacking Regressor with Ridge meta-learner
            stacking_regressor = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=1.0),
                n_jobs=self.hardware_info['recommended_n_jobs'],
                cv=3  # Reduced for M3 Pro speed
            )
            stacking_regressor.fit(X_train, y_train)
            
            y_pred_val_stacking = stacking_regressor.predict(X_val)
            ensemble_results['stacking_m3'] = {
                'model': stacking_regressor,
                'val_r2': r2_score(y_val, y_pred_val_stacking),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val_stacking))
            }
            
            logger.info(f"Ensemble M3 - Voting R²: {ensemble_results['voting_m3']['val_r2']:.4f}, "
                       f"Stacking R²: {ensemble_results['stacking_m3']['val_r2']:.4f}")
        
        return ensemble_results


def train_enhanced_models_m3_pro(X: np.ndarray, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                                groups: Optional[np.ndarray] = None, 
                                config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for M3 Pro optimized model training.
    
    Args:
        X: Feature matrix
        y_sbp: Systolic blood pressure targets
        y_dbp: Diastolic blood pressure targets
        groups: Group labels for cross-validation
        config_path: Path to M3 Pro configuration file
        
    Returns:
        Dictionary containing trained models and results
    """
    trainer = M3ProModelTrainer(config_path)
    return trainer.train_enhanced_models_m3_pro(X, y_sbp, y_dbp, groups)
