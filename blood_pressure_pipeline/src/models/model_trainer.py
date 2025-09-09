"""
Enhanced model training with improved XGBoost configuration and ensemble methods.
Addresses the XGBoost performance issues in the original implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
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
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Enhanced model training with proper XGBoost configuration and ensemble methods."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.trained_models = {}
        self.feature_selector = None
        self.scaler = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists() and YAML_AVAILABLE:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'models': {
                    'random_forest': {'enabled': True},
                    'xgboost_fixed': {'enabled': True},
                    'gradient_boosting': {'enabled': True}
                },
                'ensemble': {'enabled': True},
                'evaluation': {
                    'cv_folds': 5,
                    'test_size': 0.2,
                    'random_state': 42
                }
            }
    
    def _get_model_and_params(self, model_name: str) -> Tuple[Any, Dict]:
        """Get model instance and parameter grid based on configuration."""
        
        model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500, 800],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                    'bootstrap': [True],
                    'oob_score': [True]
                }
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5],
                    'bootstrap': [False]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 400],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost_fixed': {
                'model': xgb.XGBRegressor(
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    random_state=42,
                    tree_method='hist',
                    verbosity=0,
                    early_stopping_rounds=10
                ),
                'params': {
                    'n_estimators': [300, 500, 800],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0.1, 0.5, 1.0],      # Enhanced L1 regularization
                    'reg_lambda': [1.0, 2.0, 3.0],     # Enhanced L2 regularization
                    'min_child_weight': [1, 3, 5],     # Prevent overfitting
                    'gamma': [0, 0.1, 0.2]             # Minimum loss reduction
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1) if LIGHTGBM_AVAILABLE else None,
                'params': {
                    'n_estimators': [300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [6, 8, 10],
                    'num_leaves': [31, 50, 70],
                    'subsample': [0.8, 0.9],
                    'feature_fraction': [0.8, 0.9],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [1, 1.5]
                }
            } if LIGHTGBM_AVAILABLE else {
                'model': None,
                'params': {}
            },
            'support_vector': {
                'model': SVR(),
                'params': {
                    'kernel': ['rbf', 'linear'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 1]
                }
            },
            'neural_network': {
                'model': MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
                'params': {
                    'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01],
                    'activation': ['relu', 'tanh']
                }
            }
        }
        
        # Get from config or use default
        if model_name in self.config.get('models', {}):
            config_params = self.config['models'][model_name].get('param_grid', {})
            if config_params:
                model_configs[model_name]['params'] = config_params
        
        return model_configs[model_name]['model'], model_configs[model_name]['params']
    
    def preprocess_data(self, X: pd.DataFrame, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                       groups: np.ndarray) -> Tuple[Any, np.ndarray, np.ndarray, Any, Any]:
        """Preprocess data with scaling and feature selection."""
        
        logger.info("Preprocessing data...")
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        feature_selection_config = self.config.get('training', {}).get('feature_selection', {})
        method = feature_selection_config.get('method', 'SelectFromModel')
        
        if method == 'SelectFromModel':
            # Use XGBoost for feature selection (it's fast and effective)
            base_estimator = xgb.XGBRegressor(
                n_estimators=100, 
                random_state=42,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
            threshold = feature_selection_config.get('threshold', 'median')
            self.feature_selector = SelectFromModel(base_estimator, threshold=threshold)
            
        elif method == 'RFE':
            base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            n_features = feature_selection_config.get('n_features', 50)
            self.feature_selector = RFE(base_estimator, n_features_to_select=n_features)
        
        # Fit feature selector on SBP (assuming it's the primary target)
        if self.feature_selector:
            self.feature_selector.fit(X_scaled, y_sbp)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Convert to dense array if sparse
            try:
                if hasattr(X_selected, 'toarray'):
                    X_selected = X_selected.toarray()
            except:
                pass
            
            # Get selected feature names
            try:
                if hasattr(self.feature_selector, 'get_support'):
                    selected_mask = self.feature_selector.get_support()
                    selected_features = X.columns[selected_mask].tolist()
                    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
                else:
                    selected_features = [f"feature_{i}" for i in range(X_selected.shape[1])]
            except:
                selected_features = X.columns.tolist()
        else:
            X_selected = X_scaled
            selected_features = X.columns.tolist()
        
        return X_selected, y_sbp, y_dbp, groups, selected_features
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                          groups_train: np.ndarray, target_name: str) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization."""
        
        logger.info(f"Training {model_name} for {target_name}...")
        
        model, param_grid = self._get_model_and_params(model_name)
        
        # Setup cross-validation
        cv_config = self.config.get('evaluation', {})
        cv_folds = cv_config.get('cv_folds', 5)
        
        # Use GroupKFold to ensure patients don't appear in both train and validation
        cv = GroupKFold(n_splits=cv_folds)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv.split(X_train, y_train, groups=groups_train),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        # Special handling for XGBoost with early stopping
        if model_name == 'xgboost_fixed':
            # Create validation split for early stopping
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            
            # Fit with early stopping
            grid_search.fit(
                X_train, y_train,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
        else:
            grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Train final model on full training set
        if model_name == 'xgboost_fixed':
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            best_model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
        else:
            best_model.fit(X_train, y_train)
        
        return {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_ensemble_models(self, individual_models: Dict[str, Any], X_train: np.ndarray, 
                            y_train: np.ndarray) -> Dict[str, Any]:
        """Create and train ensemble models."""
        
        logger.info("Training ensemble models...")
        
        ensemble_models = {}
        
        # Extract individual models
        base_models = [(name, result['model']) for name, result in individual_models.items()]
        
        if len(base_models) >= 2:
            # Voting Regressor
            voting_regressor = VotingRegressor(base_models)
            voting_regressor.fit(X_train, y_train)
            ensemble_models['voting'] = voting_regressor
            
            # Stacking Regressor
            meta_learner = Ridge(alpha=1.0)
            stacking_regressor = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5
            )
            stacking_regressor.fit(X_train, y_train)
            ensemble_models['stacking'] = stacking_regressor
        
        return ensemble_models
    
    def train_all_models(self, X: pd.DataFrame, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                        groups: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train all models for both SBP and DBP prediction."""
        
        logger.info("Starting comprehensive model training...")
        
        # Preprocess data
        X_processed, y_sbp_processed, y_dbp_processed, groups_processed, selected_features = \
            self.preprocess_data(X, y_sbp, y_dbp, groups)
        
        # Split data
        test_size = self.config.get('evaluation', {}).get('test_size', 0.2)
        random_state = self.config.get('evaluation', {}).get('random_state', 42)
        
        X_train, X_test, y_train_sbp, y_test_sbp, groups_train, groups_test = train_test_split(
            X_processed, y_sbp_processed, groups_processed, 
            test_size=test_size, random_state=random_state
        )
        
        _, _, y_train_dbp, y_test_dbp, _, _ = train_test_split(
            X_processed, y_dbp_processed, groups_processed, 
            test_size=test_size, random_state=random_state
        )
        
        # Train individual models
        results = {
            'sbp_models': {},
            'dbp_models': {},
            'sbp_ensemble': {},
            'dbp_ensemble': {},
            'data_splits': {
                'X_train': X_train, 'X_test': X_test,
                'y_train_sbp': y_train_sbp, 'y_test_sbp': y_test_sbp,
                'y_train_dbp': y_train_dbp, 'y_test_dbp': y_test_dbp,
                'groups_train': groups_train, 'groups_test': groups_test
            },
            'preprocessing': {
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': selected_features
            }
        }
        
        # Get enabled models from config
        enabled_models = [
            name for name, config in self.config.get('models', {}).items() 
            if config.get('enabled', True)
        ]
        
        # Train SBP models
        logger.info("Training models for SBP prediction...")
        for model_name in enabled_models:
            try:
                result = self.train_single_model(model_name, X_train, y_train_sbp, groups_train, 'SBP')
                results['sbp_models'][model_name] = result
                logger.info(f"{model_name} SBP - Best CV Score: {result['best_score']:.4f}")
            except Exception as e:
                logger.error(f"Failed to train {model_name} for SBP: {str(e)}")
        
        # Train DBP models
        logger.info("Training models for DBP prediction...")
        for model_name in enabled_models:
            try:
                result = self.train_single_model(model_name, X_train, y_train_dbp, groups_train, 'DBP')
                results['dbp_models'][model_name] = result
                logger.info(f"{model_name} DBP - Best CV Score: {result['best_score']:.4f}")
            except Exception as e:
                logger.error(f"Failed to train {model_name} for DBP: {str(e)}")
        
        # Train ensemble models if enabled
        if self.config.get('ensemble', {}).get('enabled', True):
            if results['sbp_models']:
                results['sbp_ensemble'] = self.train_ensemble_models(
                    results['sbp_models'], X_train, y_train_sbp
                )
            
            if results['dbp_models']:
                results['dbp_ensemble'] = self.train_ensemble_models(
                    results['dbp_models'], X_train, y_train_dbp
                )
        
        self.trained_models = results
        logger.info("Model training completed successfully!")
        
        return results
    
    def save_models(self, results: Dict[str, Any], save_dir: str = "models") -> None:
        """Save trained models and preprocessing objects."""
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save preprocessing objects
        if 'preprocessing' in results:
            preprocessing = results['preprocessing']
            if preprocessing['scaler']:
                joblib.dump(preprocessing['scaler'], save_path / 'scaler.pkl')
            if preprocessing['feature_selector']:
                joblib.dump(preprocessing['feature_selector'], save_path / 'feature_selector.pkl')
            if preprocessing['selected_features']:
                joblib.dump(preprocessing['selected_features'], save_path / 'selected_features.pkl')
        
        # Save individual models
        for target in ['sbp', 'dbp']:
            model_key = f'{target}_models'
            if model_key in results:
                for model_name, model_result in results[model_key].items():
                    model_path = save_path / f'{model_name}_{target}_model.pkl'
                    joblib.dump(model_result['model'], model_path)
                    logger.info(f"Saved {model_name} {target.upper()} model to {model_path}")
        
        # Save ensemble models
        for target in ['sbp', 'dbp']:
            ensemble_key = f'{target}_ensemble'
            if ensemble_key in results:
                for ensemble_name, ensemble_model in results[ensemble_key].items():
                    ensemble_path = save_path / f'{ensemble_name}_{target}_ensemble.pkl'
                    joblib.dump(ensemble_model, ensemble_path)
                    logger.info(f"Saved {ensemble_name} {target.upper()} ensemble to {ensemble_path}")
        
        logger.info(f"All models saved to {save_path}")


def train_enhanced_models(X: pd.DataFrame, y_sbp: np.ndarray, y_dbp: np.ndarray, 
                         groups: np.ndarray, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for enhanced model training.
    
    Args:
        X: Feature DataFrame
        y_sbp: SBP target values
        y_dbp: DBP target values  
        groups: Patient group identifiers
        config_path: Path to model configuration file
        
    Returns:
        Dictionary containing trained models and results
    """
    trainer = ModelTrainer(config_path)
    return trainer.train_all_models(X, y_sbp, y_dbp, groups)
