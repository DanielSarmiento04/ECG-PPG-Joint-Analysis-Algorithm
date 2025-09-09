"""
Main pipeline orchestration for blood pressure prediction.
Integrates all components into a cohesive, production-ready workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
import time
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import pipeline components
from src.data.data_validator import validate_and_clean_data
from src.data.feature_extractor import extract_enhanced_features
from src.models.model_trainer import train_enhanced_models
from src.models.model_evaluator import evaluate_models
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class BloodPressurePipeline:
    """Main pipeline orchestrator for blood pressure prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration."""
        self.config = config or self._default_config()
        self.results = {}
        
        # Setup logging
        setup_logging(
            level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file', 'logs/pipeline.log')
        )
        
        logger.info("Blood Pressure Prediction Pipeline initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'data': {
                'file_path': 'data/Final_data_base.xlsx',
                'validation': {
                    'enabled': True,
                    'outlier_detection': True,
                    'signal_quality_check': True
                }
            },
            'feature_engineering': {
                'enhanced_extraction': True,
                'wavelet_features': True,
                'cross_correlation': True,
                'filter_enabled': True
            },
            'models': {
                'random_forest': {'enabled': True},
                'xgboost_fixed': {'enabled': True},
                'gradient_boosting': {'enabled': True},
                'neural_network': {'enabled': True}
            },
            'ensemble': {
                'enabled': True
            },
            'evaluation': {
                'clinical_validation': True,
                'generate_plots': True,
                'save_results': True
            },
            'output': {
                'model_dir': 'models',
                'report_dir': 'reports',
                'plot_dir': 'reports/plots'
            }
        }
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load and initial validation of data."""
        if file_path is None:
            file_path = self.config['data']['file_path']
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Basic data structure validation
            expected_cols = 406  # Based on your data structure
            if len(df.columns) < expected_cols:
                logger.warning(f"Expected {expected_cols} columns, got {len(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset."""
        if not self.config['data']['validation']['enabled']:
            logger.info("Data validation disabled, returning original data")
            return df
        
        logger.info("Starting data validation and cleaning...")
        start_time = time.time()
        
        try:
            cleaned_df, validation_report = validate_and_clean_data(df, self.config)
            
            # Store validation results
            self.results['data_validation'] = validation_report
            
            validation_time = time.time() - start_time
            logger.info(f"Data validation completed in {validation_time:.2f} seconds")
            logger.info(f"Records retained: {len(cleaned_df)}/{len(df)} ({len(cleaned_df)/len(df)*100:.1f}%)")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            # Return original data if validation fails
            logger.warning("Proceeding with original data")
            return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced features from physiological signals."""
        logger.info("Starting enhanced feature extraction...")
        start_time = time.time()
        
        try:
            feature_config = self.config.get('feature_engineering', {})
            feature_df = extract_enhanced_features(df, feature_config)
            
            # Store feature extraction results
            self.results['feature_extraction'] = {
                'original_records': len(df),
                'feature_records': len(feature_df),
                'num_features': len(feature_df.columns) - 4,  # Excluding patient_id, sbp, dbp, mean_ptt_original
                'extraction_time': time.time() - start_time
            }
            
            extraction_time = time.time() - start_time
            logger.info(f"Feature extraction completed in {extraction_time:.2f} seconds")
            logger.info(f"Extracted {len(feature_df.columns) - 4} features from {len(feature_df)} records")
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def train_models(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models with enhanced configurations."""
        logger.info("Starting model training...")
        start_time = time.time()
        
        try:
            # Prepare data for training
            X = feature_df.drop(['patient_id', 'sbp', 'dbp', 'mean_ptt_original'], axis=1)
            y_sbp = feature_df['sbp'].values
            y_dbp = feature_df['dbp'].values
            groups = feature_df['patient_id'].values
            
            # Train models
            training_results = train_enhanced_models(X, y_sbp, y_dbp, groups)
            
            # Store training results
            self.results['model_training'] = training_results
            self.results['model_training']['training_time'] = time.time() - start_time
            
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models with clinical metrics."""
        logger.info("Starting model evaluation...")
        start_time = time.time()
        
        try:
            evaluation_results = {}
            
            # Evaluate SBP models
            if 'sbp_models' in training_results or 'sbp_ensemble' in training_results:
                logger.info("Evaluating SBP models...")
                sbp_eval = evaluate_models(training_results, 'sbp')
                evaluation_results['sbp'] = sbp_eval
            
            # Evaluate DBP models
            if 'dbp_models' in training_results or 'dbp_ensemble' in training_results:
                logger.info("Evaluating DBP models...")
                dbp_eval = evaluate_models(training_results, 'dbp')
                evaluation_results['dbp'] = dbp_eval
            
            # Store evaluation results
            self.results['model_evaluation'] = evaluation_results
            self.results['model_evaluation']['evaluation_time'] = time.time() - start_time
            
            evaluation_time = time.time() - start_time
            logger.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def save_results(self, training_results: Dict[str, Any], 
                    evaluation_results: Dict[str, Any]) -> None:
        """Save models, results, and reports."""
        logger.info("Saving results...")
        
        try:
            # Create output directories
            model_dir = Path(self.config['output']['model_dir'])
            report_dir = Path(self.config['output']['report_dir'])
            plot_dir = Path(self.config['output']['plot_dir'])
            
            for directory in [model_dir, report_dir, plot_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Save models (if trainer has save method)
            try:
                from src.models.model_trainer import ModelTrainer
                trainer = ModelTrainer()
                trainer.save_models(training_results, str(model_dir))
            except Exception as e:
                logger.warning(f"Could not save models: {str(e)}")
            
            # Save evaluation reports
            for target in ['sbp', 'dbp']:
                if target in evaluation_results and '_summary' in evaluation_results[target]:
                    summary = evaluation_results[target]['_summary']
                    
                    # Save comparison table
                    if 'comparison_table' in summary:
                        comparison_path = report_dir / f'{target}_model_comparison.csv'
                        summary['comparison_table'].to_csv(comparison_path, index=False)
                        logger.info(f"Saved {target.upper()} comparison table to {comparison_path}")
                    
                    # Save detailed report
                    if 'report' in summary:
                        report_path = report_dir / f'{target}_evaluation_report.txt'
                        with open(report_path, 'w') as f:
                            f.write(summary['report'])
                        logger.info(f"Saved {target.upper()} evaluation report to {report_path}")
            
            # Save pipeline summary
            pipeline_summary = {
                'pipeline_config': self.config,
                'execution_results': self.results,
                'total_execution_time': sum([
                    self.results.get('model_training', {}).get('training_time', 0),
                    self.results.get('model_evaluation', {}).get('evaluation_time', 0),
                    self.results.get('feature_extraction', {}).get('extraction_time', 0)
                ])
            }
            
            summary_path = report_dir / 'pipeline_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(self._format_pipeline_summary(pipeline_summary))
            
            logger.info(f"Pipeline summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def _format_pipeline_summary(self, summary: Dict[str, Any]) -> str:
        """Format pipeline execution summary."""
        results = summary['execution_results']
        
        report = f"""
=== BLOOD PRESSURE PREDICTION PIPELINE SUMMARY ===

Execution Overview:
{'-' * 50}
Total Execution Time: {summary['total_execution_time']:.2f} seconds

Data Processing:
{'-' * 20}
"""
        
        if 'data_validation' in results:
            validation = results['data_validation']
            report += f"Original Records: {validation['original_records']}\n"
            report += f"Clean Records: {validation['cleaned_records']}\n"
            report += f"Data Quality: {validation['cleaned_records']/validation['original_records']*100:.1f}%\n"
        
        if 'feature_extraction' in results:
            features = results['feature_extraction']
            report += f"Features Extracted: {features['num_features']}\n"
            report += f"Feature Extraction Time: {features['extraction_time']:.2f}s\n"
        
        report += f"\nModel Training:\n{'-' * 20}\n"
        if 'model_training' in results:
            training = results['model_training']
            
            # Count models trained
            sbp_models = len(training.get('sbp_models', {}))
            dbp_models = len(training.get('dbp_models', {}))
            sbp_ensembles = len(training.get('sbp_ensemble', {}))
            dbp_ensembles = len(training.get('dbp_ensemble', {}))
            
            report += f"SBP Models Trained: {sbp_models}\n"
            report += f"DBP Models Trained: {dbp_models}\n"
            report += f"SBP Ensembles: {sbp_ensembles}\n"
            report += f"DBP Ensembles: {dbp_ensembles}\n"
            report += f"Training Time: {training['training_time']:.2f}s\n"
        
        report += f"\nModel Performance:\n{'-' * 20}\n"
        if 'model_evaluation' in results:
            evaluation = results['model_evaluation']
            
            for target in ['sbp', 'dbp']:
                if target in evaluation and '_summary' in evaluation[target]:
                    comparison_df = evaluation[target]['_summary'].get('comparison_table')
                    if comparison_df is not None and len(comparison_df) > 0:
                        best_model = comparison_df.iloc[0]
                        report += f"\nBest {target.upper()} Model: {best_model['Model']}\n"
                        report += f"  R²: {best_model['R²']:.3f}\n"
                        report += f"  RMSE: {best_model['RMSE']:.2f} mmHg\n"
                        report += f"  Within ±5 mmHg: {best_model['Within ±5 mmHg (%)']:.1f}%\n"
                        report += f"  BHS Grade: {best_model['BHS Grade']}\n"
        
        report += f"\n\nConfiguration Used:\n{'-' * 20}\n"
        report += f"Enhanced Feature Extraction: {summary['pipeline_config']['feature_engineering']['enhanced_extraction']}\n"
        report += f"Wavelet Features: {summary['pipeline_config']['feature_engineering']['wavelet_features']}\n"
        report += f"Ensemble Models: {summary['pipeline_config']['ensemble']['enabled']}\n"
        report += f"Clinical Validation: {summary['pipeline_config']['evaluation']['clinical_validation']}\n"
        
        return report
    
    def run_complete_pipeline(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete pipeline from data loading to model evaluation."""
        logger.info("="*60)
        logger.info("STARTING BLOOD PRESSURE PREDICTION PIPELINE")
        logger.info("="*60)
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Load data
            logger.info("Step 1/5: Loading data...")
            df = self.load_data(data_file)
            
            # Step 2: Validate and clean data
            logger.info("Step 2/5: Validating and cleaning data...")
            cleaned_df = self.validate_data(df)
            
            # Step 3: Extract features
            logger.info("Step 3/5: Extracting enhanced features...")
            feature_df = self.extract_features(cleaned_df)
            
            # Step 4: Train models
            logger.info("Step 4/5: Training models...")
            training_results = self.train_models(feature_df)
            
            # Step 5: Evaluate models
            logger.info("Step 5/5: Evaluating models...")
            evaluation_results = self.evaluate_models(training_results)
            
            # Save results
            if self.config['evaluation']['save_results']:
                self.save_results(training_results, evaluation_results)
            
            total_time = time.time() - pipeline_start_time
            
            logger.info("="*60)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
            logger.info("="*60)
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'pipeline_results': self.results,
                'total_execution_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def run_pipeline(config_file: Optional[str] = None, data_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point to run the complete blood pressure prediction pipeline.
    
    Args:
        config_file: Path to configuration file (optional)
        data_file: Path to data file (optional)
        
    Returns:
        Dictionary containing all pipeline results
    """
    # Load configuration
    config = None
    if config_file and Path(config_file).exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not available, using default configuration")
        except Exception as e:
            logger.warning(f"Could not load config file: {str(e)}")
    
    # Initialize and run pipeline
    pipeline = BloodPressurePipeline(config)
    return pipeline.run_complete_pipeline(data_file)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Blood Pressure Prediction Pipeline")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_pipeline(args.config, args.data)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"Check the output directory for detailed results and reports.")
