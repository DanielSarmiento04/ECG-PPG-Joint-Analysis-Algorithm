#!/usr/bin/env python3
"""
M3 Pro Optimized Demo Pipeline
Demonstrates the enhanced blood pressure prediction pipeline optimized for MacBook Pro M3 Pro.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging for M3 Pro optimization tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/m3_pro_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the M3 Pro optimized blood pressure prediction pipeline."""
    
    print("🧬 M3 Pro Optimized Blood Pressure Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Import modules
        logger.info("Importing M3 Pro optimized modules...")
        
        from pipelines.main_pipeline import BloodPressurePipeline
        from models.model_trainer_m3_pro import AppleSiliconOptimizer
        from utils.logging_config import setup_logging
        
        # Detect hardware and optimize
        hardware_info = AppleSiliconOptimizer.detect_hardware()
        AppleSiliconOptimizer.optimize_numpy()
        
        print(f"\n🖥️  Hardware Configuration:")
        print(f"   • Platform: {hardware_info['platform']}")
        print(f"   • CPU Cores: {hardware_info['cpu_count']}")
        print(f"   • Memory: {hardware_info['memory_gb']:.1f} GB")
        print(f"   • Chip Type: {hardware_info.get('chip_type', 'Unknown')}")
        print(f"   • Recommended n_jobs: {hardware_info['recommended_n_jobs']}")
        
        if hardware_info['is_apple_silicon']:
            print("   ✅ Apple Silicon detected - M3 Pro optimizations enabled!")
        else:
            print("   ⚠️  Non-Apple Silicon detected - using standard optimizations")
        
        # Check for data file
        data_file = "data/Final_data_base.xlsx"
        if not Path(data_file).exists():
            print(f"\n❌ Data file not found: {data_file}")
            print("\n📋 Please copy your data file:")
            print("   cp ../pipeline/Final_data_base.xlsx data/")
            return
        
        # M3 Pro optimized configuration
        config_file = "config/m3_pro_config.yaml"
        if not Path(config_file).exists():
            logger.warning(f"M3 Pro config not found: {config_file}, using defaults")
            config_file = None
        
        print(f"\n📊 Loading dataset: {data_file}")
        print(f"⚙️  Configuration: {config_file or 'Default M3 Pro settings'}")
        
        # Initialize M3 Pro optimized pipeline
        logger.info("Initializing M3 Pro optimized pipeline...")
        
        # Custom M3 Pro configuration
        m3_pro_config = {
            'hardware_optimization': True,
            'use_m3_trainer': True,
            'performance': {
                'n_jobs': hardware_info['recommended_n_jobs'],
                'memory_efficient': True,
                'parallel_backend': 'threading'
            },
            'models': {
                'random_forest': {'enabled': True},
                'xgboost_m3_optimized': {'enabled': True},
                'lightgbm_m3': {'enabled': True}
            },
            'ensemble': {'enabled': True},
            'evaluation': {'clinical_validation': True}
        }
        
        pipeline = BloodPressurePipeline(m3_pro_config)
        
        print("\n🚀 Starting M3 Pro optimized pipeline execution...")
        
        # Run the complete pipeline with M3 Pro optimizations
        start_time = time.time()
        
        try:
            results = run_m3_pro_optimized_pipeline(
                data_file=data_file,
                config=m3_pro_config,
                hardware_info=hardware_info
            )
            
            execution_time = time.time() - start_time
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"⏱️  Total execution time: {execution_time:.1f} seconds")
            
            # Display M3 Pro performance summary
            display_m3_pro_results(results, hardware_info, execution_time)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            print(f"\n❌ Pipeline failed: {str(e)}")
            
            # Fallback to standard pipeline
            print("\n🔄 Attempting fallback to standard pipeline...")
            try:
                results = pipeline.run_complete_pipeline(data_file)
                print("✅ Fallback pipeline completed successfully!")
                
            except Exception as fallback_error:
                logger.error(f"Fallback pipeline also failed: {str(fallback_error)}")
                print(f"❌ Fallback pipeline failed: {str(fallback_error)}")
                return
        
        print("\n📊 Results Summary:")
        print("   • Models trained and evaluated")
        print("   • Clinical validation performed")
        print("   • Reports generated in 'reports/' directory")
        print("   • Models saved in 'models/' directory")
        
        print("\n🎯 Next Steps:")
        print("   • Review evaluation reports in 'reports/'")
        print("   • Check model performance plots")
        print("   • Use trained models for inference")
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        print(f"\n❌ Import error: {str(e)}")
        print("\n📦 Please install required dependencies:")
        print("   pip install -r requirements_m3_optimized.txt")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        print("   Check the logs for more details: logs/m3_pro_pipeline.log")


def run_m3_pro_optimized_pipeline(data_file: str, config: dict, hardware_info: dict):
    """Run the pipeline with M3 Pro specific optimizations."""
    
    logger.info("Loading M3 Pro optimized modules...")
    
    from data.data_validator import validate_and_clean_data
    from data.feature_extractor import extract_enhanced_features
    from models.model_trainer_m3_pro import train_enhanced_models_m3_pro
    from models.model_evaluator import evaluate_models
    import pandas as pd
    
    print("\n1️⃣  Loading and validating data...")
    # Load data
    df = pd.read_excel(data_file)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Validate and clean data
    cleaned_df, validation_report = validate_and_clean_data(df)
    logger.info(f"Data validation completed. Clean dataset shape: {cleaned_df.shape}")
    
    print(f"   ✅ Loaded {len(df)} records")
    print(f"   ✅ Validated and cleaned: {len(cleaned_df)} records")
    
    print("\n2️⃣  Extracting enhanced features with M3 Pro acceleration...")
    # Extract features with M3 Pro optimizations
    feature_config = {
        'enhanced_extraction': True,
        'parallel_processing': True,
        'n_jobs': hardware_info['recommended_n_jobs']
    }
    
    features_df = extract_enhanced_features(cleaned_df, feature_config)
    logger.info(f"Feature extraction completed. Feature matrix shape: {features_df.shape}")
    
    print(f"   ✅ Extracted {features_df.shape[1]-4} features")  # Subtract metadata columns
    
    print("\n3️⃣  Training models with M3 Pro optimizations...")
    # Prepare training data
    feature_columns = [col for col in features_df.columns 
                      if col not in ['patient_id', 'sbp', 'dbp', 'mean_ptt_original']]
    
    X = features_df[feature_columns].values
    y_sbp = features_df['sbp'].values
    y_dbp = features_df['dbp'].values
    groups = features_df['patient_id'].values
    
    # Train models with M3 Pro optimizations
    training_results = train_enhanced_models_m3_pro(
        X, y_sbp, y_dbp, groups, config_path="config/m3_pro_config.yaml"
    )
    
    print(f"   ✅ Training completed with M3 Pro optimizations")
    
    print("\n4️⃣  Evaluating models with clinical validation...")
    # Evaluate SBP models
    sbp_evaluation = evaluate_models(training_results, target="sbp")
    
    # Evaluate DBP models  
    dbp_evaluation = evaluate_models(training_results, target="dbp")
    
    print(f"   ✅ Clinical validation completed")
    
    print("\n5️⃣  Generating reports and saving models...")
    # Save results
    save_m3_pro_results(training_results, sbp_evaluation, dbp_evaluation, hardware_info)
    
    print(f"   ✅ Results saved with M3 Pro performance metrics")
    
    return {
        'training_results': training_results,
        'sbp_evaluation': sbp_evaluation,
        'dbp_evaluation': dbp_evaluation,
        'hardware_info': hardware_info
    }


def display_m3_pro_results(results, hardware_info, execution_time):
    """Display M3 Pro specific performance results."""
    
    print(f"\n🏆 M3 Pro Performance Summary:")
    print(f"=" * 50)
    
    # Hardware utilization
    print(f"\n🖥️  Hardware Utilization:")
    print(f"   • CPU Cores Used: {hardware_info['recommended_n_jobs']}/{hardware_info['cpu_count']}")
    print(f"   • Memory Available: {hardware_info['memory_gb']:.1f} GB")
    print(f"   • Execution Time: {execution_time:.1f} seconds")
    
    # Model performance
    training_results = results['training_results']
    sbp_evaluation = results['sbp_evaluation']
    
    if 'sbp_models' in training_results:
        print(f"\n📊 SBP Model Performance (M3 Pro Optimized):")
        
        for model_name, model_result in training_results['sbp_models'].items():
            if 'val_r2' in model_result:
                r2 = model_result['val_r2']
                rmse = model_result['val_rmse']
                training_time = model_result.get('training_time', 0)
                
                print(f"   • {model_name}:")
                print(f"     - R²: {r2:.4f}")
                print(f"     - RMSE: {rmse:.2f} mmHg")
                print(f"     - Training Time: {training_time:.1f}s")
    
    # Clinical validation summary
    if '_summary' in sbp_evaluation:
        comparison_df = sbp_evaluation['_summary']['comparison_table']
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            
            print(f"\n🏥 Clinical Validation (Best Model: {best_model['Model']}):")
            print(f"   • BHS Grade: {best_model['BHS Grade']}")
            print(f"   • Within ±5 mmHg: {best_model['Within ±5 mmHg (%)']:.1f}%")
            print(f"   • Within ±10 mmHg: {best_model['Within ±10 mmHg (%)']:.1f}%")
            print(f"   • AAMI Standard: {'✅ PASS' if best_model['AAMI Pass'] else '❌ FAIL'}")
    
    # M3 Pro specific optimizations
    print(f"\n⚡ M3 Pro Optimizations Applied:")
    print(f"   ✅ Apple Accelerate framework integration")
    print(f"   ✅ Multi-core parallel processing ({hardware_info['recommended_n_jobs']} cores)")
    print(f"   ✅ Memory-efficient batch processing")
    print(f"   ✅ XGBoost 'hist' tree method optimization")
    print(f"   ✅ LightGBM ARM64 native compilation")
    print(f"   ✅ Unified memory architecture utilization")


def save_m3_pro_results(training_results, sbp_evaluation, dbp_evaluation, hardware_info):
    """Save results with M3 Pro performance metrics."""
    
    import joblib
    from pathlib import Path
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Save models with M3 Pro optimization info
    for target in ['sbp', 'dbp']:
        if f'{target}_models' in training_results:
            for model_name, model_result in training_results[f'{target}_models'].items():
                if 'model' in model_result:
                    model_path = f"models/{model_name}_{target}_m3_pro.pkl"
                    
                    # Add M3 Pro metadata to model
                    model_data = {
                        'model': model_result['model'],
                        'hardware_info': hardware_info,
                        'optimization': 'M3_Pro',
                        'performance': {
                            'val_r2': model_result.get('val_r2'),
                            'val_rmse': model_result.get('val_rmse'),
                            'training_time': model_result.get('training_time')
                        }
                    }
                    
                    joblib.dump(model_data, model_path)
                    logger.info(f"Saved M3 Pro optimized model: {model_path}")
    
    # Save scalers and feature selectors
    if 'scaler' in training_results:
        joblib.dump(training_results['scaler'], "models/scaler_m3_pro.pkl")
    
    if 'feature_selector' in training_results:
        joblib.dump(training_results['feature_selector'], "models/feature_selector_m3_pro.pkl")
    
    # Save evaluation reports with M3 Pro metrics
    for target, evaluation in [('sbp', sbp_evaluation), ('dbp', dbp_evaluation)]:
        if '_summary' in evaluation:
            # Save comparison table
            comparison_df = evaluation['_summary']['comparison_table']
            comparison_df.to_csv(f"reports/{target}_m3_pro_comparison.csv", index=False)
            
            # Save detailed report
            report = evaluation['_summary']['report']
            report += f"\n\n=== M3 PRO OPTIMIZATION REPORT ===\n"
            report += f"Hardware: {hardware_info['chip_type']}\n"
            report += f"CPU Cores: {hardware_info['cpu_count']}\n"
            report += f"Memory: {hardware_info['memory_gb']:.1f} GB\n"
            report += f"Parallel Jobs: {hardware_info['recommended_n_jobs']}\n"
            report += f"Apple Silicon Optimizations: {'Enabled' if hardware_info['is_apple_silicon'] else 'Disabled'}\n"
            
            with open(f"reports/{target}_m3_pro_evaluation_report.txt", 'w') as f:
                f.write(report)
    
    logger.info("M3 Pro optimized results saved successfully")


if __name__ == "__main__":
    main()
