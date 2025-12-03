from bp_pipeline import BPEstimationPipeline
from fixed_feature_extraction import filter_valid_patients, validate_patient_physiology
import os
import logging
import argparse
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_per_patient_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-patient normalized versions of key features.
    
    PTT and other features vary significantly between individuals due to different
    vascular properties (stiffness, compliance, etc.). Normalizing within each patient
    removes this inter-individual variability and focuses on BP-related changes.
    
    This is critical for BP prediction because:
    - Absolute PTT varies ~100ms between individuals with same BP
    - Changes in PTT within a patient correlate better with BP changes
    """
    logger.info("Adding per-patient normalized features...")
    
    # Features to normalize per patient
    features_to_normalize = [
        'pat_ecg_ppg',       # TRUE ECG-PPG PTT (most important!)
        'pat_to_peak',
        'pat_to_maxslope', 
        'ptt_peak_to_foot',
        'ptt_peak_to_peak',
        'amplitude_ratio_ra',
        'systolic_duration_tsd',
        'diastolic_duration_tfd',
        'hr_bpm',
        'max_upslope',
        'reflection_index',
        'systolic_area_ratio',
        # NEW: Statistical features
        'stat_mean',
        'stat_std',
        'stat_skew',
        'stat_kurtosis',
        'stat_power_cardiac',
        'stat_power_low',
        'stat_power_ratio'
    ]
    
    for feat in features_to_normalize:
        if feat not in df.columns:
            continue
            
        # Z-score normalization per patient
        norm_col = f'{feat}_norm'
        df[norm_col] = df.groupby('patient_id')[feat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        
        # Also add delta from patient baseline (first 5 minutes)
        delta_col = f'{feat}_delta'
        def calc_delta(group):
            # Use median of first 100 samples as baseline
            baseline = group.head(100).median()
            return group - baseline
        df[delta_col] = df.groupby('patient_id')[feat].transform(calc_delta)
    
    # Add BP variability features (changes from baseline)
    if 'sbp_reference' in df.columns:
        df['sbp_delta'] = df.groupby('patient_id')['sbp_reference'].transform(
            lambda x: x - x.head(100).median()
        )
    if 'dbp_reference' in df.columns:
        df['dbp_delta'] = df.groupby('patient_id')['dbp_reference'].transform(
            lambda x: x - x.head(100).median()
        )
    
    logger.info(f"Added {len(features_to_normalize) * 2} normalized feature columns")
    return df

def main():
    config_path = 'pipeline_config.yaml'
    raw_data_dir = './data/raw'
    
    
    parser = argparse.ArgumentParser(description='Run full batch processing of BP estimation pipeline.')

    # data paremeters
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process. Default is None (process all files).")
    
    args = parser.parse_args()
    
    if not os.path.exists(raw_data_dir):
        logger.error(f"Data directory {raw_data_dir} not found!")
        return

    logger.info("Initializing BP Estimation Pipeline...")
    pipeline = BPEstimationPipeline(config_path)
    
    logger.info("Starting batch processing of all files...")
    # Set max_files to the value provided by the user
    df = pipeline.run_batch(raw_data_dir, max_files=args.max_files)
    
    if not df.empty:
        logger.info("Batch processing complete.")
        logger.info(f"Total records extracted: {len(df)}")
        logger.info(f"Unique patients: {df['patient_id'].nunique()}")
        
        # Add per-patient normalized features
        df = add_per_patient_normalized_features(df)
        
        # Save the full dataset (before validation filtering)
        output_path = './data/processed/bp_dataset_features.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved full dataset to {output_path}")
        
        # ============================================
        # NEW: Patient physiology validation
        # Filter out patients with wrong PTT-BP relationships
        # ============================================
        print("\n" + "="*60)
        print("PATIENT PHYSIOLOGY VALIDATION")
        print("="*60)
        
        # Permissive validation (more samples, weaker correlation)
        print("\n--- Permissive Mode (more samples) ---")
        df_validated, patient_metrics = filter_valid_patients(df, strict=False, verbose=True)
        
        # Strict validation (fewer samples, stronger correlation)
        print("\n--- Strict Mode (better correlation) ---")
        df_strict, strict_metrics = filter_valid_patients(df, strict=True, verbose=True)
        
        if len(df_validated) > 0:
            # Save the permissive validated dataset
            validated_output_path = './data/processed/bp_dataset_validated.csv'
            df_validated.to_csv(validated_output_path, index=False)
            logger.info(f"Saved validated dataset to {validated_output_path}")
        
        if len(df_strict) > 0:
            # Save the strict validated dataset (highest quality)
            strict_output_path = './data/processed/bp_dataset_strict.csv'
            df_strict.to_csv(strict_output_path, index=False)
            logger.info(f"Saved strict dataset to {strict_output_path}")
            
            # Save patient metrics for analysis
            metrics_df = pd.DataFrame.from_dict(patient_metrics, orient='index')
            metrics_df.index.name = 'patient_id'
            metrics_df.to_csv('./data/processed/patient_validation_metrics.csv')
            logger.info("Saved patient validation metrics to patient_validation_metrics.csv")
        
        # Print correlation analysis (using strict validated data for best results)
        print("\n" + "="*60)
        print("FEATURE-BP CORRELATIONS (STRICT VALIDATED PATIENTS)")
        print("="*60)
        
        key_features = [
            'pat_ecg_ppg', 'pat_to_peak', 'pat_to_maxslope',
            'pat_ecg_ppg_norm', 'pat_ecg_ppg_delta',
            'hr_bpm', 'hr_bpm_norm',
            'amplitude_ratio_ra', 'reflection_index'
        ]
        
        # Use strict validated data for correlation analysis (best quality)
        analysis_df = df_strict if len(df_strict) > 0 else (df_validated if len(df_validated) > 0 else df)
        
        for feat in key_features:
            if feat in analysis_df.columns:
                corr_sbp = analysis_df[feat].corr(analysis_df['sbp_reference'])
                corr_dbp = analysis_df[feat].corr(analysis_df['dbp_reference'])
                # Also check correlation with BP delta (changes)
                corr_sbp_delta = analysis_df[feat].corr(analysis_df.get('sbp_delta', analysis_df['sbp_reference']))
                print(f"{feat:25s}: SBP r={corr_sbp:+.3f}, DBP r={corr_dbp:+.3f}, SBP_delta r={corr_sbp_delta:+.3f}")
        
        # Also show full dataset correlations for comparison
        if len(df_validated) > 0 and len(df_validated) != len(df):
            print("\n" + "-"*60)
            print("COMPARISON: Full dataset (unvalidated)")
            print("-"*60)
            for feat in ['pat_ecg_ppg', 'ptt_peak_to_foot']:
                if feat in df.columns:
                    corr_sbp = df[feat].corr(df['sbp_reference'])
                    print(f"{feat:25s}: SBP r={corr_sbp:+.3f}")
        
        print("\nDataset Summary:")
        print(analysis_df.describe())
    else:
        logger.warning("No features extracted. Check logs for errors.")

if __name__ == "__main__":
    main()
