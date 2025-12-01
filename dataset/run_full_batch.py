from bp_pipeline import BPEstimationPipeline
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
        
        # Save the enhanced dataset
        output_path = './data/processed/bp_dataset_features.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved enhanced dataset to {output_path}")
        
        # Print correlation analysis
        print("\n" + "="*60)
        print("FEATURE-BP CORRELATIONS (with new PAT features)")
        print("="*60)
        
        key_features = [
            'pat_ecg_ppg', 'pat_to_peak', 'pat_to_maxslope',
            'pat_ecg_ppg_norm', 'pat_ecg_ppg_delta',
            'hr_bpm', 'hr_bpm_norm',
            'amplitude_ratio_ra', 'reflection_index'
        ]
        
        for feat in key_features:
            if feat in df.columns:
                corr_sbp = df[feat].corr(df['sbp_reference'])
                corr_dbp = df[feat].corr(df['dbp_reference'])
                # Also check correlation with BP delta (changes)
                corr_sbp_delta = df[feat].corr(df.get('sbp_delta', df['sbp_reference']))
                print(f"{feat:25s}: SBP r={corr_sbp:+.3f}, DBP r={corr_dbp:+.3f}, SBP_delta r={corr_sbp_delta:+.3f}")
        
        print("\nDataset Summary:")
        print(df.describe())
    else:
        logger.warning("No features extracted. Check logs for errors.")

if __name__ == "__main__":
    main()
