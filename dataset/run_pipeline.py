#!/usr/bin/env python3
"""
Blood Pressure Dataset Pipeline - Rebuilt Version

This script provides a complete pipeline for:
1. Extracting features from raw VitalDB data
2. Assessing data quality (before cleaning)
3. Cleaning invalid samples
4. Validating the cleaned data
5. Generating quality reports and visualizations

Addresses critical data quality issues:
- PTT = 0 samples (93% of data in original)
- Amplitude ratio = 1.0 (default values)
- Invalid BP ranges
- Poor waveform quality

Author: Blood Pressure Pipeline Team
Date: December 2024
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import pipeline modules
from bp_pipeline import BPEstimationPipeline
from data_cleaning import DataCleaner, quick_quality_check
from quality_assessment import generate_quality_report, print_comparison
from validate_pipeline import validate_cleaned_data, compare_before_after

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    data_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    config_path: str = 'pipeline_config.yaml',
    max_files: int | None = None,
    strict_cleaning: bool = True,
    generate_plots: bool = True
):
    """
    Run the complete BP dataset pipeline with quality controls.
    
    Args:
        data_dir: Directory containing raw .npz files
        output_dir: Directory to save processed data
        config_path: Path to pipeline configuration
        max_files: Maximum number of files to process (None = all)
        strict_cleaning: If True, apply strict data cleaning
        generate_plots: If True, generate validation plots
        
    Returns:
        Final cleaned DataFrame
    """
    print("=" * 80)
    print("BLOOD PRESSURE DATASET PIPELINE - REBUILT VERSION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(os.path.dirname(output_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # =========================================
    # STEP 1: Extract features from raw data
    # =========================================
    print("\n[1/5] EXTRACTING FEATURES FROM RAW DATA")
    print("-" * 60)
    
    try:
        pipeline = BPEstimationPipeline(config_path)
        raw_df = pipeline.run_batch(data_dir, max_files=max_files)
        
        if raw_df.empty:
            logger.error("No data extracted from raw files!")
            return pd.DataFrame()
        
        # Save raw extracted data
        raw_output = os.path.join(output_dir, 'bp_dataset_raw.csv')
        raw_df.to_csv(raw_output, index=False)
        logger.info(f"Saved raw features to: {raw_output}")
        
        print(f"    ✓ Extracted: {len(raw_df):,} samples")
        print(f"    ✓ Patients: {raw_df['patient_id'].nunique() if 'patient_id' in raw_df.columns else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # =========================================
    # STEP 2: Quality assessment BEFORE cleaning
    # =========================================
    print("\n[2/5] ASSESSING RAW DATA QUALITY")
    print("-" * 60)
    
    report_before = generate_quality_report(raw_df, dataset_path=raw_output)
    report_before.print_summary()
    
    # Save pre-cleaning report
    report_before.to_json(os.path.join(output_dir, 'quality_report_before.json'))
    
    # Quick summary
    quick_stats = quick_quality_check(raw_df)
    print(f"\n    Quick Stats:")
    print(f"    - PTT zeros: {quick_stats.get('ptt_zeros', 0):,} ({quick_stats.get('ptt_zeros_pct', 0):.1f}%)")
    print(f"    - Amplitude = 1.0: {quick_stats.get('amp_ratio_ones', 0):,} ({quick_stats.get('amp_ratio_ones_pct', 0):.1f}%)")
    print(f"    - SBP <= DBP violations: {quick_stats.get('sbp_dbp_violations', 0):,}")
    
    # =========================================
    # STEP 3: Clean invalid samples
    # =========================================
    print("\n[3/5] CLEANING INVALID SAMPLES")
    print("-" * 60)
    
    cleaner = DataCleaner(strict_mode=strict_cleaning)
    clean_df = cleaner.filter_invalid_samples(raw_df)
    
    cleaning_stats = cleaner.get_cleaning_stats()
    print(f"\n    ✓ Initial samples: {cleaning_stats['initial_count']:,}")
    print(f"    ✓ Removed: {cleaning_stats['removed_count']:,} ({cleaning_stats['removal_percentage']:.1f}%)")
    print(f"    ✓ Remaining: {cleaning_stats['final_count']:,}")
    
    # =========================================
    # STEP 4: Quality assessment AFTER cleaning
    # =========================================
    print("\n[4/5] ASSESSING CLEANED DATA QUALITY")
    print("-" * 60)
    
    clean_output = os.path.join(output_dir, 'bp_dataset_cleaned_v2.csv')
    clean_df.to_csv(clean_output, index=False)
    
    report_after = generate_quality_report(clean_df, dataset_path=clean_output)
    report_after.print_summary()
    
    # Save post-cleaning report
    report_after.to_json(os.path.join(output_dir, 'quality_report_after.json'))
    
    # Compare before/after
    print_comparison(report_before, report_after)
    
    # =========================================
    # STEP 5: Validation and visualization
    # =========================================
    if generate_plots:
        print("\n[5/5] GENERATING VALIDATION PLOTS")
        print("-" * 60)
        
        try:
            # Validate cleaned data
            validation_metrics = validate_cleaned_data(
                clean_df, 
                output_dir=figures_dir,
                prefix="cleaned"
            )
            
            # Before/after comparison
            compare_before_after(raw_df, clean_df, output_dir=figures_dir)
            
            print(f"\n    ✓ Plots saved to: {figures_dir}")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
            print(f"    ⚠ Plot generation failed: {e}")
    
    # =========================================
    # FINAL SUMMARY
    # =========================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    print(f"\nOutput Files:")
    print(f"  - Raw features:    {raw_output}")
    print(f"  - Cleaned dataset: {clean_output}")
    print(f"  - Quality reports: {output_dir}/quality_report_*.json")
    if generate_plots:
        print(f"  - Validation plots: {figures_dir}/")
    
    # Final quality check
    print(f"\nFinal Quality Metrics:")
    print(f"  - Samples: {len(clean_df):,}")
    print(f"  - PTT-SBP correlation: {report_after.ptt_sbp_correlation:.3f}")
    print(f"  - PTT-DBP correlation: {report_after.ptt_dbp_correlation:.3f}")
    
    # Warnings/recommendations
    if not report_after.has_valid_ptt_bp_relationship:
        print(f"\n⚠️  WARNING: PTT-SBP correlation ({report_after.ptt_sbp_correlation:.3f}) is not strongly negative!")
        print("   Expected: < -0.5")
        print("   The feature extraction may still have issues.")
        print("   Consider examining individual cases manually.")
    else:
        print(f"\n✓ PTT-BP correlation looks good ({report_after.ptt_sbp_correlation:.3f})")
    
    if len(clean_df) < 500000:
        print(f"\n⚠️  WARNING: Only {len(clean_df):,} samples remaining")
        print("   This might not be enough for deep learning.")
        print("   Consider:")
        print("   - Relaxing quality thresholds")
        print("   - Processing more raw data files")
        print("   - Using data augmentation")
    else:
        print(f"\n✓ Sufficient data: {len(clean_df):,} samples")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return clean_df


def quick_test(n_files: int = 2):
    """
    Quick test of the pipeline with a small number of files.
    
    Args:
        n_files: Number of files to process
    """
    print("=" * 80)
    print("QUICK PIPELINE TEST")
    print("=" * 80)
    
    return run_full_pipeline(
        data_dir='data/raw',
        output_dir='data/processed',
        config_path='pipeline_config.yaml',
        max_files=n_files,
        strict_cleaning=True,
        generate_plots=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Blood Pressure Dataset Pipeline')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing raw .npz files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--config', type=str, default='pipeline_config.yaml',
                       help='Path to pipeline configuration')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--no-strict', action='store_true',
                       help='Disable strict data cleaning')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 2 files')
    
    args = parser.parse_args()
    
    if args.quick_test:
        df = quick_test(n_files=2)
    else:
        df = run_full_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            max_files=args.max_files,
            strict_cleaning=not args.no_strict,
            generate_plots=not args.no_plots
        )
    
    if not df.empty:
        print(f"\nPipeline completed successfully with {len(df):,} samples")
    else:
        print("\nPipeline failed - no data produced")
        sys.exit(1)
