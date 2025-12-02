#!/usr/bin/env python3
"""
Data Cleaning Module for Blood Pressure Prediction Pipeline

This module addresses the critical data quality issues identified in the 
root cause analysis:
1. PTT = 0 (93% of samples) - physiologically impossible
2. Amplitude ratio = 1.0 - default/fallback value indicating extraction failure
3. Invalid BP ranges and SBP <= DBP violations
4. Low cycle correlation indicating poor waveform quality

Author: Blood Pressure Pipeline Team
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaning for blood pressure dataset with configurable thresholds.
    
    Addresses issues:
    - PTT = 0 samples (detection failures)
    - Impossible PTT values (outside physiological range)
    - Invalid BP values (outside clinical range)
    - SBP <= DBP violations
    - Amplitude ratio = 1.0 (default values)
    - Low cycle correlation (poor waveform quality)
    """
    
    # Default physiological thresholds
    DEFAULT_THRESHOLDS = {
        # PTT thresholds (milliseconds) - within-cycle timing
        # ptt_peak_to_peak: time from cycle start to systolic peak (typically 50-400ms)
        'ptt_peak_to_peak_min': 20,   # Very fast rise
        'ptt_peak_to_peak_max': 600,  # Slow rise (low HR, elderly)
        # ptt_peak_to_foot: time from cycle start to PPG foot (typically 20-200ms)
        'ptt_peak_to_foot_min': 10,   # Allow fast transit
        'ptt_peak_to_foot_max': 300,  # Allow slower transit
        
        # PAT thresholds (R-peak to PPG foot) - TRUE pulse transit time
        # This is the physiologically meaningful measure (typically 100-350ms)
        'pat_ecg_ppg_min': 10,    # Minimum PAT (fast)
        'pat_ecg_ppg_max': 500,   # Maximum PAT (slow)
        
        # Blood pressure thresholds (mmHg)
        'sbp_min': 60,
        'sbp_max': 220,
        'dbp_min': 30,
        'dbp_max': 130,
        
        # Heart rate thresholds (BPM)
        'hr_min': 30,
        'hr_max': 220,
        
        # Amplitude ratio thresholds
        'amplitude_ratio_min': 0.01,
        'amplitude_ratio_max': 0.99,  # Exclude exactly 1.0 (default value)
        
        # Quality thresholds
        'min_cycle_correlation': 0.7,
    }
    
    def __init__(self, thresholds: Optional[Dict] = None, strict_mode: bool = True):
        """
        Initialize the data cleaner.
        
        Args:
            thresholds: Custom thresholds dictionary (overrides defaults)
            strict_mode: If True, apply all filters. If False, only remove clearly invalid data.
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        self.strict_mode = strict_mode
        self.cleaning_stats: Dict[str, Any] = {}
        
    def filter_invalid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove physiologically impossible samples from the dataset.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Cleaned DataFrame with invalid samples removed
        """
        initial_count = len(df)
        self.cleaning_stats = {'initial_count': initial_count}
        
        logger.info(f"Starting data cleaning. Initial samples: {initial_count:,}")
        
        # Track removals per rule
        removal_counts = {}
        
        # Rule 1: Remove PTT = 0 (detection failure)
        if 'ptt_peak_to_peak' in df.columns:
            ptt_zeros = (df['ptt_peak_to_peak'] == 0).sum()
            df = df[df['ptt_peak_to_peak'] > 0]
            removal_counts['ptt_peak_zero'] = ptt_zeros
            logger.info(f"  Rule 1a: Removed {ptt_zeros:,} samples with PTT peak-to-peak = 0")
            
        if 'ptt_peak_to_foot' in df.columns:
            ptt_foot_zeros = (df['ptt_peak_to_foot'] == 0).sum()
            df = df[df['ptt_peak_to_foot'] > 0]
            removal_counts['ptt_foot_zero'] = ptt_foot_zeros
            logger.info(f"  Rule 1b: Removed {ptt_foot_zeros:,} samples with PTT peak-to-foot = 0")
        
        # Rule 2: Remove impossible PTT values
        if 'ptt_peak_to_peak' in df.columns:
            ptt_min = self.thresholds['ptt_peak_to_peak_min']
            ptt_max = self.thresholds['ptt_peak_to_peak_max']
            before = len(df)
            df = df[df['ptt_peak_to_peak'].between(ptt_min, ptt_max)]
            removal_counts['ptt_peak_range'] = before - len(df)
            logger.info(f"  Rule 2a: Removed {before - len(df):,} samples with PTT peak-to-peak outside [{ptt_min}, {ptt_max}] ms")
            
        if 'ptt_peak_to_foot' in df.columns:
            ptt_min = self.thresholds['ptt_peak_to_foot_min']
            ptt_max = self.thresholds['ptt_peak_to_foot_max']
            before = len(df)
            df = df[df['ptt_peak_to_foot'].between(ptt_min, ptt_max)]
            removal_counts['ptt_foot_range'] = before - len(df)
            logger.info(f"  Rule 2b: Removed {before - len(df):,} samples with PTT peak-to-foot outside [{ptt_min}, {ptt_max}] ms")
        
        # Rule 3: Remove impossible BP values
        if 'sbp_reference' in df.columns and 'dbp_reference' in df.columns:
            sbp_min = self.thresholds['sbp_min']
            sbp_max = self.thresholds['sbp_max']
            dbp_min = self.thresholds['dbp_min']
            dbp_max = self.thresholds['dbp_max']
            
            before = len(df)
            df = df[df['sbp_reference'].between(sbp_min, sbp_max)]
            removal_counts['sbp_range'] = before - len(df)
            logger.info(f"  Rule 3a: Removed {before - len(df):,} samples with SBP outside [{sbp_min}, {sbp_max}] mmHg")
            
            before = len(df)
            df = df[df['dbp_reference'].between(dbp_min, dbp_max)]
            removal_counts['dbp_range'] = before - len(df)
            logger.info(f"  Rule 3b: Removed {before - len(df):,} samples with DBP outside [{dbp_min}, {dbp_max}] mmHg")
            
            # SBP must be greater than DBP
            before = len(df)
            df = df[df['sbp_reference'] > df['dbp_reference']]
            removal_counts['sbp_dbp_violation'] = before - len(df)
            logger.info(f"  Rule 3c: Removed {before - len(df):,} samples with SBP <= DBP")
        
        # Rule 4: Remove amplitude ratio = 1.0 (default value) - OPTIONAL
        # Dicrotic notch detection is difficult; amplitude_ratio = 1.0 doesn't invalidate PTT
        if 'amplitude_ratio_ra' in df.columns and self.strict_mode:
            # Only filter in strict mode - amplitude ratio is not critical for PTT estimation
            before = len(df)
            # Instead of removing all 1.0, just flag them
            # df = df[df['amplitude_ratio_ra'] != 1.0]
            # removal_counts['amp_ratio_default'] = before - len(df)
            # logger.info(f"  Rule 4a: Removed {before - len(df):,} samples with amplitude_ratio = 1.0 (default)")
            removal_counts['amp_ratio_default'] = 0
            logger.info(f"  Rule 4a: Skipped (amplitude_ratio = 1.0 not filtered, {(df['amplitude_ratio_ra'] == 1.0).sum()} samples)")
            
            amp_min = self.thresholds['amplitude_ratio_min']
            amp_max = self.thresholds['amplitude_ratio_max']
            before = len(df)
            # Only filter clearly invalid values (negative or > 1.0)
            df = df[(df['amplitude_ratio_ra'] >= 0) & (df['amplitude_ratio_ra'] <= 1.0)]
            removal_counts['amp_ratio_range'] = before - len(df)
            logger.info(f"  Rule 4b: Removed {before - len(df):,} samples with amplitude_ratio outside [0, 1.0]")
        
        # Rule 5: Remove impossible HR values
        if 'hr_bpm' in df.columns:
            hr_min = self.thresholds['hr_min']
            hr_max = self.thresholds['hr_max']
            before = len(df)
            df = df[df['hr_bpm'].between(hr_min, hr_max)]
            removal_counts['hr_range'] = before - len(df)
            logger.info(f"  Rule 5: Removed {before - len(df):,} samples with HR outside [{hr_min}, {hr_max}] BPM")
        
        # Rule 6: Remove low correlation cycles (poor waveform quality)
        if self.strict_mode and 'cycle_correlation' in df.columns:
            min_corr = self.thresholds['min_cycle_correlation']
            before = len(df)
            df = df[df['cycle_correlation'] >= min_corr]
            removal_counts['low_correlation'] = before - len(df)
            logger.info(f"  Rule 6: Removed {before - len(df):,} samples with correlation < {min_corr}")
        
        # Rule 7: Remove PAT values outside range (if available)
        if 'pat_ecg_ppg' in df.columns:
            pat_min = self.thresholds['pat_ecg_ppg_min']
            pat_max = self.thresholds['pat_ecg_ppg_max']
            before = len(df)
            df = df[df['pat_ecg_ppg'].between(pat_min, pat_max)]
            removal_counts['pat_range'] = before - len(df)
            logger.info(f"  Rule 7: Removed {before - len(df):,} samples with PAT outside [{pat_min}, {pat_max}] ms")
        
        # Rule 8: Remove NaN values
        before = len(df)
        df = df.dropna()
        removal_counts['nan_values'] = before - len(df)
        logger.info(f"  Rule 8: Removed {before - len(df):,} samples with NaN values")
        
        # Calculate final statistics
        final_count = len(df)
        removed = initial_count - final_count
        removal_pct = (removed / initial_count) * 100 if initial_count > 0 else 0
        
        self.cleaning_stats.update({
            'final_count': final_count,
            'removed_count': removed,
            'removal_percentage': removal_pct,
            'removal_details': removal_counts
        })
        
        logger.info(f"\nData Cleaning Summary:")
        logger.info(f"  Initial samples: {initial_count:,}")
        logger.info(f"  Removed: {removed:,} ({removal_pct:.1f}%)")
        logger.info(f"  Remaining: {final_count:,}")
        
        return df
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Return the cleaning statistics from the last run."""
        return self.cleaning_stats


def filter_invalid_samples(df: pd.DataFrame, 
                          strict_mode: bool = True,
                          thresholds: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to filter invalid samples.
    
    Args:
        df: Input DataFrame
        strict_mode: If True, apply all filters
        thresholds: Custom thresholds dictionary
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(thresholds=thresholds, strict_mode=strict_mode)
    return cleaner.filter_invalid_samples(df)


def quick_quality_check(df: pd.DataFrame) -> Dict[str, float | int]:
    """
    Quick data quality check without modifying the data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    metrics: Dict[str, float | int] = {
        'total_samples': len(df),
    }
    
    # PTT checks
    if 'ptt_peak_to_peak' in df.columns:
        metrics['ptt_zeros'] = int((df['ptt_peak_to_peak'] == 0).sum())
        metrics['ptt_zeros_pct'] = float((df['ptt_peak_to_peak'] == 0).mean() * 100)
        metrics['ptt_mean'] = float(df['ptt_peak_to_peak'].mean())
        metrics['ptt_std'] = float(df['ptt_peak_to_peak'].std())
    
    # Amplitude ratio checks
    if 'amplitude_ratio_ra' in df.columns:
        metrics['amp_ratio_ones'] = int((df['amplitude_ratio_ra'] == 1.0).sum())
        metrics['amp_ratio_ones_pct'] = float((df['amplitude_ratio_ra'] == 1.0).mean() * 100)
    
    # BP checks
    if 'sbp_reference' in df.columns and 'dbp_reference' in df.columns:
        metrics['sbp_dbp_violations'] = int((df['sbp_reference'] <= df['dbp_reference']).sum())
    
    # Correlation checks
    if 'cycle_correlation' in df.columns:
        metrics['low_correlation'] = int((df['cycle_correlation'] < 0.7).sum())
        metrics['low_correlation_pct'] = float((df['cycle_correlation'] < 0.7).mean() * 100)
    
    # Check PTT-BP correlation (should be negative)
    if 'ptt_peak_to_peak' in df.columns and 'sbp_reference' in df.columns:
        valid_mask = (df['ptt_peak_to_peak'] > 0) & (df['ptt_peak_to_peak'] < 500)
        if valid_mask.sum() > 100:
            metrics['ptt_sbp_correlation'] = float(
                df.loc[valid_mask, 'ptt_peak_to_peak'].corr(df.loc[valid_mask, 'sbp_reference'])
            )
    
    return metrics


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Default paths
    default_input = './data/processed/bp_dataset_features.csv'
    default_output = './data/processed/bp_dataset_cleaned_v2.csv'
    
    # Parse arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_input
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = default_output
    
    print(f"Loading: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Input file not found: {csv_path}")
        print("\nMake sure you've run the feature extraction first:")
        print("  python run_full_batch.py")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("BEFORE CLEANING:")
    print("="*60)
    before_stats = quick_quality_check(df)
    for k, v in before_stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("CLEANING...")
    print("="*60)
    df_clean = filter_invalid_samples(df, strict_mode=False)
    
    print("\n" + "="*60)
    print("AFTER CLEANING:")
    print("="*60)
    after_stats = quick_quality_check(df_clean)
    for k, v in after_stats.items():
        print(f"  {k}: {v}")
    
    # Save cleaned dataset
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"  Samples: {len(df_clean):,}")
    print(f"  Patients: {df_clean['patient_id'].nunique()}")
