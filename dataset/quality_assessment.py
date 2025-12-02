#!/usr/bin/env python3
"""
Quality Assessment Module for Blood Pressure Prediction Pipeline

This module provides comprehensive quality assessment and reporting for the 
BP dataset, generating metrics, reports, and visualizations to track data 
quality before and after cleaning.

Key Features:
1. Generate detailed quality reports with physiological validation
2. Track correlations (especially PTT-BP inverse relationship)
3. Identify data quality issues at patient/session level
4. Compare before/after cleaning statistics

Author: Blood Pressure Pipeline Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality report for the BP dataset."""
    
    # Basic statistics
    total_samples: int = 0
    total_patients: int = 0
    total_sessions: int = 0
    samples_per_patient_mean: float = 0.0
    samples_per_patient_std: float = 0.0
    
    # PTT quality metrics
    ptt_zeros: int = 0
    ptt_zeros_pct: float = 0.0
    ptt_outliers_low: int = 0  # < 30 ms
    ptt_outliers_high: int = 0  # > 300 ms
    ptt_mean: float = 0.0
    ptt_std: float = 0.0
    ptt_median: float = 0.0
    
    # PAT quality metrics (if available)
    pat_zeros: int = 0
    pat_zeros_pct: float = 0.0
    pat_mean: float = 0.0
    pat_std: float = 0.0
    
    # Amplitude ratio metrics
    amplitude_ones: int = 0  # Exactly 1.0 (default value)
    amplitude_ones_pct: float = 0.0
    amplitude_zeros: int = 0  # < 0.01
    amplitude_zeros_pct: float = 0.0
    amplitude_mean: float = 0.0
    amplitude_std: float = 0.0
    
    # BP metrics
    sbp_mean: float = 0.0
    sbp_std: float = 0.0
    sbp_min: float = 0.0
    sbp_max: float = 0.0
    dbp_mean: float = 0.0
    dbp_std: float = 0.0
    sbp_dbp_violations: int = 0  # Cases where SBP <= DBP
    bp_outliers: int = 0  # Outside clinical range
    
    # HR metrics
    hr_mean: float = 0.0
    hr_std: float = 0.0
    hr_outliers: int = 0  # Outside 30-200 BPM
    
    # Waveform quality
    low_correlation_count: int = 0
    low_correlation_pct: float = 0.0
    correlation_mean: float = 0.0
    
    # Key correlations (physiological validity)
    ptt_sbp_correlation: float = 0.0  # Should be negative (-0.5 to -0.8)
    ptt_dbp_correlation: float = 0.0  # Should be negative
    pat_sbp_correlation: float = 0.0  # Should be negative
    hr_sbp_correlation: float = 0.0   # Should be slightly positive (0.2-0.4)
    
    # Quality flags
    has_valid_ptt_bp_relationship: bool = False  # True if correlation < -0.3
    has_sufficient_samples: bool = False  # True if > 100,000 samples
    has_good_signal_quality: bool = False  # True if <30% low correlation
    
    # Metadata
    report_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_path: str = ""
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary with JSON-serializable types."""
        d = asdict(self)
        # Convert numpy types to Python native types for JSON serialization
        for key, value in d.items():
            if hasattr(value, 'item'):  # numpy scalar
                d[key] = value.item()
            elif isinstance(value, (bool, np.bool_)):
                d[key] = bool(value)
        return d
    
    def to_json(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else float(x) if hasattr(x, 'item') else x)
    
    def print_summary(self):
        """Print a formatted summary of the report."""
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        print(f"Generated: {self.report_timestamp}")
        print(f"Dataset: {self.dataset_path}")
        print()
        
        # Basic stats
        print("BASIC STATISTICS:")
        print(f"  Total samples: {self.total_samples:,}")
        print(f"  Total patients: {self.total_patients}")
        print(f"  Total sessions: {self.total_sessions}")
        print(f"  Samples/patient: {self.samples_per_patient_mean:.1f} ± {self.samples_per_patient_std:.1f}")
        print()
        
        # PTT quality
        print("PTT QUALITY:")
        print(f"  PTT = 0: {self.ptt_zeros:,} ({self.ptt_zeros_pct:.1f}%)")
        print(f"  PTT < 30ms: {self.ptt_outliers_low:,}")
        print(f"  PTT > 300ms: {self.ptt_outliers_high:,}")
        print(f"  PTT mean ± std: {self.ptt_mean:.1f} ± {self.ptt_std:.1f} ms")
        print()
        
        # Amplitude quality
        print("AMPLITUDE RATIO QUALITY:")
        print(f"  Ratio = 1.0 (default): {self.amplitude_ones:,} ({self.amplitude_ones_pct:.1f}%)")
        print(f"  Ratio ≈ 0: {self.amplitude_zeros:,} ({self.amplitude_zeros_pct:.1f}%)")
        print(f"  Ratio mean ± std: {self.amplitude_mean:.3f} ± {self.amplitude_std:.3f}")
        print()
        
        # BP quality
        print("BLOOD PRESSURE QUALITY:")
        print(f"  SBP range: {self.sbp_min:.0f} - {self.sbp_max:.0f} mmHg")
        print(f"  SBP mean ± std: {self.sbp_mean:.1f} ± {self.sbp_std:.1f} mmHg")
        print(f"  DBP mean ± std: {self.dbp_mean:.1f} ± {self.dbp_std:.1f} mmHg")
        print(f"  SBP <= DBP violations: {self.sbp_dbp_violations:,}")
        print(f"  BP outliers: {self.bp_outliers:,}")
        print()
        
        # Correlations (KEY METRIC)
        print("PHYSIOLOGICAL CORRELATIONS (KEY METRICS):")
        print(f"  PTT vs SBP: {self.ptt_sbp_correlation:+.3f}  (Expected: < -0.5)")
        print(f"  PTT vs DBP: {self.ptt_dbp_correlation:+.3f}  (Expected: < -0.5)")
        if self.pat_sbp_correlation != 0:
            print(f"  PAT vs SBP: {self.pat_sbp_correlation:+.3f}  (Expected: < -0.5)")
        print(f"  HR vs SBP:  {self.hr_sbp_correlation:+.3f}   (Expected: 0.2 to 0.4)")
        print()
        
        # Signal quality
        print("WAVEFORM QUALITY:")
        print(f"  Low correlation cycles: {self.low_correlation_count:,} ({self.low_correlation_pct:.1f}%)")
        print(f"  Mean cycle correlation: {self.correlation_mean:.3f}")
        print()
        
        # Quality flags
        print("QUALITY FLAGS:")
        print(f"  ✓ Valid PTT-BP relationship: {'YES ✓' if self.has_valid_ptt_bp_relationship else 'NO ✗'}")
        print(f"  ✓ Sufficient samples (>100k): {'YES ✓' if self.has_sufficient_samples else 'NO ✗'}")
        print(f"  ✓ Good signal quality (<30% low corr): {'YES ✓' if self.has_good_signal_quality else 'NO ✗'}")
        print("="*70)


def generate_quality_report(df: pd.DataFrame, 
                           dataset_path: str = "",
                           correlation_threshold: float = 0.7) -> QualityReport:
    """
    Generate a comprehensive quality report for the dataset.
    
    Args:
        df: Input DataFrame with features
        dataset_path: Path to the dataset (for logging)
        correlation_threshold: Threshold for "low correlation" cycles
        
    Returns:
        QualityReport object with all metrics
    """
    report = QualityReport(dataset_path=dataset_path)
    
    if df.empty:
        logger.warning("Empty DataFrame provided to quality assessment")
        return report
    
    # Basic statistics
    report.total_samples = len(df)
    
    if 'patient_id' in df.columns:
        report.total_patients = df['patient_id'].nunique()
        samples_per_patient = df.groupby('patient_id').size()
        report.samples_per_patient_mean = samples_per_patient.mean()
        report.samples_per_patient_std = samples_per_patient.std()
    
    if 'session_id' in df.columns:
        report.total_sessions = df['session_id'].nunique()
    
    # PTT quality metrics
    if 'ptt_peak_to_peak' in df.columns:
        ptt = df['ptt_peak_to_peak']
        report.ptt_zeros = int((ptt == 0).sum())
        report.ptt_zeros_pct = (ptt == 0).mean() * 100
        report.ptt_outliers_low = int((ptt < 30).sum())
        report.ptt_outliers_high = int((ptt > 300).sum())
        
        # Stats on non-zero values
        ptt_valid = ptt[ptt > 0]
        if len(ptt_valid) > 0:
            report.ptt_mean = float(ptt_valid.mean())
            report.ptt_std = float(ptt_valid.std())
            report.ptt_median = float(ptt_valid.median())
    
    # PAT quality metrics
    if 'pat_ecg_ppg' in df.columns:
        pat = df['pat_ecg_ppg']
        report.pat_zeros = int((pat == 0).sum())
        report.pat_zeros_pct = (pat == 0).mean() * 100
        pat_valid = pat[pat > 0]
        if len(pat_valid) > 0:
            report.pat_mean = float(pat_valid.mean())
            report.pat_std = float(pat_valid.std())
    
    # Amplitude ratio metrics
    if 'amplitude_ratio_ra' in df.columns:
        amp = df['amplitude_ratio_ra']
        report.amplitude_ones = int((amp == 1.0).sum())
        report.amplitude_ones_pct = (amp == 1.0).mean() * 100
        report.amplitude_zeros = int((amp < 0.01).sum())
        report.amplitude_zeros_pct = (amp < 0.01).mean() * 100
        
        # Stats on valid values
        amp_valid = amp[(amp > 0.01) & (amp < 0.99)]
        if len(amp_valid) > 0:
            report.amplitude_mean = float(amp_valid.mean())
            report.amplitude_std = float(amp_valid.std())
    
    # BP metrics
    if 'sbp_reference' in df.columns:
        sbp = df['sbp_reference']
        report.sbp_mean = float(sbp.mean())
        report.sbp_std = float(sbp.std())
        report.sbp_min = float(sbp.min())
        report.sbp_max = float(sbp.max())
        
        # Count outliers
        report.bp_outliers = int(((sbp < 60) | (sbp > 200)).sum())
    
    if 'dbp_reference' in df.columns:
        dbp = df['dbp_reference']
        report.dbp_mean = float(dbp.mean())
        report.dbp_std = float(dbp.std())
        
        report.bp_outliers += int(((dbp < 30) | (dbp > 120)).sum())
    
    if 'sbp_reference' in df.columns and 'dbp_reference' in df.columns:
        report.sbp_dbp_violations = int((df['sbp_reference'] <= df['dbp_reference']).sum())
    
    # HR metrics
    if 'hr_bpm' in df.columns:
        hr = df['hr_bpm']
        report.hr_mean = float(hr.mean())
        report.hr_std = float(hr.std())
        report.hr_outliers = int(((hr < 30) | (hr > 200)).sum())
    
    # Waveform quality
    if 'cycle_correlation' in df.columns:
        corr = df['cycle_correlation']
        report.low_correlation_count = int((corr < correlation_threshold).sum())
        report.low_correlation_pct = (corr < correlation_threshold).mean() * 100
        report.correlation_mean = float(corr.mean())
    
    # Key correlations (THIS IS THE CRITICAL METRIC)
    # We should only compute correlations on valid data (non-zero PTT)
    if 'ptt_peak_to_peak' in df.columns and 'sbp_reference' in df.columns:
        valid_mask = (df['ptt_peak_to_peak'] > 0) & (df['ptt_peak_to_peak'] < 500)
        if valid_mask.sum() > 100:
            report.ptt_sbp_correlation = float(
                df.loc[valid_mask, 'ptt_peak_to_peak'].corr(df.loc[valid_mask, 'sbp_reference'])
            )
    
    if 'ptt_peak_to_peak' in df.columns and 'dbp_reference' in df.columns:
        valid_mask = (df['ptt_peak_to_peak'] > 0) & (df['ptt_peak_to_peak'] < 500)
        if valid_mask.sum() > 100:
            report.ptt_dbp_correlation = float(
                df.loc[valid_mask, 'ptt_peak_to_peak'].corr(df.loc[valid_mask, 'dbp_reference'])
            )
    
    if 'pat_ecg_ppg' in df.columns and 'sbp_reference' in df.columns:
        valid_mask = (df['pat_ecg_ppg'] > 0) & (df['pat_ecg_ppg'] < 600)
        if valid_mask.sum() > 100:
            report.pat_sbp_correlation = float(
                df.loc[valid_mask, 'pat_ecg_ppg'].corr(df.loc[valid_mask, 'sbp_reference'])
            )
    
    if 'hr_bpm' in df.columns and 'sbp_reference' in df.columns:
        valid_mask = (df['hr_bpm'] > 30) & (df['hr_bpm'] < 200)
        if valid_mask.sum() > 100:
            report.hr_sbp_correlation = float(
                df.loc[valid_mask, 'hr_bpm'].corr(df.loc[valid_mask, 'sbp_reference'])
            )
    
    # Set quality flags
    report.has_valid_ptt_bp_relationship = report.ptt_sbp_correlation < -0.3
    report.has_sufficient_samples = report.total_samples > 100000
    report.has_good_signal_quality = report.low_correlation_pct < 30
    
    return report


def compare_reports(before: QualityReport, after: QualityReport) -> Dict:
    """
    Compare two quality reports (before/after cleaning).
    
    Args:
        before: Quality report before cleaning
        after: Quality report after cleaning
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'samples_removed': before.total_samples - after.total_samples,
        'samples_removed_pct': (before.total_samples - after.total_samples) / before.total_samples * 100 if before.total_samples > 0 else 0,
        
        'ptt_zeros_removed': before.ptt_zeros - after.ptt_zeros,
        'amplitude_ones_removed': before.amplitude_ones - after.amplitude_ones,
        'sbp_dbp_violations_removed': before.sbp_dbp_violations - after.sbp_dbp_violations,
        'low_corr_removed': before.low_correlation_count - after.low_correlation_count,
        
        'ptt_sbp_corr_change': after.ptt_sbp_correlation - before.ptt_sbp_correlation,
        'ptt_dbp_corr_change': after.ptt_dbp_correlation - before.ptt_dbp_correlation,
        
        'quality_improvement': {
            'valid_ptt_bp_relationship': (not before.has_valid_ptt_bp_relationship) and after.has_valid_ptt_bp_relationship,
            'good_signal_quality': (not before.has_good_signal_quality) and after.has_good_signal_quality,
        }
    }
    
    return comparison


def print_comparison(before: QualityReport, after: QualityReport):
    """Print a formatted comparison of two reports."""
    comparison = compare_reports(before, after)
    
    print("\n" + "="*70)
    print("BEFORE vs AFTER CLEANING COMPARISON")
    print("="*70)
    
    print(f"\nSAMPLES:")
    print(f"  Before: {before.total_samples:,}")
    print(f"  After:  {after.total_samples:,}")
    print(f"  Removed: {comparison['samples_removed']:,} ({comparison['samples_removed_pct']:.1f}%)")
    
    print(f"\nDATA QUALITY ISSUES RESOLVED:")
    print(f"  PTT = 0 samples removed: {comparison['ptt_zeros_removed']:,}")
    print(f"  Amplitude = 1.0 removed: {comparison['amplitude_ones_removed']:,}")
    print(f"  SBP <= DBP violations removed: {comparison['sbp_dbp_violations_removed']:,}")
    print(f"  Low correlation cycles removed: {comparison['low_corr_removed']:,}")
    
    print(f"\nCORRELATION CHANGES:")
    print(f"  PTT-SBP correlation: {before.ptt_sbp_correlation:+.3f} → {after.ptt_sbp_correlation:+.3f} (Δ = {comparison['ptt_sbp_corr_change']:+.3f})")
    print(f"  PTT-DBP correlation: {before.ptt_dbp_correlation:+.3f} → {after.ptt_dbp_correlation:+.3f} (Δ = {comparison['ptt_dbp_corr_change']:+.3f})")
    
    print(f"\nQUALITY FLAGS:")
    print(f"  Valid PTT-BP relationship: {before.has_valid_ptt_bp_relationship} → {after.has_valid_ptt_bp_relationship}")
    print(f"  Good signal quality: {before.has_good_signal_quality} → {after.has_good_signal_quality}")
    print("="*70)


def get_patient_quality_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get quality metrics broken down by patient.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with per-patient quality metrics
    """
    if 'patient_id' not in df.columns:
        logger.warning("No patient_id column found")
        return pd.DataFrame()
    
    metrics = []
    
    for patient_id, group in df.groupby('patient_id'):
        patient_metrics = {
            'patient_id': patient_id,
            'n_samples': len(group),
        }
        
        if 'ptt_peak_to_peak' in group.columns:
            ptt = group['ptt_peak_to_peak']
            patient_metrics['ptt_zeros_pct'] = (ptt == 0).mean() * 100
            patient_metrics['ptt_mean'] = ptt[ptt > 0].mean() if (ptt > 0).any() else np.nan
        
        if 'amplitude_ratio_ra' in group.columns:
            amp = group['amplitude_ratio_ra']
            patient_metrics['amp_ones_pct'] = (amp == 1.0).mean() * 100
        
        if 'cycle_correlation' in group.columns:
            patient_metrics['mean_correlation'] = group['cycle_correlation'].mean()
        
        if 'sbp_reference' in group.columns:
            patient_metrics['sbp_mean'] = group['sbp_reference'].mean()
            patient_metrics['sbp_std'] = group['sbp_reference'].std()
        
        # PTT-SBP correlation for this patient
        if 'ptt_peak_to_peak' in group.columns and 'sbp_reference' in group.columns:
            valid = (group['ptt_peak_to_peak'] > 0) & (group['ptt_peak_to_peak'] < 500)
            if valid.sum() > 10:
                patient_metrics['ptt_sbp_corr'] = group.loc[valid, 'ptt_peak_to_peak'].corr(
                    group.loc[valid, 'sbp_reference']
                )
            else:
                patient_metrics['ptt_sbp_corr'] = np.nan
        
        metrics.append(patient_metrics)
    
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        
        report = generate_quality_report(df, dataset_path=csv_path)
        report.print_summary()
        
        # Save report
        report_path = csv_path.replace('.csv', '_quality_report.json')
        report.to_json(report_path)
        print(f"\nReport saved to: {report_path}")
        
        # Per-patient breakdown
        print("\nPer-patient quality breakdown (top 10 by sample count):")
        patient_df = get_patient_quality_breakdown(df)
        if not patient_df.empty:
            print(patient_df.nlargest(10, 'n_samples').to_string(index=False))
    else:
        print("Usage: python quality_assessment.py <path_to_csv>")
        print("\nExample:")
        print("  python quality_assessment.py data/processed/bp_dataset_features.csv")
