#!/usr/bin/env python3
"""
Comprehensive Forensic Analysis for Blood Pressure Prediction Pipeline
Diagnoses data quality, feature integrity, and training issues.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Output formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_section(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}--- {title} ---{Colors.END}\n")

def print_ok(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warn(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"  {msg}")

def main():
    # Paths
    data_path = Path('src/data/bp_dataset_features.csv')
    
    if not data_path.exists():
        print_error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    print_header("BLOOD PRESSURE PIPELINE - FORENSIC ANALYSIS")
    
    # =========================================================================
    # PHASE 1: DATA LOADING & STRUCTURE
    # =========================================================================
    print_section("PHASE 1: DATA LOADING & STRUCTURE")
    
    print("Loading dataset (this may take a moment for large files)...")
    df = pd.read_csv(data_path, low_memory=False)
    
    print_ok(f"Dataset loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
    print_info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column analysis
    print_section("Column Inventory")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    
    # Check for critical columns
    critical_cols = ['patient_id', 'sbp_reference', 'dbp_reference']
    dynamic_cols = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'ptt_peak_to_maxslope', 
                    'amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd', 
                    'time_to_maxslope_t1', 'hr_bpm', 'cycle_correlation']
    static_num_cols = ['age', 'bmi']
    static_cat_cols = ['sex', 'position', 'approach', 'aline1', 'preop_ecg', 'dx', 'opname']
    
    print_section("Critical Column Check")
    for col in critical_cols:
        if col in df.columns:
            print_ok(f"{col}: Present")
        else:
            print_error(f"{col}: MISSING!")
    
    # =========================================================================
    # PHASE 2: TARGET VARIABLE ANALYSIS
    # =========================================================================
    print_header("PHASE 2: TARGET VARIABLE ANALYSIS")
    
    sbp = df['sbp_reference']
    dbp = df['dbp_reference']
    
    print_section("SBP Statistics")
    print(f"  Count:    {sbp.count():,}")
    print(f"  Min:      {sbp.min():.2f} mmHg")
    print(f"  Max:      {sbp.max():.2f} mmHg")
    print(f"  Mean:     {sbp.mean():.2f} mmHg")
    print(f"  Median:   {sbp.median():.2f} mmHg")
    print(f"  Std:      {sbp.std():.2f} mmHg")
    print(f"  Missing:  {sbp.isna().sum()}")
    
    print_section("DBP Statistics")
    print(f"  Count:    {dbp.count():,}")
    print(f"  Min:      {dbp.min():.2f} mmHg")
    print(f"  Max:      {dbp.max():.2f} mmHg")
    print(f"  Mean:     {dbp.mean():.2f} mmHg")
    print(f"  Median:   {dbp.median():.2f} mmHg")
    print(f"  Std:      {dbp.std():.2f} mmHg")
    print(f"  Missing:  {dbp.isna().sum()}")
    
    print_section("Target Variable Validation")
    
    # Physiologically impossible values
    sbp_too_low = (sbp < 60).sum()
    sbp_too_high = (sbp > 200).sum()
    dbp_too_low = (dbp < 30).sum()
    dbp_too_high = (dbp > 120).sum()
    sbp_lt_dbp = (sbp < dbp).sum()
    
    if sbp_too_low > 0:
        print_warn(f"SBP < 60 mmHg (hypotension crisis): {sbp_too_low:,} samples ({sbp_too_low/len(df)*100:.2f}%)")
    else:
        print_ok("No SBP < 60 mmHg")
        
    if sbp_too_high > 0:
        print_warn(f"SBP > 200 mmHg (hypertensive crisis): {sbp_too_high:,} samples ({sbp_too_high/len(df)*100:.2f}%)")
    else:
        print_ok("No SBP > 200 mmHg")
        
    if dbp_too_low > 0:
        print_warn(f"DBP < 30 mmHg (extreme hypotension): {dbp_too_low:,} samples ({dbp_too_low/len(df)*100:.2f}%)")
    else:
        print_ok("No DBP < 30 mmHg")
        
    if dbp_too_high > 0:
        print_warn(f"DBP > 120 mmHg (severe hypertension): {dbp_too_high:,} samples ({dbp_too_high/len(df)*100:.2f}%)")
    else:
        print_ok("No DBP > 120 mmHg")
        
    if sbp_lt_dbp > 0:
        print_error(f"SBP < DBP (IMPOSSIBLE): {sbp_lt_dbp:,} samples ({sbp_lt_dbp/len(df)*100:.2f}%)")
    else:
        print_ok("No impossible SBP < DBP")
    
    # Outlier analysis
    print_section("Outlier Analysis (3σ rule)")
    sbp_outliers = ((sbp - sbp.mean()).abs() > 3 * sbp.std()).sum()
    dbp_outliers = ((dbp - dbp.mean()).abs() > 3 * dbp.std()).sum()
    print(f"  SBP outliers (>3σ): {sbp_outliers:,} ({sbp_outliers/len(df)*100:.2f}%)")
    print(f"  DBP outliers (>3σ): {dbp_outliers:,} ({dbp_outliers/len(df)*100:.2f}%)")
    
    # Distribution analysis
    print_section("Target Distribution Percentiles")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {pct}th percentile: SBP={sbp.quantile(pct/100):.1f}, DBP={dbp.quantile(pct/100):.1f}")
    
    # =========================================================================
    # PHASE 3: FEATURE INTEGRITY ANALYSIS
    # =========================================================================
    print_header("PHASE 3: FEATURE INTEGRITY ANALYSIS")
    
    print_section("Dynamic Features")
    
    expected_ranges = {
        'ptt_peak_to_peak': (50, 400),
        'ptt_peak_to_foot': (50, 400),
        'ptt_peak_to_maxslope': (50, 400),
        'amplitude_ratio_ra': (0, 10),
        'systolic_duration_tsd': (100, 600),
        'diastolic_duration_tfd': (100, 1000),
        'time_to_maxslope_t1': (0, 300),
        'hr_bpm': (40, 180),
        'cycle_correlation': (0, 1)
    }
    
    for col in dynamic_cols:
        if col in df.columns:
            data = df[col]
            missing = data.isna().sum()
            missing_pct = missing / len(df) * 100
            
            exp_min, exp_max = expected_ranges.get(col, (None, None))
            out_of_range = 0
            if exp_min is not None:
                out_of_range = ((data < exp_min) | (data > exp_max)).sum()
                out_pct = out_of_range / len(df) * 100
            
            status = "✓" if missing_pct < 1 and out_pct < 5 else "⚠" if missing_pct < 5 else "✗"
            
            print(f"\n{status} {col}:")
            print(f"    Missing: {missing:,} ({missing_pct:.2f}%)")
            print(f"    Range: [{data.min():.2f}, {data.max():.2f}]")
            print(f"    Expected: [{exp_min}, {exp_max}]")
            print(f"    Out of range: {out_of_range:,} ({out_pct:.2f}%)")
            print(f"    Mean±Std: {data.mean():.2f} ± {data.std():.2f}")
            
            # Check for infinite or NaN
            inf_count = np.isinf(data.replace([np.nan], 0)).sum()
            if inf_count > 0:
                print_error(f"    Contains {inf_count} infinite values!")
        else:
            print_error(f"{col}: MISSING from dataset!")
    
    # =========================================================================
    # PHASE 4: CATEGORICAL VARIABLES ANALYSIS
    # =========================================================================
    print_header("PHASE 4: CATEGORICAL VARIABLES ANALYSIS")
    
    for col in static_cat_cols:
        if col in df.columns:
            data = df[col].astype(str)
            n_unique = data.nunique()
            missing = (data.isna() | (data == 'nan') | (data == 'unknown') | (data == '')).sum()
            
            print_section(f"{col}")
            print(f"  Unique values: {n_unique}")
            print(f"  Missing/Unknown: {missing:,} ({missing/len(df)*100:.2f}%)")
            
            # Show value counts (top 10)
            vc = data.value_counts()
            print(f"  Top values:")
            for val, count in vc.head(10).items():
                print(f"    {val}: {count:,} ({count/len(df)*100:.2f}%)")
            
            if n_unique > 20:
                print_warn(f"  High cardinality ({n_unique} classes) - may need aggregation")
        else:
            print_warn(f"{col}: Not present in dataset")
    
    # =========================================================================
    # PHASE 5: PATIENT-LEVEL ANALYSIS
    # =========================================================================
    print_header("PHASE 5: PATIENT-LEVEL ANALYSIS")
    
    if 'patient_id' in df.columns:
        patient_counts = df['patient_id'].value_counts()
        
        print(f"  Total unique patients: {patient_counts.count():,}")
        print(f"  Samples per patient:")
        print(f"    Min: {patient_counts.min()}")
        print(f"    Max: {patient_counts.max()}")
        print(f"    Mean: {patient_counts.mean():.1f}")
        print(f"    Median: {patient_counts.median():.1f}")
        print(f"    Std: {patient_counts.std():.1f}")
        
        # Patients with too few samples
        seq_len = 30
        short_patients = (patient_counts < seq_len).sum()
        if short_patients > 0:
            print_warn(f"  Patients with < {seq_len} samples: {short_patients} (cannot form sequences)")
        
        # Distribution of samples per patient
        print(f"\n  Samples per patient distribution:")
        for pct in [10, 25, 50, 75, 90, 99]:
            print(f"    {pct}th percentile: {patient_counts.quantile(pct/100):.0f}")
        
        # Imbalanced patients
        top10_patients = patient_counts.head(10).sum()
        top10_pct = top10_patients / len(df) * 100
        print(f"\n  Top 10 patients contribute: {top10_pct:.2f}% of samples")
        if top10_pct > 20:
            print_warn("  Heavy patient imbalance may cause model to memorize patient-specific patterns")
    else:
        print_error("patient_id column not found!")
    
    # =========================================================================
    # PHASE 6: CORRELATION ANALYSIS
    # =========================================================================
    print_header("PHASE 6: FEATURE-TARGET CORRELATION ANALYSIS")
    
    numeric_cols = dynamic_cols + static_num_cols + ['sbp_reference', 'dbp_reference']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    df_numeric = df[numeric_cols].dropna()
    
    if len(df_numeric) > 0:
        corr_matrix = df_numeric.corr()
        
        print_section("Correlations with SBP")
        sbp_corr = corr_matrix['sbp_reference'].drop(['sbp_reference', 'dbp_reference']).sort_values(key=abs, ascending=False)
        for feat, corr in sbp_corr.items():
            strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
            print(f"  {feat}: {corr:.4f} ({strength})")
        
        print_section("Correlations with DBP")
        dbp_corr = corr_matrix['dbp_reference'].drop(['sbp_reference', 'dbp_reference']).sort_values(key=abs, ascending=False)
        for feat, corr in dbp_corr.items():
            strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
            print(f"  {feat}: {corr:.4f} ({strength})")
        
        # Check if ANY feature has strong correlation
        max_corr_sbp = sbp_corr.abs().max()
        max_corr_dbp = dbp_corr.abs().max()
        
        if max_corr_sbp < 0.3:
            print_error(f"\n  CRITICAL: No feature has |correlation| > 0.3 with SBP! (max={max_corr_sbp:.4f})")
            print_error("  This suggests features may not be predictive of BP!")
        
        if max_corr_dbp < 0.3:
            print_error(f"\n  CRITICAL: No feature has |correlation| > 0.3 with DBP! (max={max_corr_dbp:.4f})")
        
        # Inter-feature correlation (check for multicollinearity)
        print_section("Highly Correlated Feature Pairs (|r| > 0.9)")
        feature_cols = [c for c in numeric_cols if c not in ['sbp_reference', 'dbp_reference']]
        high_corr_pairs = []
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((col1, col2, corr_val))
                    print_warn(f"  {col1} vs {col2}: {corr_val:.4f}")
        
        if not high_corr_pairs:
            print_ok("  No highly correlated feature pairs found")
    
    # =========================================================================
    # PHASE 7: DATA LEAKAGE CHECK
    # =========================================================================
    print_header("PHASE 7: DATA LEAKAGE ANALYSIS")
    
    print_section("Potential Leakage Indicators")
    
    # Check if cycle_correlation is suspiciously high
    if 'cycle_correlation' in df.columns:
        corr_with_bp = df['cycle_correlation'].corr(df['sbp_reference'])
        if abs(corr_with_bp) > 0.5:
            print_warn(f"  cycle_correlation highly correlated with SBP ({corr_with_bp:.4f})")
    
    # Check session_id patterns
    if 'session_id' in df.columns:
        print(f"  Unique sessions: {df['session_id'].nunique()}")
    
    # Timestamp monotonicity check
    if 'timestamp' in df.columns:
        print("\n  Timestamp analysis:")
        print(f"    Min: {df['timestamp'].min()}")
        print(f"    Max: {df['timestamp'].max()}")
        
        # Check for temporal autocorrelation in targets
        if 'patient_id' in df.columns:
            # Sample a few patients to check autocorrelation
            sample_pids = df['patient_id'].unique()[:5]
            print(f"\n  Target autocorrelation (lag=1) for sample patients:")
            for pid in sample_pids:
                patient_data = df[df['patient_id'] == pid].sort_values('timestamp')
                if len(patient_data) > 10:
                    sbp_autocorr = patient_data['sbp_reference'].autocorr(lag=1)
                    print(f"    Patient {pid}: {sbp_autocorr:.4f}")
    
    # =========================================================================
    # PHASE 8: VARIANCE ANALYSIS
    # =========================================================================
    print_header("PHASE 8: VARIANCE ANALYSIS")
    
    print_section("Within-Patient vs Between-Patient Variance")
    
    if 'patient_id' in df.columns:
        # Calculate within-patient and between-patient variance for SBP
        patient_means = df.groupby('patient_id')['sbp_reference'].mean()
        patient_stds = df.groupby('patient_id')['sbp_reference'].std()
        
        between_patient_var = patient_means.var()
        within_patient_var = (patient_stds ** 2).mean()
        
        print(f"  SBP Between-patient variance: {between_patient_var:.2f}")
        print(f"  SBP Within-patient variance (mean): {within_patient_var:.2f}")
        print(f"  Ratio (between/within): {between_patient_var/within_patient_var:.2f}")
        
        if between_patient_var > 10 * within_patient_var:
            print_warn("  Between-patient variance dominates! Model may learn patient identity instead of BP.")
            print_warn("  Consider: Within-patient normalization or patient-adaptive calibration.")
        
        # Same for DBP
        patient_means_dbp = df.groupby('patient_id')['dbp_reference'].mean()
        patient_stds_dbp = df.groupby('patient_id')['dbp_reference'].std()
        
        between_patient_var_dbp = patient_means_dbp.var()
        within_patient_var_dbp = (patient_stds_dbp ** 2).mean()
        
        print(f"\n  DBP Between-patient variance: {between_patient_var_dbp:.2f}")
        print(f"  DBP Within-patient variance (mean): {within_patient_var_dbp:.2f}")
        print(f"  Ratio (between/within): {between_patient_var_dbp/within_patient_var_dbp:.2f}")
    
    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    print_header("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    
    issues = []
    recommendations = []
    
    # Check correlation issues
    if max_corr_sbp < 0.2:
        issues.append("CRITICAL: Features have near-zero correlation with targets")
        recommendations.append("1. Re-examine feature extraction - PTT and morphology features should correlate with BP")
        recommendations.append("2. Consider raw waveform input instead of handcrafted features")
        recommendations.append("3. Verify that reference BP values are correctly synchronized with features")
    
    # Check patient imbalance
    if 'patient_id' in df.columns and top10_pct > 30:
        issues.append("Patient imbalance detected")
        recommendations.append("4. Implement weighted sampling or under-sampling for dominant patients")
    
    # Check variance ratio
    if 'patient_id' in df.columns and between_patient_var > 5 * within_patient_var:
        issues.append("High between-patient variance vs within-patient variance")
        recommendations.append("5. Use per-patient normalization of targets (calibration-based approach)")
        recommendations.append("6. Consider predicting BP CHANGE instead of absolute BP")
    
    # Check missing values
    total_missing = df[dynamic_cols].isna().sum().sum() if all(c in df.columns for c in dynamic_cols) else 0
    if total_missing > len(df) * len(dynamic_cols) * 0.01:
        issues.append(f"High missing value rate in features: {total_missing}")
        recommendations.append("7. Investigate and fix missing value causes in feature extraction")
    
    print(f"{Colors.RED}ISSUES FOUND: {len(issues)}{Colors.END}\n")
    for issue in issues:
        print_error(issue)
    
    print(f"\n{Colors.GREEN}RECOMMENDATIONS:{Colors.END}\n")
    for rec in recommendations:
        print(f"  {rec}")
    
    if not recommendations:
        print("  Data quality appears acceptable. Issue may be in model architecture or training.")
        print("  - Check learning rate (try 1e-4 to 1e-3)")
        print("  - Check sequence length (try 60-100 timesteps)")
        print("  - Verify data normalization is applied correctly")
        print("  - Check for gradient issues (vanishing/exploding)")
    
    print_header("ANALYSIS COMPLETE")

if __name__ == '__main__':
    main()
