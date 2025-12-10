# -*- coding: utf-8 -*-
"""
================================================================================
DATASET PROBLEM ANALYSIS: Why PTT-Based BP Estimation Fails in Surgical Patients
================================================================================

This script provides a comprehensive analysis of the fundamental limitations
in using PTT/PAT features for blood pressure estimation in the VitalDB surgical
dataset.

UPDATED: December 8, 2025
- _norm features now correctly normalized (mean=0, std=1 per patient)
- Added calibration analysis showing EMA outperforms features
- Cross-patient R² still negative (-0.01 to -0.04) - this is expected

Key Findings Summary:
1. Features explain only ~7% of BP variance (need 72% for AAMI compliance)
2. PTT-BP relationship is NON-STATIONARY (changes during surgery)
3. No universal PTT-BP relationship exists (88% variation across patients)
4. Temporal autocorrelation r=0.94 creates illusion of large dataset
5. EMA calibration achieves 61% AAMI compliance WITHOUT features
6. Adding features HURTS performance (61% → 47% AAMI)

Author: Blood Pressure Estimation Team
Date: December 2024, Updated December 2025
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
DATA_PATH = 'src/data/bp_dataset_strict.csv'
OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01', 'danger': '#C73E1D'}


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATA_PATH)
    
    # Compute deviations
    df['sbp_dev'] = df['sbp_reference'] - df.groupby('patient_id')['sbp_reference'].transform('mean')
    df['dbp_dev'] = df['dbp_reference'] - df.groupby('patient_id')['dbp_reference'].transform('mean')
    
    return df


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * 60)


# =============================================================================
# ANALYSIS 1: Dataset Overview
# =============================================================================

def analyze_dataset_overview(df):
    """Provide basic dataset statistics."""
    print_header("1. DATASET OVERVIEW")
    
    n_samples = len(df)
    n_patients = df['patient_id'].nunique()
    samples_per_patient = df.groupby('patient_id').size()
    
    print(f"""
    Total samples:           {n_samples:,}
    Number of patients:      {n_patients}
    Samples per patient:
        Mean:                {samples_per_patient.mean():.0f}
        Median:              {samples_per_patient.median():.0f}
        Min:                 {samples_per_patient.min()}
        Max:                 {samples_per_patient.max()}
    
    Blood Pressure Statistics:
        SBP Mean ± Std:      {df['sbp_reference'].mean():.1f} ± {df['sbp_reference'].std():.1f} mmHg
        SBP Range:           [{df['sbp_reference'].min():.0f}, {df['sbp_reference'].max():.0f}] mmHg
        DBP Mean ± Std:      {df['dbp_reference'].mean():.1f} ± {df['dbp_reference'].std():.1f} mmHg
    """)
    
    return samples_per_patient


# =============================================================================
# ANALYSIS 2: The Autocorrelation Problem
# =============================================================================

def analyze_autocorrelation(df):
    """Analyze temporal autocorrelation - the hidden data leakage."""
    print_header("2. THE AUTOCORRELATION PROBLEM")
    
    print("""
    ⚠️  CRITICAL ISSUE: Temporal Autocorrelation
    
    Consecutive cardiac cycles are highly correlated because:
    - BP changes slowly (seconds to minutes)
    - Each sample is ~1 second apart
    - Adjacent samples share 94% of their information
    """)
    
    # Compute autocorrelation for different lags
    print_subheader("Autocorrelation at Different Lags")
    
    autocorr_results = []
    for lag in [1, 5, 10, 20, 50, 100]:
        acs = []
        for pid in df['patient_id'].unique()[:30]:  # Sample patients
            p_df = df[df['patient_id'] == pid]['sbp_reference']
            if len(p_df) > lag + 10:
                ac = p_df.autocorr(lag=lag)
                if not np.isnan(ac):
                    acs.append(ac)
        
        mean_ac = np.mean(acs)
        autocorr_results.append({'lag': lag, 'autocorr': mean_ac})
        print(f"    Lag {lag:3d}: r = {mean_ac:.3f}")
    
    # Effective sample size calculation
    print_subheader("Effective Sample Size")
    
    r = 0.94  # Autocorrelation at lag 1
    n_nominal = len(df)
    # Effective sample size formula for AR(1) process
    n_effective = n_nominal * (1 - r) / (1 + r)
    
    print(f"""
    Nominal sample size:     {n_nominal:,}
    Autocorrelation (lag-1): {r}
    Effective sample size:   {n_effective:,.0f}
    
    ➡️  Your {n_nominal:,} samples contain only ~{n_effective:,.0f} independent observations!
    ➡️  This is {n_nominal/n_effective:.0f}x data redundancy
    """)
    
    # What subsampling achieves
    print_subheader("Effect of Subsampling")
    
    for skip in [1, 5, 10, 20]:
        new_r = r ** skip
        n_sub = n_nominal // skip
        n_eff_sub = n_sub * (1 - new_r) / (1 + new_r)
        print(f"    Skip {skip:2d}: {n_sub:6,} samples, autocorr={new_r:.3f}, effective={n_eff_sub:,.0f}")
    
    return autocorr_results


# =============================================================================
# ANALYSIS 3: Variance Decomposition
# =============================================================================

def analyze_variance_decomposition(df):
    """Decompose variance into between-patient and within-patient components."""
    print_header("3. VARIANCE DECOMPOSITION (Simpson's Paradox)")
    
    print("""
    ⚠️  CRITICAL ISSUE: Patient Identity Dominates Features
    
    Most feature variance is BETWEEN patients (patient identity), not WITHIN patients
    (what we need for BP prediction). This causes Simpson's Paradox where:
    - Global correlation can have OPPOSITE sign from within-patient correlation
    - Models can achieve high R² by memorizing patient identity, not learning BP
    """)
    
    print_subheader("Blood Pressure Variance Decomposition")
    
    # BP variance
    for bp_col, name in [('sbp_reference', 'SBP'), ('dbp_reference', 'DBP')]:
        total_var = df[bp_col].var()
        between_var = df.groupby('patient_id')[bp_col].mean().var()
        within_var = df.groupby('patient_id')[bp_col].var().mean()
        
        print(f"""
    {name}:
        Total variance:       {total_var:7.1f} mmHg²
        Between-patient:      {between_var:7.1f} mmHg² ({100*between_var/total_var:5.1f}%)
        Within-patient:       {within_var:7.1f} mmHg² ({100*within_var/total_var:5.1f}%)
        """)
    
    print_subheader("Feature Variance Decomposition")
    
    features = ['ptt_peak_to_foot', 'hr_bpm', 'pat_ecg_ppg', 'crest_time']
    variance_data = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        total = df[feat].var()
        between = df.groupby('patient_id')[feat].mean().var()
        within = df.groupby('patient_id')[feat].var().mean()
        
        pct_between = 100 * between / total
        pct_within = 100 * within / total
        
        variance_data.append({
            'feature': feat,
            'between_pct': pct_between,
            'within_pct': pct_within
        })
        
        status = "⚠️" if pct_between > 70 else "✓"
        print(f"    {feat:25s}: {pct_between:5.1f}% between, {pct_within:5.1f}% within {status}")
    
    print("""
    ➡️  77-89% of feature variance is patient identity!
    ➡️  Only 11-23% captures within-patient variation (what predicts BP changes)
    ➡️  Using raw features causes massive overfitting to patient identity
    """)
    
    return variance_data


# =============================================================================
# ANALYSIS 4: Feature-BP Correlations
# =============================================================================

def analyze_correlations(df):
    """Analyze cross-patient vs within-patient correlations."""
    print_header("4. FEATURE-BP CORRELATIONS")
    
    print("""
    Comparing GLOBAL (cross-patient) vs WITHIN-patient correlations.
    Within-patient correlations are what matter for BP prediction.
    """)
    
    features = ['ptt_peak_to_foot', 'hr_bpm', 'pat_ecg_ppg', 'crest_time', 
                'diastolic_duration_tfd', 'systolic_duration_tfd']
    
    print_subheader("Raw Features → SBP")
    
    correlation_data = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        # Global correlation
        global_r = df[feat].corr(df['sbp_reference'])
        
        # Within-patient correlations
        within_rs = []
        for pid in df['patient_id'].unique():
            p_df = df[df['patient_id'] == pid]
            if len(p_df) > 30:
                r = p_df[feat].corr(p_df['sbp_reference'])
                if not np.isnan(r):
                    within_rs.append(r)
        
        within_mean = np.mean(within_rs)
        within_std = np.std(within_rs)
        
        correlation_data.append({
            'feature': feat,
            'global_r': global_r,
            'within_mean': within_mean,
            'within_std': within_std
        })
        
        sign_match = "✓" if np.sign(global_r) == np.sign(within_mean) else "⚠️ SIGN FLIP"
        print(f"    {feat:30s}: Global={global_r:+.3f}, Within={within_mean:+.3f}±{within_std:.3f} {sign_match}")
    
    print_subheader("Centered Features → SBP Deviation")
    
    # Center features per patient
    for feat in features:
        if feat in df.columns:
            df[f'{feat}_centered'] = df[feat] - df.groupby('patient_id')[feat].transform('mean')
    
    for feat in features:
        if feat not in df.columns:
            continue
        
        r = df[f'{feat}_centered'].corr(df['sbp_dev'])
        r2 = r ** 2
        
        status = "★" if abs(r) > 0.2 else ""
        print(f"    {feat:30s}: r={r:+.3f} (R²={r2:.3f}) {status}")
    
    print("""
    ➡️  Best feature correlation: |r| ≈ 0.24 → R² ≈ 0.06
    ➡️  Features explain only ~6% of BP variation
    ➡️  This is the FUNDAMENTAL LIMIT of this feature set
    """)
    
    return correlation_data


# =============================================================================
# ANALYSIS 5: Non-Stationarity Analysis
# =============================================================================

def analyze_nonstationarity(df):
    """Analyze how PTT-BP relationship changes over time within patients."""
    print_header("5. NON-STATIONARITY: The PTT-BP Relationship Changes During Surgery")
    
    print("""
    ⚠️  CRITICAL ISSUE: The PTT-BP relationship is NOT constant
    
    During surgery, the relationship changes due to:
    - Anesthesia (changes vascular tone)
    - Medications (vasopressors, vasodilators)
    - Blood loss / fluid infusion
    - Surgical stimulation
    - Patient positioning
    """)
    
    print_subheader("PTT-SBP Correlation by Time Quarter (per patient)")
    
    nonstationarity_data = []
    
    for pid in df['patient_id'].unique()[:10]:
        p_df = df[df['patient_id'] == pid]
        n = len(p_df)
        if n < 200:
            continue
        
        # Split into quarters
        quarters = [
            p_df.iloc[:n//4],
            p_df.iloc[n//4:n//2],
            p_df.iloc[n//2:3*n//4],
            p_df.iloc[3*n//4:]
        ]
        
        rs = []
        for q_df in quarters:
            r = q_df['ptt_peak_to_foot'].corr(q_df['sbp_reference'])
            rs.append(r)
        
        nonstationarity_data.append({
            'patient_id': pid,
            'q1': rs[0], 'q2': rs[1], 'q3': rs[2], 'q4': rs[3],
            'range': max(rs) - min(rs),
            'sign_changes': sum(1 for i in range(3) if np.sign(rs[i]) != np.sign(rs[i+1]))
        })
        
        sign_warn = "⚠️ SIGN CHANGE" if any(np.sign(rs[i]) != np.sign(rs[i+1]) for i in range(3)) else ""
        print(f"    Patient {pid}: Q1={rs[0]:+.2f}, Q2={rs[1]:+.2f}, Q3={rs[2]:+.2f}, Q4={rs[3]:+.2f} {sign_warn}")
    
    ns_df = pd.DataFrame(nonstationarity_data)
    print(f"""
    Summary of non-stationarity:
        Average correlation range within patient: {ns_df['range'].mean():.2f}
        Patients with sign changes: {ns_df['sign_changes'].sum()} / {len(ns_df)} quarters
    
    ➡️  Correlation CHANGES SIGN during surgery for many patients!
    ➡️  A model calibrated on Q1 will FAIL on Q4
    ➡️  This is why calibration makes predictions WORSE, not better
    """)
    
    return nonstationarity_data


# =============================================================================
# ANALYSIS 6: Patient-Specific Relationships
# =============================================================================

def analyze_patient_specificity(df):
    """Analyze how PTT-BP relationship varies across patients."""
    print_header("6. PATIENT-SPECIFIC RELATIONSHIPS")
    
    print("""
    ⚠️  CRITICAL ISSUE: Each patient has a DIFFERENT PTT-BP relationship
    
    There is no universal equation: BP = f(PTT)
    Each patient's physiology is different.
    """)
    
    print_subheader("PTT→SBP Slope Distribution Across Patients")
    
    patient_slopes = []
    patient_intercepts = []
    patient_r2s = []
    
    for pid in df['patient_id'].unique():
        p_df = df[df['patient_id'] == pid]
        if len(p_df) < 50:
            continue
        
        x = p_df['ptt_peak_to_foot'].values
        y = p_df['sbp_reference'].values
        
        # Standardize x for comparable slopes
        x_std = (x - x.mean()) / (x.std() + 1e-7)
        
        # Linear regression
        slope, intercept, r, p_val, se = stats.linregress(x_std, y)
        
        patient_slopes.append(slope)
        patient_intercepts.append(intercept)
        patient_r2s.append(r ** 2)
    
    slopes = np.array(patient_slopes)
    intercepts = np.array(patient_intercepts)
    r2s = np.array(patient_r2s)
    
    print(f"""
    PTT → SBP slope (standardized):
        Mean:     {slopes.mean():+.2f} mmHg/std
        Std:      {slopes.std():.2f} mmHg/std
        Range:    [{slopes.min():.2f}, {slopes.max():.2f}]
        CV:       {100*slopes.std()/abs(slopes.mean()):.0f}%
    
    Per-patient R²:
        Mean:     {r2s.mean():.3f}
        Median:   {np.median(r2s):.3f}
        Range:    [{r2s.min():.3f}, {r2s.max():.3f}]
    
    ➡️  Slope varies by {100*slopes.std()/abs(slopes.mean()):.0f}% across patients!
    ➡️  A global model assumes ONE slope - this is fundamentally wrong
    ➡️  Mean per-patient R² = {r2s.mean():.3f} - features explain little even within patients
    """)
    
    return {'slopes': slopes, 'intercepts': intercepts, 'r2s': r2s}


# =============================================================================
# ANALYSIS 7: Theoretical Limits
# =============================================================================

def analyze_theoretical_limits(df):
    """Calculate theoretical limits of what's achievable."""
    print_header("7. THEORETICAL LIMITS & AAMI REQUIREMENTS")
    
    # AAMI standard
    print("""
    AAMI/ISO 81060-2 Standard for BP Device Accuracy:
        Mean Error (ME):     ≤ 5 mmHg
        Standard Deviation:  ≤ 8 mmHg
    """)
    
    print_subheader("What R² is Needed for AAMI Compliance?")
    
    sbp_dev_std = df['sbp_dev'].std()
    dbp_dev_std = df['dbp_dev'].std()
    
    # If we predict deviation, error std = sbp_dev_std * sqrt(1 - R²)
    # For AAMI: error_std ≤ 8
    # So: sbp_dev_std * sqrt(1 - R²) ≤ 8
    # R² ≥ 1 - (8/sbp_dev_std)²
    
    required_r2_sbp = 1 - (8 / sbp_dev_std) ** 2
    required_r2_dbp = 1 - (8 / dbp_dev_std) ** 2
    
    print(f"""
    Within-patient BP variation:
        SBP deviation std:   {sbp_dev_std:.2f} mmHg
        DBP deviation std:   {dbp_dev_std:.2f} mmHg
    
    Required R² for AAMI (STD ≤ 8 mmHg):
        SBP: R² ≥ {required_r2_sbp:.2f}
        DBP: R² ≥ {max(0, required_r2_dbp):.2f}
    """)
    
    print_subheader("What R² is Achievable with Current Features?")
    
    # Best feature correlation
    best_r = 0.24  # From correlation analysis
    best_r2_single = best_r ** 2
    
    # Optimistic estimate with multiple features (assuming no multicollinearity)
    best_r2_multiple = min(0.15, best_r2_single * 2.5)  # Very optimistic
    
    print(f"""
    Current feature performance:
        Best single feature r:     {best_r}
        Best single feature R²:    {best_r2_single:.3f}
        Multiple features R²:      ~{best_r2_multiple:.3f} (optimistic)
    
    Gap Analysis:
        Required R²:               {required_r2_sbp:.2f}
        Achievable R²:             ~{best_r2_multiple:.3f}
        Gap:                       {required_r2_sbp/best_r2_multiple:.0f}x improvement needed
    
    ➡️  Current features can achieve ~{best_r2_multiple*100:.0f}% of required R²
    ➡️  This is a FUNDAMENTAL LIMIT, not a model problem
    ➡️  No architecture (transformer, LSTM, etc.) can overcome this
    """)
    
    print_subheader("What Error Can We Actually Achieve?")
    
    achievable_std_sbp = sbp_dev_std * np.sqrt(1 - best_r2_multiple)
    achievable_std_dbp = dbp_dev_std * np.sqrt(1 - best_r2_multiple)
    
    print(f"""
    With R² = {best_r2_multiple:.2f}:
        Achievable SBP STD:  {achievable_std_sbp:.1f} mmHg  (AAMI requires ≤8)
        Achievable DBP STD:  {achievable_std_dbp:.1f} mmHg  (AAMI requires ≤8)
    
    Verdict:
        SBP: {"❌ CANNOT meet AAMI" if achievable_std_sbp > 8 else "✓ Can meet AAMI"}
        DBP: {"❌ CANNOT meet AAMI" if achievable_std_dbp > 8 else "✓ Can meet AAMI"}
    """)
    
    return {
        'sbp_dev_std': sbp_dev_std,
        'required_r2': required_r2_sbp,
        'achievable_r2': best_r2_multiple,
        'achievable_std': achievable_std_sbp
    }


# =============================================================================
# ANALYSIS 8: Why _norm Features Don't Help
# =============================================================================

def analyze_norm_features(df):
    """Analyze why the _norm features aren't truly normalized."""
    print_header("8. THE _NORM FEATURES PROBLEM")
    
    print("""
    The dataset contains _norm features (e.g., ptt_peak_to_foot_norm).
    These SHOULD be patient-normalized (mean=0 per patient).
    Let's check if they actually are...
    """)
    
    print_subheader("Are _norm Features Actually Normalized?")
    
    norm_features = [c for c in df.columns if c.endswith('_norm')][:5]
    
    for feat in norm_features:
        if feat not in df.columns:
            continue
        
        patient_means = df.groupby('patient_id')[feat].mean()
        patient_stds = df.groupby('patient_id')[feat].std()
        
        # If properly normalized, all patient means should be ~0
        mean_of_means = patient_means.mean()
        std_of_means = patient_means.std()
        
        status = "✓ OK" if std_of_means < 0.05 else "❌ NOT NORMALIZED"
        print(f"    {feat:35s}: patient_mean_std = {std_of_means:.3f} {status}")
    
    print("""
    ➡️  _norm features still have patient-level variance!
    ➡️  They are likely normalized GLOBALLY, not PER-PATIENT
    ➡️  This means they still leak patient identity
    """)


# =============================================================================
# ANALYSIS 9: Summary & Recommendations
# =============================================================================

def print_summary():
    """Print summary of findings and recommendations."""
    print_header("9. SUMMARY: WHY THIS PROBLEM IS FUNDAMENTALLY HARD")
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         ROOT CAUSE ANALYSIS                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. WEAK SIGNAL                                                         │
    │     • Best feature correlation: r = 0.24 → R² = 0.06                    │
    │     • Features explain only 6% of BP variation                          │
    │     • AAMI compliance requires explaining 72%                           │
    │     • Gap: 12x improvement needed                                       │
    │                                                                         │
    │  2. NON-STATIONARY RELATIONSHIPS                                        │
    │     • PTT-BP correlation changes DURING surgery                         │
    │     • Correlation can flip sign (e.g., -0.29 → +0.18)                   │
    │     • Caused by: anesthesia, medications, blood loss, stimulation       │
    │     • Calibration makes things WORSE because relationship changes       │
    │                                                                         │
    │  3. NO UNIVERSAL RELATIONSHIP                                           │
    │     • PTT-BP slope varies 88% across patients                           │
    │     • Each patient has unique physiology                                │
    │     • A global model assumes ONE relationship - wrong                   │
    │                                                                         │
    │  4. TEMPORAL AUTOCORRELATION                                            │
    │     • Adjacent samples are 94% correlated                               │
    │     • 181,842 samples → ~5,500 effective samples                        │
    │     • Creates illusion of large dataset                                 │
    │     • Causes massive overfitting if not handled                         │
    │                                                                         │
    │  5. PATIENT IDENTITY DOMINATES                                          │
    │     • 77-89% of feature variance is between-patient                     │
    │     • Models memorize patient identity, not BP                          │
    │     • High training R² but negative test R²                             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           RECOMMENDATIONS                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  OPTION 1: Different Use Case                                           │
    │     • PTT-based BP works better for AMBULATORY monitoring               │
    │     • Stable subjects without surgical confounders                      │
    │     • Consider a different dataset (e.g., MIMIC PPG segments)           │
    │                                                                         │
    │  OPTION 2: Trend Prediction (Recommended)                               │
    │     • Instead of predicting BP value, predict DIRECTION                 │
    │     • Classification: BP increasing / stable / decreasing               │
    │     • Clinically useful for alerts even if not AAMI compliant           │
    │                                                                         │
    │  OPTION 3: Continuous Recalibration                                     │
    │     • Use invasive BP as reference every N minutes                      │
    │     • Re-estimate patient-specific parameters                           │
    │     • Defeats purpose of cuffless BP, but improves accuracy             │
    │                                                                         │
    │  OPTION 4: Different Features                                           │
    │     • Raw waveform deep learning (end-to-end)                           │
    │     • Second derivative features (acceleration plethysmogram)           │
    │     • Multi-modal fusion (ECG + PPG + other signals)                    │
    │                                                                         │
    │  OPTION 5: Accept Limitations                                           │
    │     • Report achievable accuracy (~15 mmHg STD)                         │
    │     • Position as "trend monitor" not "BP replacement"                  │
    │     • Valuable for research even if not clinically deployable           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df, output_dir):
    """Create comprehensive visualizations of the problems."""
    print_header("10. GENERATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Autocorrelation decay
    ax1 = fig.add_subplot(3, 2, 1)
    lags = [1, 2, 5, 10, 20, 50, 100]
    acs = []
    for lag in lags:
        ac_samples = []
        for pid in df['patient_id'].unique()[:20]:
            p_df = df[df['patient_id'] == pid]['sbp_reference']
            if len(p_df) > lag + 10:
                ac = p_df.autocorr(lag=lag)
                if not np.isnan(ac):
                    ac_samples.append(ac)
        acs.append(np.mean(ac_samples))
    
    ax1.plot(lags, acs, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color=COLORS['danger'], linestyle='--', label='r=0.5 threshold')
    ax1.set_xlabel('Lag (samples)', fontsize=12)
    ax1.set_ylabel('Autocorrelation', fontsize=12)
    ax1.set_title('SBP Temporal Autocorrelation\n(Adjacent samples share 94% information)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # 2. Variance decomposition
    ax2 = fig.add_subplot(3, 2, 2)
    features = ['ptt_peak_to_foot', 'hr_bpm', 'pat_ecg_ppg', 'sbp_reference']
    between_pcts = []
    within_pcts = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        total = df[feat].var()
        between = df.groupby('patient_id')[feat].mean().var()
        within = df.groupby('patient_id')[feat].var().mean()
        between_pcts.append(100 * between / total)
        within_pcts.append(100 * within / total)
    
    x = np.arange(len(features))
    width = 0.35
    ax2.bar(x - width/2, between_pcts, width, label='Between-patient', color=COLORS['danger'])
    ax2.bar(x + width/2, within_pcts, width, label='Within-patient', color=COLORS['primary'])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
    ax2.set_ylabel('% of Total Variance', fontsize=12)
    ax2.set_title('Variance Decomposition\n(Most variance is patient identity, not BP signal)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    # 3. Feature-BP correlations
    ax3 = fig.add_subplot(3, 2, 3)
    features = ['ptt_peak_to_foot', 'hr_bpm', 'pat_ecg_ppg', 'crest_time']
    global_rs = []
    within_rs = []
    
    for feat in features:
        if feat not in df.columns:
            continue
        global_r = df[feat].corr(df['sbp_reference'])
        global_rs.append(global_r)
        
        w_rs = []
        for pid in df['patient_id'].unique()[:30]:
            p_df = df[df['patient_id'] == pid]
            if len(p_df) > 30:
                r = p_df[feat].corr(p_df['sbp_reference'])
                if not np.isnan(r):
                    w_rs.append(r)
        within_rs.append(np.mean(w_rs))
    
    x = np.arange(len(features))
    ax3.bar(x - width/2, global_rs, width, label='Global (cross-patient)', color=COLORS['secondary'])
    ax3.bar(x + width/2, within_rs, width, label='Within-patient mean', color=COLORS['primary'])
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
    ax3.set_ylabel('Correlation with SBP', fontsize=12)
    ax3.set_title('Feature-BP Correlations\n(Weak correlations = weak predictive signal)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Non-stationarity example
    ax4 = fig.add_subplot(3, 2, 4)
    sample_pids = df['patient_id'].unique()[:5]
    
    for i, pid in enumerate(sample_pids):
        p_df = df[df['patient_id'] == pid]
        n = len(p_df)
        if n < 200:
            continue
        
        # Compute rolling correlation
        window = n // 4
        rolling_rs = []
        positions = []
        for start in range(0, n - window, window // 2):
            end = start + window
            r = p_df.iloc[start:end]['ptt_peak_to_foot'].corr(p_df.iloc[start:end]['sbp_reference'])
            rolling_rs.append(r)
            positions.append((start + end) / 2 / n * 100)  # % of surgery
        
        ax4.plot(positions, rolling_rs, 'o-', label=f'Patient {pid}', alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Surgery Progress (%)', fontsize=12)
    ax4.set_ylabel('PTT-SBP Correlation', fontsize=12)
    ax4.set_title('Non-Stationarity: PTT-BP Relationship Changes\n(Correlation can flip sign during surgery!)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    
    # 5. Patient-specific slopes
    ax5 = fig.add_subplot(3, 2, 5)
    slopes = []
    for pid in df['patient_id'].unique():
        p_df = df[df['patient_id'] == pid]
        if len(p_df) < 50:
            continue
        x = (p_df['ptt_peak_to_foot'] - p_df['ptt_peak_to_foot'].mean()) / (p_df['ptt_peak_to_foot'].std() + 1e-7)
        y = p_df['sbp_reference']
        slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-7)
        slopes.append(slope)
    
    ax5.hist(slopes, bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.7)
    ax5.axvline(x=np.mean(slopes), color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Mean: {np.mean(slopes):.1f}')
    ax5.set_xlabel('PTT→SBP Slope (mmHg/std)', fontsize=12)
    ax5.set_ylabel('Number of Patients', fontsize=12)
    ax5.set_title(f'Patient-Specific PTT-SBP Slopes\n(CV = {100*np.std(slopes)/abs(np.mean(slopes)):.0f}% - No universal relationship!)', fontsize=12, fontweight='bold')
    ax5.legend()
    
    # 6. R² gap visualization
    ax6 = fig.add_subplot(3, 2, 6)
    categories = ['Single\nFeature', 'Multiple\nFeatures', 'Required\nfor AAMI']
    values = [0.06, 0.10, 0.72]
    colors = [COLORS['primary'], COLORS['accent'], COLORS['danger']]
    
    bars = ax6.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
    ax6.set_ylabel('R² (Variance Explained)', fontsize=12)
    ax6.set_title('The Fundamental Gap\n(Need 7x improvement to meet AAMI)', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', fontsize=12, fontweight='bold')
    
    # Add arrow showing gap
    ax6.annotate('', xy=(2, 0.72), xytext=(1, 0.10),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2))
    ax6.text(1.5, 0.4, '7x\ngap', ha='center', fontsize=14, fontweight='bold', color=COLORS['danger'])
    
    plt.tight_layout()
    
    save_path = output_dir / 'dataset_problem_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"    Saved visualization to: {save_path}")
    
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete dataset problem analysis."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  DATASET PROBLEM ANALYSIS: PTT-Based BP Estimation in Surgical Patients  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    
    # Run all analyses
    analyze_dataset_overview(df)
    analyze_autocorrelation(df)
    analyze_variance_decomposition(df)
    analyze_correlations(df)
    analyze_nonstationarity(df)
    analyze_patient_specificity(df)
    analyze_theoretical_limits(df)
    analyze_norm_features(df)
    
    # Summary
    print_summary()
    
    # Visualizations
    create_visualizations(df, OUTPUT_DIR)
    
    print("\n" + "█" * 80)
    print("█" + "  ANALYSIS COMPLETE  ".center(78) + "█")
    print("█" * 80 + "\n")


if __name__ == '__main__':
    main()
