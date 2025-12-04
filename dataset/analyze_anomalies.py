"""
Analyze anomalies in PTT_peak_to_foot and amplitude_ratio_ra distributions
"""
import pandas as pd
import numpy as np

def analyze_ptt_boundary():
    df = pd.read_csv('data/processed/bp_dataset_features.csv')
    
    print('='*70)
    print('DEEP DIVE: PTT_PEAK_TO_FOOT BOUNDARY ISSUE')
    print('='*70)
    
    ptt_foot = df['ptt_peak_to_foot']
    
    # Check exact values at the boundary
    print('\n--- Exact values in 20-32ms range ---')
    boundary_values = ptt_foot[(ptt_foot >= 20) & (ptt_foot <= 32)]
    print(boundary_values.value_counts().sort_index().head(20))
    
    # Check if these are per-patient or spread across all patients
    print('\n--- Per-patient analysis of 24-28ms values ---')
    boundary_mask = (ptt_foot >= 24) & (ptt_foot <= 28)
    boundary_df = df[boundary_mask]
    print(f'Patients with values in 24-28ms: {boundary_df["patient_id"].nunique()}')
    for pid in boundary_df['patient_id'].unique()[:8]:
        count = len(boundary_df[boundary_df['patient_id'] == pid])
        total = len(df[df['patient_id'] == pid])
        print(f'  Patient {pid}: {count}/{total} ({count/total*100:.1f}%) in boundary')
    
    # Check if it correlates with specific features
    print('\n--- Features for boundary vs non-boundary samples ---')
    print(f'\nBoundary (24-28ms) samples: {len(boundary_df):,}')
    print(f'  Mean SBP: {boundary_df["sbp_reference"].mean():.1f}')
    print(f'  Mean DBP: {boundary_df["dbp_reference"].mean():.1f}')
    print(f'  Mean HR: {boundary_df["hr_bpm"].mean():.1f}')
    
    non_boundary = df[~boundary_mask]
    print(f'\nNon-boundary samples: {len(non_boundary):,}')
    print(f'  Mean SBP: {non_boundary["sbp_reference"].mean():.1f}')
    print(f'  Mean DBP: {non_boundary["dbp_reference"].mean():.1f}')
    print(f'  Mean HR: {non_boundary["hr_bpm"].mean():.1f}')
    
    # Check pat_ecg_ppg (should be same as ptt_peak_to_foot)
    print('\n--- Comparing pat_ecg_ppg vs ptt_peak_to_foot ---')
    if 'pat_ecg_ppg' in df.columns:
        diff = (df['pat_ecg_ppg'] - df['ptt_peak_to_foot']).abs()
        print(f'Max difference: {diff.max():.2f} ms')
        print(f'Samples with difference > 1ms: {(diff > 1).sum()}')
        print(f'\npat_ecg_ppg distribution:')
        pat = df['pat_ecg_ppg']
        print(f'  Mean: {pat.mean():.2f} ms')
        print(f'  Median: {pat.median():.2f} ms')
        print(f'  Min: {pat.min():.2f} ms')
        print(f'  Max: {pat.max():.2f} ms')
        print(f'  Values exactly 24ms: {(pat == 24).sum():,}')
        print(f'  Values exactly 26ms: {(pat == 26).sum():,}')
    
    # Check pat_to_peak distribution
    print('\n--- pat_to_peak distribution ---')
    if 'pat_to_peak' in df.columns:
        pat_peak = df['pat_to_peak']
        print(f'  Mean: {pat_peak.mean():.2f} ms')
        print(f'  Median: {pat_peak.median():.2f} ms')
        print(f'  Min: {pat_peak.min():.2f} ms')
        print(f'  Max: {pat_peak.max():.2f} ms')
        # Check if more spread out
        print(f'  Values < 100ms: {(pat_peak < 100).sum():,} ({(pat_peak < 100).mean()*100:.1f}%)')
        print(f'  Values 100-200ms: {((pat_peak >= 100) & (pat_peak < 200)).sum():,}')
        print(f'  Values 200-400ms: {((pat_peak >= 200) & (pat_peak < 400)).sum():,}')
        print(f'  Values >= 400ms: {(pat_peak >= 400).sum():,}')

def analyze_amplitude_ratio():
    df = pd.read_csv('data/processed/bp_dataset_features.csv')
    
    print('\n' + '='*70)
    print('DEEP DIVE: AMPLITUDE_RATIO_RA FALLBACK ISSUE')
    print('='*70)
    
    amp_ratio = df['amplitude_ratio_ra']
    
    # Check per-patient distribution
    print('\n--- Per-patient amplitude ratio analysis ---')
    for pid in df['patient_id'].unique()[:8]:
        patient_amp = df[df['patient_id'] == pid]['amplitude_ratio_ra']
        fallback_pct = (patient_amp == 0.5).mean() * 100
        print(f'  Patient {pid}: {fallback_pct:.1f}% at 0.5 fallback, mean={patient_amp.mean():.3f}')
    
    # Check if there's any pattern
    print('\n--- Samples NOT at fallback (not 0.5) ---')
    non_fallback = df[df['amplitude_ratio_ra'] != 0.5]
    print(f'Total non-fallback samples: {len(non_fallback):,} ({len(non_fallback)/len(df)*100:.1f}%)')
    if len(non_fallback) > 0:
        print(f'  Mean amp_ratio: {non_fallback["amplitude_ratio_ra"].mean():.3f}')
        print(f'  Median amp_ratio: {non_fallback["amplitude_ratio_ra"].median():.3f}')
        print(f'  Mean SBP: {non_fallback["sbp_reference"].mean():.1f} vs {df["sbp_reference"].mean():.1f}')
        print(f'  Mean HR: {non_fallback["hr_bpm"].mean():.1f} vs {df["hr_bpm"].mean():.1f}')

def check_foot_detection_code():
    print('\n' + '='*70)
    print('CHECKING: Minimum search window in foot detection')
    print('='*70)
    
    # At 500Hz, check what 24ms corresponds to
    sample_rate = 500
    ms_to_samples = lambda ms: int(ms * sample_rate / 1000)
    samples_to_ms = lambda s: s * 1000 / sample_rate
    
    print(f'\nAt 500Hz sampling rate:')
    print(f'  24ms = {ms_to_samples(24)} samples')
    print(f'  25ms = {ms_to_samples(25)} samples  <- current min_start_samples?')
    print(f'  50ms = {ms_to_samples(50)} samples')
    
    print(f'\nIf min_start_samples = 12:')
    print(f'  12 samples = {samples_to_ms(12):.0f}ms')
    print(f'  Foot found at sample 12 => 24ms PAT')
    
    print('\n>>> The issue: foot search starts at sample 12 (24ms)')
    print('>>> When foot detection fails, it returns sample 12 (minimum)')
    print('>>> This causes 52% of values to be exactly 24-26ms')

if __name__ == '__main__':
    analyze_ptt_boundary()
    analyze_amplitude_ratio()
    check_foot_detection_code()
