#!/usr/bin/env python3
"""
Test script to verify PTT and amplitude ratio boundary fixes.
"""

import numpy as np
import os
import sys

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixed_feature_extraction import FixedFeatureExtractor, detect_global_ppg_polarity
from scipy.signal import find_peaks


def test_single_case(case_path):
    """Test feature extraction on a single case."""
    print(f"\nTesting: {os.path.basename(case_path)}")
    print("=" * 50)
    
    # Load data (VitalDB format)
    case = np.load(case_path, allow_pickle=True)
    ppg = case['wave_SNUADC_PLETH']
    ecg = case['wave_SNUADC_ECG_II']
    fs = 500
    
    # Handle NaN values
    ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg))
    ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg))
    
    # Find R-peaks with better detection
    from scipy.signal import butter, sosfiltfilt
    
    # Bandpass filter ECG (0.5-40 Hz)
    nyq = fs / 2
    sos = butter(4, [0.5/nyq, 40/nyq], btype='band', output='sos')
    ecg_filt = sosfiltfilt(sos, ecg)
    
    # Find peaks with adaptive threshold
    ecg_std = np.std(ecg_filt)
    r_peaks, _ = find_peaks(ecg_filt, distance=int(0.4*fs), height=0.5*ecg_std, prominence=0.3*ecg_std)
    print(f'Found {len(r_peaks)} R-peaks')
    
    # Test global polarity detection
    is_inverted = detect_global_ppg_polarity(ppg, fs, r_peaks)
    print(f'Global polarity: {"INVERTED" if is_inverted else "NORMAL"}')
    
    # Extract features for first 100 beats
    extractor = FixedFeatureExtractor(fs)
    # Set global polarity by passing the full signal
    extractor.set_global_polarity(ppg, r_peaks)
    
    ptt_values = []
    ra_values = []
    
    n_beats = min(100, len(r_peaks) - 1)
    for i in range(n_beats):
        start = r_peaks[i]
        end = r_peaks[i+1] if i+1 < len(r_peaks) else start + int(fs * 0.8)
        
        if end - start < 100:  # Too short
            continue
            
        ppg_beat = ppg[start:end]
        hr = 60.0 * fs / (end - start)
        
        features = extractor.extract_features(ppg_beat, hr)
        if features:
            ptt_values.append(features.get('ptt_peak_to_foot', 0))
            ra_values.append(features.get('amplitude_ratio_ra', 0))
    
    if len(ptt_values) == 0:
        print("ERROR: No features extracted!")
        return
    
    # PTT analysis
    print(f'\nPTT Analysis ({len(ptt_values)} beats):')
    print(f'  Mean: {np.mean(ptt_values):.1f} ms')
    print(f'  Std:  {np.std(ptt_values):.1f} ms')
    print(f'  Range: {np.min(ptt_values):.1f} - {np.max(ptt_values):.1f} ms')
    
    # Check for boundary clustering
    boundaries = [50, 60, 80, 100, 120]
    print(f'\n  Boundary clustering (±1ms):')
    for b in boundaries:
        count = sum(b-1 < p < b+1 for p in ptt_values)
        pct = 100 * count / len(ptt_values)
        marker = " <-- ISSUE!" if pct > 10 else ""
        print(f'    {b}ms: {count} ({pct:.1f}%){marker}')
    
    # Amplitude ratio analysis
    print(f'\nAmplitude Ratio Analysis:')
    print(f'  Mean: {np.mean(ra_values):.3f}')
    print(f'  Std:  {np.std(ra_values):.3f}')
    print(f'  Range: {np.min(ra_values):.3f} - {np.max(ra_values):.3f}')
    
    # Check for boundary clustering
    ra_boundaries = [0.50, 0.85, 0.98, 0.99, 1.00]
    print(f'\n  Boundary clustering (±0.005):')
    for b in ra_boundaries:
        count = sum(b-0.005 < r < b+0.005 for r in ra_values)
        pct = 100 * count / len(ra_values)
        marker = " <-- ISSUE!" if pct > 10 else ""
        print(f'    {b:.2f}: {count} ({pct:.1f}%){marker}')


def main():
    # Find test cases
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    # Get first 3 cases
    cases = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])[:3]
    
    if not cases:
        print("ERROR: No .npz files found")
        return
    
    print("=" * 60)
    print("PTT AND AMPLITUDE RATIO BOUNDARY FIX VERIFICATION")
    print("=" * 60)
    
    for case_file in cases:
        case_path = os.path.join(data_dir, case_file)
        test_single_case(case_path)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
