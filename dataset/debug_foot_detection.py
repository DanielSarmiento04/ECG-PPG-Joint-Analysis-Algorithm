#!/usr/bin/env python3
"""
Debug script to trace through foot detection and understand boundary clustering.
"""

import numpy as np
import os
import sys
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixed_feature_extraction import (
    FixedFeatureExtractor, 
    detect_global_ppg_polarity,
    remove_spikes
)


def debug_foot_detection_single_beat(ppg_beat, fs=500, verbose=True):
    """
    Trace through the foot detection algorithm step by step.
    """
    results = {
        'input_length': len(ppg_beat),
        'steps': []
    }
    
    max_search_ms = 400
    max_search_samples = min(int(max_search_ms / 1000 * fs), len(ppg_beat))
    
    # Step 1: Remove spikes
    ppg_clean = remove_spikes(ppg_beat, threshold_mad=4.0)
    
    # Step 2: Remove baseline trend
    if len(ppg_clean) > 20:
        x = np.arange(len(ppg_clean))
        coeffs = np.polyfit(x, ppg_clean, 1)
        trend = np.polyval(coeffs, x)
        ppg_work = ppg_clean - trend + np.mean(ppg_clean)
    else:
        ppg_work = ppg_clean
    
    # Step 3: Normalize
    ppg_min = np.min(ppg_work)
    ppg_max = np.max(ppg_work)
    if ppg_max - ppg_min < 1e-6:
        results['error'] = 'flat_signal'
        return results
    
    ppg_norm = (ppg_work - ppg_min) / (ppg_max - ppg_min)
    search_region = ppg_norm[:max_search_samples]
    smoothed = gaussian_filter1d(search_region, sigma=3)
    
    # Step 4: Find systolic peak
    peak_search_start = int(0.080 * fs)  # 80ms = 40 samples
    peak_search_end = min(int(0.450 * fs), len(smoothed))  # 450ms = 225 samples
    
    if peak_search_end <= peak_search_start + 10:
        results['error'] = 'peak_search_region_too_small'
        return results
    
    peak_region = smoothed[peak_search_start:peak_search_end]
    local_peak_idx = int(np.argmax(peak_region))
    peak_idx = peak_search_start + local_peak_idx
    peak_time_ms = peak_idx / fs * 1000
    
    results['peak_idx'] = peak_idx
    results['peak_time_ms'] = peak_time_ms
    
    # Step 5: Define foot search region
    foot_search_start = int(0.050 * fs)  # 50ms = 25 samples
    foot_search_end = peak_idx - int(0.020 * fs)  # 20ms before peak
    
    results['foot_search_start'] = foot_search_start
    results['foot_search_start_ms'] = foot_search_start / fs * 1000
    results['foot_search_end'] = foot_search_end
    results['foot_search_end_ms'] = foot_search_end / fs * 1000
    
    if foot_search_end <= foot_search_start + 10:
        results['error'] = 'foot_search_region_too_small'
        results['fallback_used'] = 'peak_too_early'
        foot_idx = max(int(peak_idx * 0.6), int(0.050 * fs))
        results['final_foot_idx'] = foot_idx
        results['final_foot_ms'] = foot_idx / fs * 1000
        return results
    
    # Calculate derivatives
    d1 = np.diff(smoothed)
    d2 = np.diff(d1)
    
    # === Method 1: Max second derivative ===
    d2_start = max(0, foot_search_start)
    d2_end = min(len(d2), foot_search_end)
    
    foot_d2 = foot_search_start
    if d2_end > d2_start and d2_end <= len(d2):
        d2_search = d2[d2_start:d2_end]
        if len(d2_search) > 5:
            d2_smooth = gaussian_filter1d(d2_search, sigma=2)
            max_d2_idx = int(np.argmax(d2_smooth))
            foot_d2 = foot_search_start + max_d2_idx
    
    results['method1_d2'] = foot_d2
    results['method1_d2_ms'] = foot_d2 / fs * 1000
    
    # === Method 2: Intersecting tangent ===
    d1_start = foot_search_start
    d1_end = min(len(d1), foot_search_end)
    
    foot_tangent = foot_search_start
    if d1_end > d1_start:
        d1_search = d1[d1_start:d1_end]
        if len(d1_search) > 5:
            max_slope_local_idx = int(np.argmax(d1_search))
            max_slope_idx = foot_search_start + max_slope_local_idx
            max_slope = d1_search[max_slope_local_idx]
            
            if max_slope > 0.005:
                ppg_at_slope = smoothed[max_slope_idx]
                baseline_region = smoothed[foot_search_start:max_slope_idx] if max_slope_idx > foot_search_start else [smoothed[foot_search_start]]
                baseline = np.min(baseline_region)
                
                foot_tangent = max_slope_idx - (ppg_at_slope - baseline) / max_slope
                foot_tangent = int(max(foot_search_start, min(foot_tangent, foot_search_end)))
    
    results['method2_tangent'] = foot_tangent
    results['method2_tangent_ms'] = foot_tangent / fs * 1000
    
    # === Method 3: 10% threshold ===
    foot_search_region = smoothed[foot_search_start:foot_search_end]
    foot_thresh = foot_search_start
    if len(foot_search_region) > 5:
        min_in_region = np.min(foot_search_region)
        max_in_region = smoothed[peak_idx] if peak_idx < len(smoothed) else np.max(foot_search_region)
        threshold_10 = min_in_region + 0.10 * (max_in_region - min_in_region)
        
        above_thresh = np.where(foot_search_region > threshold_10)[0]
        if len(above_thresh) > 0:
            foot_thresh = foot_search_start + int(above_thresh[0])
    
    results['method3_thresh'] = foot_thresh
    results['method3_thresh_ms'] = foot_thresh / fs * 1000
    
    # === Method 4: Minimum ===
    foot_min = foot_search_start
    if len(foot_search_region) > 0:
        foot_min = foot_search_start + int(np.argmin(foot_search_region))
    
    results['method4_min'] = foot_min
    results['method4_min_ms'] = foot_min / fs * 1000
    
    # === Consensus ===
    all_candidates = [foot_d2, foot_tangent, foot_thresh, foot_min]
    results['all_candidates'] = all_candidates
    results['all_candidates_ms'] = [c / fs * 1000 for c in all_candidates]
    
    # Physiological bounds
    min_foot_ms = 60
    max_foot_ms = min(300, peak_time_ms - 10)
    min_foot_samples = int(min_foot_ms / 1000 * fs)
    max_foot_samples = int(max_foot_ms / 1000 * fs)
    
    results['min_foot_samples'] = min_foot_samples
    results['max_foot_samples'] = max_foot_samples
    
    # Filter candidates
    candidates = [c for c in all_candidates if min_foot_samples <= c <= max_foot_samples]
    results['candidates_in_range'] = candidates
    results['candidates_in_range_ms'] = [c / fs * 1000 for c in candidates]
    
    if len(candidates) == 0:
        # Soft minimum fallback
        soft_min = int(0.050 * fs)
        candidates = [c for c in all_candidates if soft_min <= c <= max_foot_samples]
        results['soft_fallback'] = True
        results['candidates_soft'] = candidates
    
    if len(candidates) == 0:
        foot_idx = max(int(peak_idx * 0.5), int(0.050 * fs))
        results['peak_ratio_fallback'] = True
    elif len(candidates) == 1:
        foot_idx = candidates[0]
    else:
        foot_idx = int(np.median(candidates))
    
    # Final bounds
    foot_idx = max(foot_idx, int(0.050 * fs))
    foot_idx = min(foot_idx, max_foot_samples)
    
    results['final_foot_idx'] = foot_idx
    results['final_foot_ms'] = foot_idx / fs * 1000
    results['hit_minimum_bound'] = (foot_idx == int(0.050 * fs))
    
    return results


def analyze_case(case_path, n_beats=20):
    """Analyze foot detection on a case."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {os.path.basename(case_path)}")
    print('='*70)
    
    # Load data
    case = np.load(case_path, allow_pickle=True)
    ppg = case['wave_SNUADC_PLETH']
    ecg = case['wave_SNUADC_ECG_II']
    fs = 500
    
    # Handle NaN
    ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg))
    ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg))
    
    # Find R-peaks
    nyq = fs / 2
    sos = butter(4, [0.5/nyq, 40/nyq], btype='band', output='sos')
    ecg_filt = sosfiltfilt(sos, ecg)
    ecg_std = np.std(ecg_filt)
    r_peaks, _ = find_peaks(ecg_filt, distance=int(0.4*fs), height=0.5*ecg_std, prominence=0.3*ecg_std)
    
    print(f"Found {len(r_peaks)} R-peaks")
    
    # Detect polarity
    is_inverted = detect_global_ppg_polarity(ppg, fs, r_peaks)
    print(f"PPG polarity: {'INVERTED' if is_inverted else 'NORMAL'}")
    
    # Analyze beats
    hits_minimum = 0
    method_stats = {'d2': [], 'tangent': [], 'thresh': [], 'min': []}
    
    for i in range(min(n_beats, len(r_peaks) - 1)):
        start = r_peaks[i]
        end = r_peaks[i + 1] if i + 1 < len(r_peaks) else start + int(fs * 0.8)
        
        if end - start < 100:
            continue
        
        ppg_beat = ppg[start:end]
        
        # Apply polarity correction
        if is_inverted:
            ppg_beat = -ppg_beat
        
        results = debug_foot_detection_single_beat(ppg_beat, fs)
        
        if 'error' in results:
            print(f"Beat {i}: ERROR - {results['error']}")
            continue
        
        method_stats['d2'].append(results['method1_d2_ms'])
        method_stats['tangent'].append(results['method2_tangent_ms'])
        method_stats['thresh'].append(results['method3_thresh_ms'])
        method_stats['min'].append(results['method4_min_ms'])
        
        if results['hit_minimum_bound']:
            hits_minimum += 1
            if hits_minimum <= 5:  # Print first 5 that hit minimum
                print(f"\nBeat {i} HIT MINIMUM BOUND:")
                print(f"  Peak at: {results['peak_time_ms']:.1f}ms")
                print(f"  Search region: {results['foot_search_start_ms']:.1f} - {results['foot_search_end_ms']:.1f}ms")
                print(f"  Method estimates (ms):")
                print(f"    d2:      {results['method1_d2_ms']:.1f}")
                print(f"    tangent: {results['method2_tangent_ms']:.1f}")
                print(f"    thresh:  {results['method3_thresh_ms']:.1f}")
                print(f"    min:     {results['method4_min_ms']:.1f}")
                print(f"  All candidates: {results['all_candidates_ms']}")
                print(f"  In range (60-{results['max_foot_samples']/fs*1000:.0f}ms): {results.get('candidates_in_range_ms', [])}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {hits_minimum}/{n_beats} beats hit minimum bound ({100*hits_minimum/n_beats:.1f}%)")
    print(f"\nMethod statistics (ms):")
    for method, values in method_stats.items():
        if values:
            print(f"  {method:10s}: mean={np.mean(values):.1f}, std={np.std(values):.1f}, min={np.min(values):.1f}, max={np.max(values):.1f}")
    
    # Key insight: check if methods are returning values < 60ms
    print(f"\nMethods returning values below 60ms:")
    for method, values in method_stats.items():
        below_60 = sum(1 for v in values if v < 60)
        print(f"  {method}: {below_60}/{len(values)} ({100*below_60/len(values) if values else 0:.1f}%)")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    cases = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])[:3]
    
    print("="*70)
    print("FOOT DETECTION DEBUG ANALYSIS")
    print("="*70)
    
    for case_file in cases:
        case_path = os.path.join(data_dir, case_file)
        analyze_case(case_path, n_beats=50)


if __name__ == "__main__":
    main()
