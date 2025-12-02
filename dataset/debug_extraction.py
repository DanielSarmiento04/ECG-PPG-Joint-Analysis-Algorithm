import numpy as np
import matplotlib.pyplot as plt
from fixed_feature_extraction import FixedFeatureExtractor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_extraction():
    # Load a case that failed (e.g., case_02240.npz)
    file_path = './data/raw/case_02240.npz'
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File {file_path} not found. Trying another one...")
        # Try to find any .npz file
        import os
        files = [f for f in os.listdir('./data/raw') if f.endswith('.npz')]
        if not files:
            print("No .npz files found.")
            return
        file_path = os.path.join('./data/raw', files[0])
        data = np.load(file_path, allow_pickle=True)
        
    print(f"Debugging with {file_path}")
    
    ecg = data['wave_SNUADC_ECG_II']
    ppg = data['wave_SNUADC_PLETH']
    fs = 500
    
    # Handle NaNs
    if np.isnan(ecg).any():
        print(f"ECG contains {np.isnan(ecg).sum()} NaNs. Filling with mean.")
        if np.isnan(ecg).all():
            print("ECG is all NaNs. Skipping.")
            return
        ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg))
        
    if np.isnan(ppg).any():
        print(f"PPG contains {np.isnan(ppg).sum()} NaNs. Filling with mean.")
        if np.isnan(ppg).all():
            print("PPG is all NaNs. Skipping.")
            return
        ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg))
    
    # Simple R-peak detection (just to get a cycle)
    from scipy.signal import find_peaks
    
    # Print stats to debug peak detection
    print(f"ECG stats: min={np.min(ecg):.2f}, max={np.max(ecg):.2f}, mean={np.mean(ecg):.2f}")
    
    # Try finding peaks on raw ECG first (often works for clean signals)
    peaks, _ = find_peaks(ecg, height=np.max(ecg)*0.5, distance=fs*0.5)
    
    if len(peaks) == 0:
        print("No peaks on raw ECG. Trying derivative...")
        # Differentiate ECG
        d_ecg = np.diff(ecg)
        peaks, _ = find_peaks(np.abs(d_ecg), height=np.mean(np.abs(d_ecg))*1.5, distance=fs*0.5)
    
    # Check Synchronization
    print("\n--- Synchronization Check ---")
    # Find peaks in PPG to compare timing
    ppg_peaks, _ = find_peaks(ppg, height=np.mean(ppg), distance=fs*0.4)
    
    print(f"First 5 ECG Peaks (samples): {peaks[:5]}")
    print(f"First 5 PPG Peaks (samples): {ppg_peaks[:5]}")
    
    # Calculate delays
    # For each ECG peak, find the nearest following PPG peak
    delays = []
    for r in peaks[:10]:
        # Find first PPG peak after R
        following_ppg = ppg_peaks[ppg_peaks > r]
        if len(following_ppg) > 0:
            p = following_ppg[0]
            # Only consider if it's within reasonable range (e.g. < 600ms)
            if p - r < fs * 0.6:
                delays.append(p - r)
    
    if delays:
        avg_delay_samples = np.mean(delays)
        avg_delay_ms = (avg_delay_samples / fs) * 1000
        print(f"Average Delay (R to PPG Peak): {avg_delay_ms:.1f} ms ({avg_delay_samples:.1f} samples)")
    else:
        print("Could not calculate delays (no matching PPG peaks found nearby).")

    extractor = FixedFeatureExtractor(fs=fs)
    
    print(f"Found {len(peaks)} peaks. Checking first 10 cycles...")
    
    for i in range(min(10, len(peaks)-1)):
        start = peaks[i]
        end = peaks[i+1]
        ppg_cycle = ppg[start:end]
        
        print(f"\nCycle {i}: Len {len(ppg_cycle)}")
        
        # Extract
        features = extractor.extract_features(ppg_cycle, hr_bpm=60, r_peak_sample=start, cycle_start_sample=start)
        
        print("Features:", features)
        
        # Check internal values by calling preprocess manually
        ppg_smooth, d1, d2 = extractor._preprocess(ppg_cycle)
        
        # Normalize
        ppg_min, ppg_max = np.min(ppg_smooth), np.max(ppg_smooth)
        ppg_norm = (ppg_smooth - ppg_min) / (ppg_max - ppg_min)
        
        peak_idx = np.argmax(ppg_norm[:int(len(ppg_norm)*0.7)])
        
        onset_search = d2[:peak_idx] if peak_idx > 2 else []
        if len(onset_search) > 0:
            foot_idx = np.argmax(onset_search)
            print(f"  Peak idx: {peak_idx}, Foot idx: {foot_idx}")
            print(f"  PAT: {(foot_idx/fs)*1000:.1f} ms")
            print(f"  Max D2 at foot: {d2[foot_idx]:.4f}")
            print(f"  D2 start: {d2[0]:.4f}")
        else:
            print("  No onset search region")

if __name__ == "__main__":
    debug_extraction()
