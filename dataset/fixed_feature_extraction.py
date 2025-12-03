#!/usr/bin/env python3
"""
Fixed Feature Extraction for Blood Pressure Prediction

This module contains corrected implementations of the feature extraction
functions that address the bugs identified in the forensic analysis.

Key Fixes:
1. PTT calculation: Now correctly computes R-peak to PPG foot
2. Dicrotic notch detection: Uses second derivative for robustness
3. Amplitude ratio: Proper normalization and fallback handling
4. Added new predictive features with known BP correlation
5. ADDED: Robust signal quality checks (preprocess_signal, signal_quality_check)
6. ADDED: Adaptive peak detection with prominence requirements
7. ADDED: Extraction failure logging for debugging
"""

import numpy as np
from scipy import signal as sig
from scipy.stats import skew, kurtosis, pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, find_peaks
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Extraction failure tracking for debugging
_extraction_failures = {
    'flat_signal': 0,
    'no_upstroke': 0,
    'invalid_ptt': 0,
    'low_quality': 0,
    'nan_features': 0,
    'total_attempts': 0
}


def get_extraction_stats() -> Dict:
    """Return extraction failure statistics."""
    return _extraction_failures.copy()


def reset_extraction_stats():
    """Reset extraction failure statistics."""
    global _extraction_failures
    for key in _extraction_failures:
        _extraction_failures[key] = 0


def preprocess_signal(signal_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Clean signal with bandpass filter to remove baseline drift and high-freq noise.
    
    Args:
        signal_data: Raw signal
        fs: Sampling frequency in Hz
        
    Returns:
        Filtered signal
    """
    if len(signal_data) < 20:
        return signal_data
    
    # Handle NaN values
    if np.isnan(signal_data).any():
        signal_data = np.nan_to_num(signal_data, nan=float(np.nanmean(signal_data)))
    
    # Bandpass filter 0.5-10 Hz (removes baseline drift and high-freq noise)
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = min(10.0, nyquist * 0.9) / nyquist  # Ensure high < nyquist
    
    try:
        sos = butter(4, [low, high], btype='band', output='sos')
        filtered = sig.sosfiltfilt(sos, signal_data)
        return filtered
    except Exception:
        # Fallback to original if filtering fails
        return signal_data


def signal_quality_check(ppg_segment: np.ndarray, min_snr: float = 3.0) -> Tuple[bool, str]:
    """
    Check if signal is good enough for feature extraction.
    
    Args:
        ppg_segment: PPG waveform segment
        min_snr: Minimum signal-to-noise ratio
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Check 1: Sufficient length
    if len(ppg_segment) < 20:
        return False, "segment_too_short"
    
    # Check 2: Not all NaN
    if np.isnan(ppg_segment).all():
        return False, "all_nan"
    
    # Check 3: Sufficient amplitude variation
    ppg_std = np.nanstd(ppg_segment)
    ppg_range = np.nanmax(ppg_segment) - np.nanmin(ppg_segment)
    
    if ppg_range < 1e-6:
        return False, "flat_signal"
    
    if ppg_std < 1e-6:
        return False, "no_variation"
    
    # Check 4: Not saturated (clipping)
    unique_vals = len(np.unique(ppg_segment))
    if unique_vals < 5:
        return False, "saturated"
    
    # Check 5: SNR estimate (simple version)
    # SNR = signal variance / noise variance (estimated from high-freq components)
    diff = np.diff(ppg_segment)
    noise_estimate = np.std(diff) / np.sqrt(2)  # Noise std estimate
    
    if noise_estimate > 0:
        snr = ppg_std / noise_estimate
        if snr < min_snr:
            return False, f"low_snr_{snr:.2f}"
    
    return True, "ok"


def detect_peaks_adaptive(signal_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Adaptive peak detection that adjusts to signal amplitude.
    
    Args:
        signal_data: Signal waveform
        fs: Sampling frequency
        
    Returns:
        Array of peak indices
    """
    if len(signal_data) < 10:
        return np.array([])
    
    # Calculate adaptive threshold based on signal statistics
    signal_mean = np.mean(signal_data)
    signal_std = np.std(signal_data)
    
    if signal_std < 1e-6:
        return np.array([])
    
    # Threshold = mean + 0.5*std (adaptive to signal amplitude)
    threshold = signal_mean + 0.5 * signal_std
    
    # Minimum distance between peaks = 0.4 seconds (HR 150 bpm max)
    min_distance = max(int(0.4 * fs), 1)
    
    # Minimum prominence = 0.3 * std
    min_prominence = 0.3 * signal_std
    
    try:
        peaks, properties = find_peaks(
            signal_data,
            height=threshold,
            distance=min_distance,
            prominence=min_prominence
        )
        return peaks
    except Exception:
        # Fallback: simple peak detection
        peaks, _ = find_peaks(signal_data, distance=min_distance)
        return peaks


def calculate_waveform_correlation(ppg: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate correlation between PPG segment and a reference waveform.
    
    Args:
        ppg: PPG segment
        reference: Reference waveform (e.g., template or BP waveform)
        
    Returns:
        Absolute correlation coefficient (0-1)
    """
    if len(ppg) != len(reference):
        # Resample to match lengths
        target_len = min(len(ppg), len(reference))
        ppg = np.asarray(sig.resample(ppg, target_len))
        reference = np.asarray(sig.resample(reference, target_len))
    
    # Normalize signals
    ppg_std = np.std(ppg)
    ref_std = np.std(reference)
    
    if ppg_std < 1e-6 or ref_std < 1e-6:
        return 0.0
    
    ppg_norm = (ppg - np.mean(ppg)) / ppg_std
    ref_norm = (reference - np.mean(reference)) / ref_std
    
    try:
        result = pearsonr(ppg_norm, ref_norm)
        # pearsonr returns (correlation, pvalue) - use numpy to extract
        corr = float(np.asarray(result)[0])
        return abs(corr)
    except Exception:
        return 0.0


def log_extraction_failure(reason: str):
    """Log extraction failure for debugging."""
    global _extraction_failures
    _extraction_failures['total_attempts'] += 1
    if reason in _extraction_failures:
        _extraction_failures[reason] += 1
    logger.debug(f"Feature extraction failed: {reason}")


class FixedFeatureExtractor:
    """
    Corrected feature extraction for PPG-based BP estimation.
    
    Addresses bugs in:
    - ptt_peak_to_foot (was 93% zeros)
    - amplitude_ratio_ra (was 78% near-zero)
    - time_to_maxslope_t1 (many zeros)
    """
    
    def __init__(self, fs: int = 500):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.expected_ptt_range = (100, 400)  # ms
        
    def find_ppg_foot(self, ppg_segment: np.ndarray, 
                      max_search_ms: float = 400) -> Tuple[int, float]:
        """
        Find PPG foot (onset) using multiple methods with consensus.
        
        Methods used:
        1. Second derivative maximum (acceleration peak before upstroke)
        2. Intersecting tangent (line from max slope to baseline)
        3. Threshold crossing (10% of pulse amplitude)
        
        The foot is expected at 50-200ms after R-peak for normal physiology.
        
        Args:
            ppg_segment: PPG waveform starting from R-peak
            max_search_ms: Maximum time to search for foot (ms)
            
        Returns:
            Tuple of (foot_index, foot_time_ms)
        """
        max_search_samples = min(int(max_search_ms / 1000 * self.fs), len(ppg_segment))
        
        if len(ppg_segment) < 50:  # Need at least 100ms of signal
            return 0, 0.0
        
        # Detect PPG polarity: In some systems, PPG is inverted
        mid_point = len(ppg_segment) // 2
        first_quarter = np.mean(ppg_segment[:len(ppg_segment)//4])
        second_quarter = np.mean(ppg_segment[len(ppg_segment)//4:mid_point])
        is_inverted = first_quarter > second_quarter
        
        # Work with signal in "normal" orientation
        ppg_work = -ppg_segment if is_inverted else ppg_segment
            
        # Normalize the segment
        ppg_min = np.min(ppg_work)
        ppg_max = np.max(ppg_work)
        if ppg_max - ppg_min < 1e-6:
            return 0, 0.0
        ppg_norm = (ppg_work - ppg_min) / (ppg_max - ppg_min)
        
        search_region = ppg_norm[:max_search_samples]
        
        # Smooth the signal
        smoothed = gaussian_filter1d(search_region, sigma=2)
        
        # Skip first 25ms (12-13 samples at 500Hz) to avoid ECG artifact
        # The PPG foot physiologically occurs at ~50-200ms after R-peak
        skip_samples = int(0.025 * self.fs)  # 25ms = 12-13 samples
        
        # Find systolic peak in region 50-500ms (skip artifacts at start)
        peak_search_start = int(0.050 * self.fs)  # 50ms
        peak_search_end = min(int(0.500 * self.fs), len(smoothed))  # 500ms
        
        if peak_search_end <= peak_search_start + 10:
            return 0, 0.0
            
        peak_region = smoothed[peak_search_start:peak_search_end]
        local_peak_idx = int(np.argmax(peak_region))
        peak_idx = peak_search_start + local_peak_idx
        
        # Search for foot between 25ms and peak (but not in the artifact region)
        foot_search_start = skip_samples
        foot_search_end = peak_idx
        
        if foot_search_end <= foot_search_start + 5:
            # Very short search region - use a default
            foot_idx = int(0.075 * self.fs)  # Default to 75ms
            return foot_idx, float(foot_idx / self.fs * 1000)
        
        foot_search_region = smoothed[foot_search_start:foot_search_end]
        
        if len(foot_search_region) < 5:
            foot_idx = int(0.075 * self.fs)
            return foot_idx, float(foot_idx / self.fs * 1000)
        
        # === Method 1: Second derivative maximum (in search region) ===
        # This detects the point of maximum acceleration (start of upstroke)
        d1 = np.diff(smoothed)
        d2 = np.diff(d1)
        
        # Second derivative in the search region
        d2_start = max(0, foot_search_start - 1)
        d2_end = min(len(d2), foot_search_end - 1)
        
        foot_d2 = 0
        if d2_end > d2_start:
            d2_search = d2[d2_start:d2_end]
            if len(d2_search) > 0:
                # Find the maximum second derivative (acceleration peak)
                max_d2_idx = int(np.argmax(d2_search))
                foot_d2 = foot_search_start + max_d2_idx
            
        # === Method 2: Intersecting tangent ===
        d_search = d1[foot_search_start:foot_search_end] if foot_search_end < len(d1) else d1[foot_search_start:]
        
        if len(d_search) > 0:
            max_slope_local_idx = np.argmax(d_search)
            max_slope_idx = foot_search_start + max_slope_local_idx
            max_slope = d_search[max_slope_local_idx]
            
            if max_slope > 0.01:  # Need meaningful slope
                ppg_at_slope = smoothed[max_slope_idx]
                min_val = np.min(foot_search_region[:max_slope_local_idx]) if max_slope_local_idx > 0 else 0
                
                # Intersecting tangent calculation
                foot_tangent = max_slope_idx - (ppg_at_slope - min_val) / max_slope
                foot_tangent = int(max(foot_search_start, min(foot_tangent, foot_search_end - 1)))
            else:
                foot_tangent = 0
        else:
            foot_tangent = 0
            
        # === Method 3: Threshold crossing (10% of amplitude) ===
        threshold = 0.1
        above_threshold = np.where(foot_search_region > threshold)[0]
        if len(above_threshold) > 0:
            foot_threshold = foot_search_start + int(above_threshold[0])
        else:
            foot_threshold = 0
            
        # === Method 4: Minimum in search region ===
        foot_min = foot_search_start + int(np.argmin(foot_search_region))
        
        # === Method 5: Percentage of rise (find where signal reaches 5% of rise to peak) ===
        min_val_in_region = np.min(foot_search_region)
        max_val_in_region = np.max(foot_search_region)
        rise_threshold = min_val_in_region + 0.05 * (max_val_in_region - min_val_in_region)
        above_rise = np.where(foot_search_region > rise_threshold)[0]
        if len(above_rise) > 0:
            foot_rise = foot_search_start + int(above_rise[0])
        else:
            foot_rise = 0
        
        # === Consensus: Use median of valid estimates ===
        candidates = []
        
        # Physiological range for PPG foot: 25-200ms from R-peak (12-100 samples at 500Hz)
        min_foot_samples = int(0.025 * self.fs)  # 25ms
        max_foot_samples = int(0.200 * self.fs)  # 200ms
        
        for foot_est in [foot_d2, foot_tangent, foot_threshold, foot_min, foot_rise]:
            # Must be in physiological range and before peak
            if min_foot_samples <= foot_est <= min(max_foot_samples, peak_idx - 1):
                candidates.append(foot_est)
        
        if len(candidates) == 0:
            # Fallback: use minimum point in search region
            if foot_search_start < foot_min < foot_search_end:
                foot_idx = foot_min
            else:
                # Last resort: estimate at 75ms (typical foot location)
                foot_idx = int(0.075 * self.fs)
        elif len(candidates) == 1:
            foot_idx = candidates[0]
        else:
            # Use median of candidates
            foot_idx = int(np.median(candidates))
        
        # BOUNDARY CHECK: If foot_idx is at the minimum allowed boundary,
        # and the signal has meaningful slope, use the 5% rise method
        boundary_threshold = min_foot_samples + 3  # 28ms
        if foot_idx <= boundary_threshold and foot_rise > boundary_threshold:
            foot_idx = foot_rise
        
        foot_time_ms = float(foot_idx / self.fs * 1000)
        
        return foot_idx, foot_time_ms
    
    def find_systolic_peak(self, ppg_segment: np.ndarray,
                          min_prominence: float = 0.05) -> Tuple[int, float]:
        """
        Find the systolic peak in PPG waveform.
        
        IMPORTANT: PPG cycles extracted R-peak to R-peak have initial artifact/noise
        from the ECG. The systolic peak occurs ~100-400ms after the R-peak,
        corresponding to the pulse wave arriving at the finger.
        
        Args:
            ppg_segment: PPG waveform (should be normalized 0-1)
            min_prominence: Minimum prominence for peak detection
            
        Returns:
            Tuple of (peak_index, peak_value)
        """
        # Detect PPG polarity: In some systems, PPG is inverted
        # (minimum = systolic peak due to maximum blood volume absorption)
        # Check if the signal decreases in the first half (inverted) or increases (normal)
        mid_point = len(ppg_segment) // 2
        first_quarter = np.mean(ppg_segment[:len(ppg_segment)//4])
        second_quarter = np.mean(ppg_segment[len(ppg_segment)//4:mid_point])
        
        # If signal decreases from start to middle, it's likely inverted
        is_inverted = first_quarter > second_quarter
        
        # Work with the signal in "normal" orientation (peak = maximum)
        if is_inverted:
            ppg_work = -ppg_segment  # Invert to find minimum as "peak"
        else:
            ppg_work = ppg_segment
        
        # Normalize
        ppg_min = np.min(ppg_work)
        ppg_max = np.max(ppg_work)
        if ppg_max - ppg_min > 1e-6:
            ppg_norm = (ppg_work - ppg_min) / (ppg_max - ppg_min)
        else:
            return len(ppg_segment) // 3, 0.5  # Safe default
        
        # Skip the initial ECG artifact region (first 50ms at 500Hz = 25 samples)
        # The PPG systolic peak physiologically occurs at ~100-400ms after R-peak
        min_peak_delay_ms = 50
        min_search_start = int(min_peak_delay_ms / 1000 * self.fs)
        
        if len(ppg_norm) < min_search_start + 20:
            peak_idx = int(np.argmax(ppg_norm))
            # Return value from normalized signal, not original (handles inverted signals)
            return peak_idx, float(ppg_norm[peak_idx])
        
        # Search region: from 50ms to 600ms (or 70% of cycle)
        max_search_end = int(min(len(ppg_norm) * 0.7, 0.6 * self.fs))
        max_search_end = max(max_search_end, min_search_start + 50)
        
        search_region = ppg_norm[min_search_start:max_search_end]
        
        if len(search_region) < 10:
            peak_idx = int(np.argmax(ppg_norm))
            return peak_idx, float(ppg_norm[peak_idx])
        
        # Find peaks with reduced prominence requirement
        peaks, properties = sig.find_peaks(search_region, prominence=min_prominence * 0.5)
        
        if len(peaks) == 0:
            local_peak_idx = int(np.argmax(search_region))
            peak_idx = min_search_start + local_peak_idx
            return peak_idx, float(ppg_norm[peak_idx])
        
        # Take the most prominent peak
        most_prominent_idx = np.argmax(properties['prominences'])
        local_peak_idx = int(peaks[most_prominent_idx])
        peak_idx = min_search_start + local_peak_idx
        
        # Return value from normalized signal (handles inverted signals correctly)
        return peak_idx, float(ppg_norm[peak_idx])
    
    def find_dicrotic_notch(self, ppg_segment: np.ndarray,
                           systolic_peak_idx: int) -> Optional[int]:
        """
        Find dicrotic notch using multiple methods.
        
        The dicrotic notch is the inflection point between the systolic
        peak and diastolic peak, representing aortic valve closure.
        
        Args:
            ppg_segment: PPG waveform (normalized 0-1)
            systolic_peak_idx: Index of systolic peak
            
        Returns:
            Index of dicrotic notch, or None if not found
        """
        # Search region: from systolic peak to ~350ms after (or 80% of remaining cycle)
        remaining_samples = len(ppg_segment) - systolic_peak_idx
        max_notch_delay_samples = min(int(0.35 * self.fs), int(remaining_samples * 0.8))
        search_end = min(systolic_peak_idx + max_notch_delay_samples, len(ppg_segment) - 2)
        
        if search_end <= systolic_peak_idx + 10:
            return None
        
        # Get the post-peak segment
        post_peak = ppg_segment[systolic_peak_idx:search_end]
        
        if len(post_peak) < 15:
            return None
        
        # Smooth to reduce noise
        smoothed = gaussian_filter1d(post_peak, sigma=2)
        
        # === Method 1: Second derivative zero-crossing (inflection point) ===
        d1 = np.diff(smoothed)
        d2 = np.diff(d1)
        
        notch_candidates = []
        
        # Find sign changes in second derivative (concave to convex transition)
        sign_changes = np.where(np.diff(np.sign(d2)))[0]
        if len(sign_changes) > 0:
            # First inflection point after peak is usually the notch
            notch_d2 = systolic_peak_idx + sign_changes[0] + 1
            notch_candidates.append(notch_d2)
        
        # === Method 2: Local minimum (valley) ===
        valleys, properties = sig.find_peaks(-smoothed, prominence=0.01)
        if len(valleys) > 0:
            # Take the first valley (closest to systolic peak)
            notch_valley = systolic_peak_idx + valleys[0]
            notch_candidates.append(notch_valley)
        
        # === Method 3: Maximum negative slope (steepest descent after peak) ===
        if len(d1) > 5:
            # Find where slope is most negative (steepest descent)
            min_slope_idx = np.argmin(d1)
            # Notch is typically shortly after the steepest descent
            notch_slope = systolic_peak_idx + min(min_slope_idx + 5, len(d1) - 1)
            # Only use if it's in a reasonable range (20-150ms after peak)
            if 10 < (notch_slope - systolic_peak_idx) < int(0.15 * self.fs):
                notch_candidates.append(notch_slope)
        
        # === Method 4: Percentage drop from peak ===
        # Notch typically occurs when signal drops to 70-90% of peak amplitude
        peak_val = smoothed[0]  # Since post_peak starts at systolic peak
        end_val = smoothed[-1]
        target_drop = peak_val - 0.25 * (peak_val - end_val)  # 25% drop
        
        below_target = np.where(smoothed < target_drop)[0]
        if len(below_target) > 0 and below_target[0] > 3:
            notch_drop = systolic_peak_idx + below_target[0]
            notch_candidates.append(notch_drop)
        
        # === Consensus: use median of valid candidates ===
        if len(notch_candidates) == 0:
            return None
        elif len(notch_candidates) == 1:
            return notch_candidates[0]
        else:
            # Use median for robustness
            return int(np.median(notch_candidates))
    
    def find_max_slope(self, ppg_segment: np.ndarray,
                      start_idx: int = 0,
                      end_idx: Optional[int] = None) -> Tuple[int, float]:
        """
        Find the point of maximum upslope (max dP/dt).
        
        Args:
            ppg_segment: PPG waveform
            start_idx: Start of search region
            end_idx: End of search region (defaults to len/2)
            
        Returns:
            Tuple of (index, slope_value)
        """
        if end_idx is None:
            end_idx = len(ppg_segment) // 2
        
        search_region = ppg_segment[start_idx:end_idx]
        
        if len(search_region) < 3:
            return start_idx, 0.0
        
        # First derivative
        d_ppg = np.diff(search_region) * self.fs  # Scale to proper units
        
        max_slope_rel_idx = int(np.argmax(d_ppg))
        max_slope_idx = int(start_idx + max_slope_rel_idx)
        max_slope_value = float(d_ppg[max_slope_rel_idx])
        
        return max_slope_idx, max_slope_value
    
    def extract_features(self, ppg_cycle: np.ndarray, 
                        hr_bpm: float,
                        r_peak_sample: int = 0,
                        cycle_start_sample: int = 0,
                        validate_quality: bool = True) -> Dict[str, float]:
        """
        Extract comprehensive features from a single PPG cardiac cycle.
        
        Args:
            ppg_cycle: PPG waveform for one cardiac cycle (R-peak to R-peak)
            hr_bpm: Heart rate in beats per minute
            r_peak_sample: Sample index of R-peak in original signal (for true PTT)
            cycle_start_sample: Start sample of this cycle in original signal
            validate_quality: If True, perform signal quality checks before extraction
            
        Returns:
            Dictionary of extracted features, or empty dict if quality check fails
        """
        features = {}
        
        # Signal quality check (NEW)
        if validate_quality:
            is_valid, reason = signal_quality_check(ppg_cycle)
            if not is_valid:
                log_extraction_failure(reason)
                return {}
        
        # Normalize PPG to [0, 1]
        ppg_min = np.min(ppg_cycle)
        ppg_max = np.max(ppg_cycle)
        
        if ppg_max - ppg_min < 1e-6:
            # Invalid cycle (flat signal)
            log_extraction_failure('flat_signal')
            return {}
        
        ppg_norm = (ppg_cycle - ppg_min) / (ppg_max - ppg_min)
        
        # ===================
        # 1. PTT Features (FIXED)
        # ===================
        
        # Find systolic peak first (needed for foot search constraint)
        peak_idx, peak_val = self.find_systolic_peak(ppg_norm)
        
        # Find PPG foot (onset) - this is the key fix
        # Constraint: Foot must be BEFORE the peak
        # We pass the peak_idx as a hint or constraint if possible, 
        # but find_ppg_foot currently takes max_search_ms.
        # Let's limit max_search_ms to the peak time.
        peak_time_ms = peak_idx / self.fs * 1000
        foot_search_window = min(400, peak_time_ms) if peak_time_ms > 50 else 400
        
        foot_idx, ptt_foot_ms = self.find_ppg_foot(ppg_norm, max_search_ms=foot_search_window)
        
        # Fallback if foot detection fails (returns 0 or very small)
        if ptt_foot_ms < 10.0:  # Less than 10ms is almost certainly wrong
            # Use minimum point in first portion of cycle, but after initial transient
            # Skip first 10 samples (20ms) to avoid R-peak artifact
            search_start = min(10, len(ppg_norm) // 10)
            search_end = max(search_start + 10, min(int(len(ppg_norm) * 0.4), peak_idx))
            
            if search_end > search_start:
                foot_idx = search_start + int(np.argmin(ppg_norm[search_start:search_end]))
                ptt_foot_ms = float(foot_idx / self.fs * 1000)
            else:
                # Last resort: estimate foot at 1/4 of time to peak
                foot_idx = max(5, peak_idx // 4)
                ptt_foot_ms = float(foot_idx / self.fs * 1000)
            
            # Validate: foot should be between 20ms and 200ms for normal physiology
            if ptt_foot_ms < 20.0:
                foot_idx = int(0.020 * self.fs)  # Set to 20ms minimum
                ptt_foot_ms = 20.0
            elif ptt_foot_ms > 200.0:
                foot_idx = int(0.100 * self.fs)  # Set to 100ms (reasonable default)
                ptt_foot_ms = 100.0
        
        features['ptt_peak_to_foot'] = ptt_foot_ms
        
        # ===================
        # TRUE ECG-PPG PTT (Pulse Arrival Time)
        # ===================
        # This is the physiologically meaningful measure: 
        # Time from ECG R-peak to PPG foot (pulse arrival)
        # The cycle starts at R-peak, so foot_idx IS the PAT in samples
        pat_ms = foot_idx / self.fs * 1000
        features['pat_ecg_ppg'] = pat_ms  # Pulse Arrival Time
        
        # PAT to peak (R-peak to PPG systolic peak)
        pat_peak_ms = peak_idx / self.fs * 1000
        features['pat_to_peak'] = pat_peak_ms
        
        # Find max slope
        max_slope_idx, max_slope_val = self.find_max_slope(ppg_norm, 0, peak_idx + 1)
        
        # PAT to max slope
        pat_maxslope_ms = max_slope_idx / self.fs * 1000
        features['pat_to_maxslope'] = pat_maxslope_ms
        
        # Legacy features (for compatibility)
        ptt_peak_ms = peak_idx / self.fs * 1000
        features['ptt_peak_to_peak'] = ptt_peak_ms
        ptt_maxslope_ms = max_slope_idx / self.fs * 1000
        features['ptt_peak_to_maxslope'] = ptt_maxslope_ms
        
        # Time from foot to max slope
        t1_ms = (max_slope_idx - foot_idx) / self.fs * 1000
        features['time_to_maxslope_t1'] = max(0, t1_ms)
        
        # ===================
        # 2. Morphology Features (FIXED)
        # ===================
        
        # Find dicrotic notch
        notch_idx = self.find_dicrotic_notch(ppg_norm, peak_idx)
        
        # Use values directly from ppg_norm for consistent calculation
        # (peak_val from find_systolic_peak may use different normalization)
        actual_peak_val = ppg_norm[peak_idx]
        actual_foot_val = ppg_norm[foot_idx] if 0 <= foot_idx < len(ppg_norm) else ppg_norm[0]
        
        if notch_idx is not None and 0 < notch_idx < len(ppg_norm):
            notch_val = ppg_norm[notch_idx]
            
            # Amplitude ratio (Ra) - FIXED calculation
            # Ra = (peak - notch) / (peak - foot)
            # Handle both normal and inverted signals
            denominator = abs(actual_peak_val - actual_foot_val)
            if denominator > 0.01:  # Avoid division by very small numbers
                # For normal signals: peak > foot, notch between them
                # For inverted signals: after our foot/peak detection, peak_val should be higher
                ra = abs(actual_peak_val - notch_val) / denominator
                ra = np.clip(ra, 0, 1)  # Ra should be in [0, 1]
            else:
                ra = 0.5  # Default if waveform is too flat
            
            features['amplitude_ratio_ra'] = ra
            
            # Systolic duration (foot to notch)
            tsd_ms = (notch_idx - foot_idx) / self.fs * 1000
            features['systolic_duration_tsd'] = max(0, tsd_ms)
            
            # Diastolic duration (notch to end)
            tfd_ms = (len(ppg_norm) - notch_idx) / self.fs * 1000
            features['diastolic_duration_tfd'] = max(0, tfd_ms)
            
        else:
            # Fallback: estimate amplitude ratio from waveform shape
            # Even without clear notch, we can estimate Ra from the decay pattern
            cycle_duration_ms = len(ppg_norm) / self.fs * 1000
            
            # The notch typically occurs 50-150ms AFTER the systolic peak
            # Estimate notch position relative to peak, not absolute cycle position
            notch_delay_samples = int(0.08 * self.fs)  # ~80ms after peak
            estimated_notch_pos = peak_idx + notch_delay_samples
            
            if 0 < peak_idx < estimated_notch_pos < len(ppg_norm) - 5:
                # Estimate Ra from signal at estimated notch position
                estimated_notch_val = ppg_norm[estimated_notch_pos]
                foot_val = ppg_norm[foot_idx] if 0 <= foot_idx < len(ppg_norm) else ppg_norm[0]
                
                denominator = peak_val - foot_val
                if denominator > 0.01:
                    ra = (peak_val - estimated_notch_val) / denominator
                    ra = np.clip(ra, 0.1, 0.9)  # Constrain to reasonable range
                else:
                    ra = 0.5
            else:
                # Alternative: use the minimum in the post-peak region (30-60% of remaining cycle)
                remaining_start = peak_idx + int(0.02 * self.fs)  # 20ms after peak
                remaining_end = min(peak_idx + int(0.2 * self.fs), len(ppg_norm) - 1)  # up to 200ms after
                
                if remaining_end > remaining_start + 5:
                    post_peak_region = ppg_norm[remaining_start:remaining_end]
                    local_min_idx = remaining_start + int(np.argmin(post_peak_region))
                    estimated_notch_val = ppg_norm[local_min_idx]
                    foot_val = ppg_norm[foot_idx] if 0 <= foot_idx < len(ppg_norm) else ppg_norm[0]
                    
                    denominator = peak_val - foot_val
                    if denominator > 0.01:
                        ra = (peak_val - estimated_notch_val) / denominator
                        ra = np.clip(ra, 0.1, 0.9)
                    else:
                        ra = 0.5
                else:
                    ra = 0.5
            
            features['amplitude_ratio_ra'] = ra
            features['systolic_duration_tsd'] = cycle_duration_ms * 0.35
            features['diastolic_duration_tfd'] = cycle_duration_ms * 0.65
        
        # ===================
        # 3. NEW Predictive Features
        # ===================
        
        # Heart rate
        features['hr_bpm'] = hr_bpm
        
        # First derivative features (velocity)
        d_ppg = np.diff(ppg_norm) * self.fs
        features['max_upslope'] = np.max(d_ppg)
        features['max_downslope'] = np.min(d_ppg)
        features['slope_ratio'] = abs(features['max_downslope'] / (features['max_upslope'] + 1e-6))
        
        # Second derivative features (acceleration)
        d2_ppg = np.diff(d_ppg) * self.fs
        features['max_acceleration'] = np.max(d2_ppg)
        features['min_acceleration'] = np.min(d2_ppg)
        
        # Area features
        systolic_area = np.trapezoid(ppg_norm[:peak_idx]) if peak_idx > 0 else 0
        diastolic_area = np.trapezoid(ppg_norm[peak_idx:]) if peak_idx < len(ppg_norm) else 0
        total_area = systolic_area + diastolic_area
        
        if total_area > 0:
            features['systolic_area_ratio'] = systolic_area / total_area
        else:
            features['systolic_area_ratio'] = 0.5
        
        # Pulse width at different amplitudes
        half_amplitude = 0.5
        above_half = ppg_norm > half_amplitude
        if np.any(above_half):
            transitions = np.diff(above_half.astype(int))
            rises = np.where(transitions == 1)[0]
            falls = np.where(transitions == -1)[0]
            
            if len(rises) > 0 and len(falls) > 0:
                if rises[0] < falls[0]:
                    pulse_width_samples = falls[0] - rises[0]
                    features['pulse_width_50'] = pulse_width_samples / self.fs * 1000
                else:
                    features['pulse_width_50'] = 0
            else:
                features['pulse_width_50'] = 0
        else:
            features['pulse_width_50'] = 0
        
        # Crest time (time to peak)
        features['crest_time'] = peak_idx / self.fs * 1000
        
        # Reflection index (if diastolic peak exists)
        if notch_idx is not None:
            post_notch = ppg_norm[notch_idx:]
            if len(post_notch) > 5:
                diastolic_peaks, _ = sig.find_peaks(post_notch)
                if len(diastolic_peaks) > 0:
                    diastolic_peak_val = post_notch[diastolic_peaks[0]]
                    features['reflection_index'] = diastolic_peak_val / peak_val if peak_val > 0 else 0
                else:
                    features['reflection_index'] = 0
            else:
                features['reflection_index'] = 0
        else:
            features['reflection_index'] = 0
            
        # ===================
        # 4. Statistical Features (Robust Fallback)
        # ===================
        # Added from old pipeline for robustness against noise
        features['stat_mean'] = float(np.mean(ppg_cycle))
        features['stat_std'] = float(np.std(ppg_cycle))
        features['stat_skew'] = float(skew(ppg_cycle))
        features['stat_kurtosis'] = float(kurtosis(ppg_cycle))
        
        # Frequency domain features
        if len(ppg_cycle) > 10:
            try:
                freqs, psd = sig.welch(ppg_cycle, fs=self.fs, nperseg=min(len(ppg_cycle), 256))
                # Cardiac band (0.5-4 Hz) vs Respiratory/Low freq (<0.5 Hz)
                idx_cardiac = np.logical_and(freqs >= 0.5, freqs <= 4.0)
                idx_low = np.logical_and(freqs >= 0.0, freqs < 0.5)
                
                power_cardiac = np.sum(psd[idx_cardiac])
                power_low = np.sum(psd[idx_low])
                
                features['stat_power_cardiac'] = float(power_cardiac)
                features['stat_power_low'] = float(power_low)
                features['stat_power_ratio'] = float(power_cardiac / (power_low + 1e-9))
            except Exception:
                features['stat_power_cardiac'] = 0.0
                features['stat_power_low'] = 0.0
                features['stat_power_ratio'] = 0.0
        else:
            features['stat_power_cardiac'] = 0.0
            features['stat_power_low'] = 0.0
            features['stat_power_ratio'] = 0.0
        
        return features


def validate_features(features: Dict[str, float], strict: bool = False) -> bool:
    """
    Validate extracted features are within physiological bounds.
    
    Args:
        features: Dictionary of extracted features
        strict: If True, apply stricter physiological bounds
        
    Returns:
        True if all features are valid
    """
    if not features:
        return False
    
    # Must have minimum required features (including new PAT)
    required_features = ['pat_ecg_ppg', 'ptt_peak_to_foot', 'amplitude_ratio_ra', 'hr_bpm']
    for key in required_features:
        if key not in features:
            logger.debug(f"Missing required feature: {key}")
            log_extraction_failure('nan_features')
            return False
    
    # Check all features for NaN/inf
    for key, val in features.items():
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                logger.debug(f"Feature {key} is NaN or inf")
                log_extraction_failure('nan_features')
                return False
    
    # Physiological validation thresholds
    if strict:
        # Strict thresholds - clinically validated ranges
        pat_range = (50, 500)  # ms - R-peak to PPG foot
        ptt_range = (50, 400)  # ms - within PPG cycle
        hr_range = (30, 200)   # BPM
        amp_range = (0.05, 0.95)  # amplitude ratio
    else:
        # Permissive thresholds - only reject clearly broken data
        pat_range = (1, 1000)
        ptt_range = (1, 2000)
        hr_range = (10, 300)
        amp_range = (0.001, 1.5)  # Allow slight overshoot
    
    # PAT validation
    pat = features.get('pat_ecg_ppg', 0)
    if not (pat_range[0] <= pat <= pat_range[1]):
        logger.debug(f"PAT out of range: {pat} (allowed: {pat_range})")
        log_extraction_failure('invalid_ptt')
        return False
    
    # PTT validation
    ptt = features.get('ptt_peak_to_foot', 0)
    if not (ptt_range[0] <= ptt <= ptt_range[1]):
        logger.debug(f"PTT out of range: {ptt} (allowed: {ptt_range})")
        log_extraction_failure('invalid_ptt')
        return False
    
    # HR validation
    hr = features.get('hr_bpm', 0)
    if not (hr_range[0] <= hr <= hr_range[1]):
        logger.debug(f"HR out of range: {hr} (allowed: {hr_range})")
        return False
    
    # Amplitude ratio validation
    amp = features.get('amplitude_ratio_ra', 0)
    if not (amp_range[0] <= amp <= amp_range[1]):
        logger.debug(f"Amplitude ratio out of range: {amp} (allowed: {amp_range})")
        return False
    
    # Additional: amplitude ratio = exactly 1.0 is suspicious (default value)
    if strict and amp == 1.0:
        logger.debug("Amplitude ratio exactly 1.0 (suspicious default)")
        return False
    
    return True


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate synthetic PPG waveform for testing
    fs = 500
    duration_s = 0.8  # ~75 bpm
    t = np.linspace(0, duration_s, int(duration_s * fs))
    
    # Simplified PPG model: systolic + diastolic components
    systolic = 0.7 * np.exp(-((t - 0.15) ** 2) / (2 * 0.03 ** 2))
    diastolic = 0.3 * np.exp(-((t - 0.4) ** 2) / (2 * 0.08 ** 2))
    ppg = systolic + diastolic + 0.02 * np.random.randn(len(t))
    
    # Extract features
    extractor = FixedFeatureExtractor(fs=fs)
    features = extractor.extract_features(ppg, hr_bpm=75)
    
    print("Extracted Features:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.4f}")
    
    print(f"\nFeatures valid: {validate_features(features)}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # PPG waveform
    t_ms = t * 1000
    axes[0].plot(t_ms, ppg, 'b-', label='PPG')
    
    # Mark detected points
    foot_idx = int(features['ptt_peak_to_foot'] / 1000 * fs)
    peak_idx = int(features['ptt_peak_to_peak'] / 1000 * fs)
    
    axes[0].axvline(features['ptt_peak_to_foot'], color='g', linestyle='--', label='Foot')
    axes[0].axvline(features['ptt_peak_to_peak'], color='r', linestyle='--', label='Peak')
    axes[0].axvline(features['ptt_peak_to_maxslope'], color='orange', linestyle='--', label='Max Slope')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('PPG Waveform with Detected Fiducial Points')
    axes[0].legend()
    
    # First derivative
    d_ppg = np.diff(ppg) * fs
    axes[1].plot(t_ms[:-1], d_ppg, 'g-')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('dP/dt')
    axes[1].set_title('First Derivative (Velocity)')
    axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # Second derivative
    d2_ppg = np.diff(d_ppg) * fs
    axes[2].plot(t_ms[:-2], d2_ppg, 'r-')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('d²P/dt²')
    axes[2].set_title('Second Derivative (Acceleration)')
    axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save to dataset/data/figures/
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'data', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, 'feature_extraction_demo.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
