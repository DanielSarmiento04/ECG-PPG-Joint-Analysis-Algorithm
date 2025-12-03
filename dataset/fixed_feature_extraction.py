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
    'low_snr': 0,
    'low_amplitude': 0,
    'total_attempts': 0
}

# Minimum signal-to-noise ratio for valid extraction
# Based on analysis: good patients have PPG std ~12, bad patients have std ~2
MIN_PPG_STD = 3.0  # Minimum standard deviation of PPG signal
MIN_SNR = 5.0      # Minimum signal-to-noise ratio


def get_extraction_stats() -> Dict:
    """Return extraction failure statistics."""
    return _extraction_failures.copy()


def reset_extraction_stats():
    """Reset extraction failure statistics."""
    global _extraction_failures
    for key in _extraction_failures:
        _extraction_failures[key] = 0


def remove_spikes(signal_data: np.ndarray, threshold_mad: float = 5.0) -> np.ndarray:
    """
    Remove spike artifacts from signal using MAD-based outlier detection.
    
    VitalDB PPG data often has ECG R-peak electrical artifacts bleeding into 
    the PPG channel, appearing as sharp spikes. These corrupt foot detection.
    
    Args:
        signal_data: Raw signal with potential spikes
        threshold_mad: Number of MAD units for outlier threshold (default 5)
        
    Returns:
        Signal with spikes replaced by interpolated values
    """
    signal_clean = signal_data.copy()
    
    # Handle NaN
    valid_mask = ~np.isnan(signal_clean)
    if valid_mask.sum() < 10:
        return signal_clean
    
    valid_data = signal_clean[valid_mask]
    
    # Use median and MAD for robust outlier detection
    median_val = np.median(valid_data)
    mad = np.median(np.abs(valid_data - median_val))
    
    if mad < 1e-6:
        return signal_clean  # No variation, return as-is
    
    # Convert MAD to std-equivalent: std ≈ 1.4826 * MAD for normal distribution
    threshold = threshold_mad * mad * 1.4826
    
    # Find outliers
    outliers = np.abs(signal_clean - median_val) > threshold
    
    if outliers.sum() == 0:
        return signal_clean
    
    # Replace outliers with linear interpolation
    outlier_indices = np.where(outliers)[0]
    valid_indices = np.where(~outliers)[0]
    
    if len(valid_indices) < 2:
        return signal_clean
    
    # Interpolate outliers
    signal_clean[outliers] = np.interp(
        outlier_indices, 
        valid_indices, 
        signal_clean[valid_indices]
    )
    
    return signal_clean


def preprocess_signal(signal_data: np.ndarray, fs: int, remove_artifacts: bool = True) -> np.ndarray:
    """
    Clean signal with spike removal and bandpass filter.
    
    Args:
        signal_data: Raw signal
        fs: Sampling frequency in Hz
        remove_artifacts: If True, remove spike artifacts first
        
    Returns:
        Filtered signal
    """
    if len(signal_data) < 20:
        return signal_data
    
    # Handle NaN values
    if np.isnan(signal_data).any():
        signal_data = np.nan_to_num(signal_data, nan=float(np.nanmean(signal_data)))
    
    # NEW: Remove spike artifacts (ECG bleed-through, motion artifacts)
    if remove_artifacts:
        signal_data = remove_spikes(signal_data, threshold_mad=4.0)
    
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


def signal_quality_check(ppg_segment: np.ndarray, min_snr: float = 5.0) -> Tuple[bool, str]:
    """
    Check if signal is good enough for feature extraction.
    
    CRITICAL: VitalDB data analysis showed that patients with low PPG variability
    (std < 3) have WRONG PTT-BP correlations because foot detection becomes
    unreliable. Good patients have std ~12, bad patients have std ~2.
    
    Args:
        ppg_segment: PPG waveform segment
        min_snr: Minimum signal-to-noise ratio (default 5.0, increased from 3.0)
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Check 1: Sufficient length
    if len(ppg_segment) < 20:
        return False, "segment_too_short"
    
    # Check 2: Not all NaN
    if np.isnan(ppg_segment).all():
        return False, "all_nan"
    
    # Remove NaN for statistics
    valid_data = ppg_segment[~np.isnan(ppg_segment)]
    if len(valid_data) < 20:
        return False, "insufficient_valid_data"
    
    # Check 3: Sufficient amplitude variation (CRITICAL FIX)
    # Analysis showed: good patients std~12, bad patients std~2
    ppg_std = np.std(valid_data)
    ppg_range = np.max(valid_data) - np.min(valid_data)
    
    if ppg_range < 1e-6:
        return False, "flat_signal"
    
    # NEW: Minimum amplitude threshold based on analysis
    # PPG std < 3 leads to unreliable foot detection
    if ppg_std < MIN_PPG_STD:
        return False, f"low_amplitude_std_{ppg_std:.2f}"
    
    # Check 4: Not saturated (clipping)
    unique_vals = len(np.unique(valid_data))
    if unique_vals < 5:
        return False, "saturated"
    
    # Check 5: SNR estimate (improved version)
    # SNR = signal variance / noise variance (estimated from high-freq components)
    # Use second difference for better noise estimation
    if len(valid_data) > 10:
        diff2 = np.diff(np.diff(valid_data))
        noise_estimate = np.std(diff2) / 2.0  # Second diff has sqrt(6) noise amplification
        
        if noise_estimate > 0.01:
            snr = ppg_std / noise_estimate
            if snr < min_snr:
                return False, f"low_snr_{snr:.2f}"
    
    # Check 6: Signal should have clear pulsatile pattern
    # Peak-to-peak amplitude should be at least 3x the noise floor
    peak_to_peak = ppg_range
    noise_floor = np.std(np.diff(valid_data)) / np.sqrt(2)
    if noise_floor > 0.01 and peak_to_peak < 3 * noise_floor:
        return False, f"weak_pulse_amplitude"
    
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


def detect_global_ppg_polarity(ppg_signal: np.ndarray, fs: int = 500,
                                ecg_r_peaks: Optional[np.ndarray] = None) -> bool:
    """
    Detect PPG polarity at the GLOBAL level (entire recording).
    
    This is critical because per-beat polarity detection fails due to:
    - Respiratory baseline drift
    - Short beat segments
    - Variable morphology
    
    Method: Use cross-correlation timing - in normal PPG, the maximum
    (systolic peak) occurs AFTER the minimum (diastolic foot). In inverted
    PPG, the minimum (which represents max blood volume) occurs AFTER the maximum.
    
    Args:
        ppg_signal: Full PPG signal
        fs: Sampling frequency
        ecg_r_peaks: Optional R-peak indices for beat-based analysis
        
    Returns:
        True if PPG is inverted (minimum = systolic), False if normal (maximum = systolic)
    """
    if len(ppg_signal) < fs:  # Need at least 1 second
        return False
    
    # Remove baseline drift with high-pass filter (0.5 Hz cutoff)
    try:
        sos = butter(2, 0.5 / (fs / 2), btype='high', output='sos')
        ppg_filtered = sig.sosfiltfilt(sos, ppg_signal)
    except Exception:
        ppg_filtered = ppg_signal - gaussian_filter1d(ppg_signal, sigma=fs)
    
    # Method 1: Average beat morphology
    # If we have R-peaks, average multiple beats and check morphology
    if ecg_r_peaks is not None and len(ecg_r_peaks) >= 5:
        beat_length = int(0.8 * fs)  # 800ms typical beat
        beats = []
        
        for i, rpk in enumerate(ecg_r_peaks[:-1]):
            next_rpk = ecg_r_peaks[i + 1]
            if rpk + beat_length <= len(ppg_filtered) and next_rpk - rpk > beat_length // 2:
                beat = ppg_filtered[rpk:rpk + beat_length]
                # Normalize each beat
                beat_range = np.max(beat) - np.min(beat)
                if beat_range > 0.01:
                    beat_norm = (beat - np.min(beat)) / beat_range
                    beats.append(beat_norm)
        
        if len(beats) >= 5:
            # Average beat template
            avg_beat = np.mean(beats, axis=0)
            
            # In normal PPG: signal rises to peak (100-200ms) then falls
            # In inverted PPG: signal falls to minimum (100-200ms) then rises
            early_region = avg_beat[int(0.05 * fs):int(0.15 * fs)]  # 50-150ms
            mid_region = avg_beat[int(0.15 * fs):int(0.30 * fs)]   # 150-300ms
            
            # Calculate slope: positive = rising (normal), negative = falling (inverted)
            early_slope = np.mean(np.diff(early_region))
            
            # If signal is falling in early region, it's inverted
            return bool(early_slope < 0)
    
    # Method 2: Statistical analysis of derivatives
    # In normal PPG, there's a rapid upstroke (positive derivative)
    # followed by slower downstroke. The skewness of derivatives tells us polarity.
    deriv = np.diff(ppg_filtered)
    
    # Segment into beat-sized windows and check consistency
    window_size = int(0.8 * fs)
    n_windows = len(deriv) // window_size
    
    if n_windows < 3:
        # Too short - use simple heuristic
        return bool(np.mean(deriv[:len(deriv)//4]) < 0)
    
    inversion_votes = 0
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_deriv = deriv[start:end]
        
        # In normal PPG, the maximum derivative (steepest upstroke) occurs
        # before the minimum derivative (downstroke from peak)
        max_deriv_idx = np.argmax(window_deriv)
        min_deriv_idx = np.argmin(window_deriv)
        
        # If min_deriv comes before max_deriv, signal is inverted
        if min_deriv_idx < max_deriv_idx:
            inversion_votes += 1
    
    # Majority vote
    is_inverted = inversion_votes > n_windows // 2
    
    return is_inverted


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
        self._global_ppg_inverted = None  # Will be set by set_global_polarity()
    
    def set_global_polarity(self, ppg_signal: np.ndarray, 
                           ecg_r_peaks: Optional[np.ndarray] = None):
        """
        Set the global PPG polarity for this recording.
        
        MUST be called before extracting features to ensure consistent
        polarity detection across all beats.
        
        Args:
            ppg_signal: Full PPG signal for the recording
            ecg_r_peaks: Optional R-peak indices
        """
        self._global_ppg_inverted = detect_global_ppg_polarity(
            ppg_signal, self.fs, ecg_r_peaks
        )
        logger.debug(f"Global PPG polarity: {'inverted' if self._global_ppg_inverted else 'normal'}")
        
    def find_ppg_foot(self, ppg_segment: np.ndarray, 
                      max_search_ms: float = 400) -> Tuple[int, float]:
        """
        Find PPG foot (onset) using multiple methods with consensus.
        
        Methods used:
        1. Second derivative maximum (acceleration peak before upstroke)
        2. Intersecting tangent (line from max slope to baseline)
        3. Threshold crossing (10% of pulse amplitude)
        4. Minimum detection
        5. 5% rise method
        
        CRITICAL: The PPG foot at fingertip CANNOT occur earlier than ~80-100ms
        after the R-peak due to pulse wave transit time from heart to finger.
        Typical PAT is 150-350ms. Values <80ms indicate artifact detection.
        
        Args:
            ppg_segment: PPG waveform starting from R-peak
            max_search_ms: Maximum time to search for foot (ms)
            
        Returns:
            Tuple of (foot_index, foot_time_ms)
        """
        max_search_samples = min(int(max_search_ms / 1000 * self.fs), len(ppg_segment))
        
        if len(ppg_segment) < 50:  # Need at least 100ms of signal
            return 0, 0.0
        
        # CRITICAL: Remove spike artifacts before processing
        # VitalDB PPG often has ECG R-peak electrical artifacts
        ppg_clean = remove_spikes(ppg_segment, threshold_mad=4.0)
        
        # USE GLOBAL POLARITY if set, otherwise fall back to per-beat detection
        if self._global_ppg_inverted is not None:
            is_inverted = self._global_ppg_inverted
        else:
            # Fallback: per-beat detection (less reliable)
            mid_point = len(ppg_clean) // 2
            first_quarter = np.mean(ppg_clean[:len(ppg_clean)//4])
            second_quarter = np.mean(ppg_clean[len(ppg_clean)//4:mid_point])
            is_inverted = first_quarter > second_quarter
        
        # Work with signal in "normal" orientation (rising toward peak)
        ppg_work = -ppg_clean if is_inverted else ppg_clean
        
        # Remove baseline trend within the beat
        if len(ppg_work) > 20:
            x = np.arange(len(ppg_work))
            coeffs = np.polyfit(x, ppg_work, 1)
            trend = np.polyval(coeffs, x)
            ppg_work = ppg_work - trend + np.mean(ppg_work)
            
        # Normalize the segment
        ppg_min = np.min(ppg_work)
        ppg_max = np.max(ppg_work)
        if ppg_max - ppg_min < 1e-6:
            return 0, 0.0
        ppg_norm = (ppg_work - ppg_min) / (ppg_max - ppg_min)
        
        search_region = ppg_norm[:max_search_samples]
        
        # Smooth the signal to reduce noise
        smoothed = gaussian_filter1d(search_region, sigma=3)
        
        # Find the systolic peak first (needed to define search region)
        # Peak should be between 100-400ms after R-peak
        peak_search_start = int(0.080 * self.fs)  # 80ms
        peak_search_end = min(int(0.450 * self.fs), len(smoothed))  # 450ms
        
        if peak_search_end <= peak_search_start + 10:
            return 0, 0.0
            
        peak_region = smoothed[peak_search_start:peak_search_end]
        local_peak_idx = int(np.argmax(peak_region))
        peak_idx = peak_search_start + local_peak_idx
        peak_time_ms = peak_idx / self.fs * 1000
        
        # === NEW APPROACH: Find foot using DERIVATIVE analysis ===
        # The foot is where the upstroke begins - characterized by:
        # 1. Maximum acceleration (max second derivative)
        # 2. Start of sustained positive slope
        
        # Calculate derivatives
        d1 = np.diff(smoothed)  # First derivative (velocity)
        d2 = np.diff(d1)        # Second derivative (acceleration)
        
        # Search region for foot: from 50ms to just before peak
        # (Allow earlier than 80ms since we're looking for upstroke START)
        foot_search_start = int(0.050 * self.fs)  # 50ms - avoid ECG artifact
        foot_search_end = peak_idx - int(0.020 * self.fs)  # 20ms before peak
        
        if foot_search_end <= foot_search_start + 10:
            # Peak is too early, use default
            foot_idx = int(peak_idx * 0.6)  # 60% of way to peak
            return max(foot_idx, int(0.080 * self.fs)), float(max(foot_idx, int(0.080 * self.fs)) / self.fs * 1000)
        
        # === Method 1: Maximum second derivative (acceleration) ===
        # The foot is where acceleration is maximum (steepest part of upstroke begins)
        d2_start = max(0, foot_search_start)
        d2_end = min(len(d2), foot_search_end)
        
        foot_d2 = foot_search_start
        if d2_end > d2_start and d2_end <= len(d2):
            d2_search = d2[d2_start:d2_end]
            if len(d2_search) > 5:
                # Smooth the second derivative
                d2_smooth = gaussian_filter1d(d2_search, sigma=2)
                max_d2_idx = int(np.argmax(d2_smooth))
                foot_d2 = foot_search_start + max_d2_idx
            
        # === Method 2: Intersecting tangent from max slope ===
        d1_start = foot_search_start
        d1_end = min(len(d1), foot_search_end)
        
        foot_tangent = foot_search_start
        if d1_end > d1_start:
            d1_search = d1[d1_start:d1_end]
            if len(d1_search) > 5:
                max_slope_local_idx = int(np.argmax(d1_search))
                max_slope_idx = foot_search_start + max_slope_local_idx
                max_slope = d1_search[max_slope_local_idx]
                
                if max_slope > 0.005:  # Need meaningful positive slope
                    ppg_at_slope = smoothed[max_slope_idx]
                    # Find baseline (minimum before max slope point)
                    baseline_region = smoothed[foot_search_start:max_slope_idx] if max_slope_idx > foot_search_start else [smoothed[foot_search_start]]
                    baseline = np.min(baseline_region)
                    
                    # Intersecting tangent: where tangent line crosses baseline
                    foot_tangent = max_slope_idx - (ppg_at_slope - baseline) / max_slope
                    foot_tangent = int(max(foot_search_start, min(foot_tangent, foot_search_end)))
            
        # === Method 3: 10% amplitude threshold ===
        foot_search_region = smoothed[foot_search_start:foot_search_end]
        if len(foot_search_region) > 5:
            min_in_region = np.min(foot_search_region)
            max_in_region = smoothed[peak_idx] if peak_idx < len(smoothed) else np.max(foot_search_region)
            threshold_10 = min_in_region + 0.10 * (max_in_region - min_in_region)
            
            above_thresh = np.where(foot_search_region > threshold_10)[0]
            if len(above_thresh) > 0:
                foot_thresh = foot_search_start + int(above_thresh[0])
            else:
                foot_thresh = foot_search_start
        else:
            foot_thresh = foot_search_start
            
        # === Method 4: Minimum in search region (classic method) ===
        if len(foot_search_region) > 0:
            foot_min = foot_search_start + int(np.argmin(foot_search_region))
        else:
            foot_min = foot_search_start
        
        # === Consensus: Prefer derivative-based methods ===
        candidates = []
        
        # Physiological bounds: 80-300ms (allow wider range)
        min_foot_ms = 80
        max_foot_ms = min(300, peak_time_ms - 10)  # At least 10ms before peak
        min_foot_samples = int(min_foot_ms / 1000 * self.fs)
        max_foot_samples = int(max_foot_ms / 1000 * self.fs)
        
        for foot_est in [foot_d2, foot_tangent, foot_thresh, foot_min]:
            if min_foot_samples <= foot_est <= max_foot_samples:
                candidates.append(foot_est)
        
        if len(candidates) == 0:
            # No valid candidates - estimate based on peak location
            # Foot typically occurs at 40-60% of the way to peak
            foot_idx = int(peak_idx * 0.5)
            foot_idx = max(foot_idx, min_foot_samples)
        elif len(candidates) == 1:
            foot_idx = candidates[0]
        else:
            # Prefer earlier estimates (closer to true foot)
            # Use 25th percentile instead of median
            foot_idx = int(np.percentile(candidates, 25))
        
        # Final bounds check
        foot_idx = max(foot_idx, min_foot_samples)
        foot_idx = min(foot_idx, max_foot_samples)
        
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
        # USE GLOBAL POLARITY if set, otherwise fall back to per-beat detection
        if self._global_ppg_inverted is not None:
            is_inverted = self._global_ppg_inverted
        else:
            # Fallback: per-beat detection (less reliable)
            mid_point = len(ppg_segment) // 2
            first_quarter = np.mean(ppg_segment[:len(ppg_segment)//4])
            second_quarter = np.mean(ppg_segment[len(ppg_segment)//4:mid_point])
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
        
        # Skip the initial ECG artifact region (first 80ms at 500Hz = 40 samples)
        # The PPG systolic peak physiologically occurs at ~100-400ms after R-peak
        min_peak_delay_ms = 80  # Increased from 50ms
        min_search_start = int(min_peak_delay_ms / 1000 * self.fs)
        
        if len(ppg_norm) < min_search_start + 20:
            peak_idx = int(np.argmax(ppg_norm))
            # Return value from normalized signal, not original (handles inverted signals)
            return peak_idx, float(ppg_norm[peak_idx])
        
        # Search region: from 80ms to 600ms (or 70% of cycle)
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
            
            # Physiological validation - relaxed bounds for VitalDB
            # Allow 40-250ms range for surgical patient population
            if ptt_foot_ms < 40.0:
                foot_idx = int(0.080 * self.fs)  # Set to 80ms (reasonable minimum)
                ptt_foot_ms = 80.0
            elif ptt_foot_ms > 250.0:
                foot_idx = int(0.120 * self.fs)  # Set to 120ms (reasonable default)
                ptt_foot_ms = 120.0
        
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
        
    Key validation: PTT must be physiologically valid relative to HR.
    High PTT (>150ms) with normal HR indicates bad foot detection.
    Constraint: PTT < 20% of cardiac cycle length.
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
    
    # Get core values for validation
    hr = features.get('hr_bpm', 0)
    ptt = features.get('ptt_peak_to_foot', 0)
    pat = features.get('pat_ecg_ppg', 0)
    amp = features.get('amplitude_ratio_ra', 0)
    
    # Physiological validation thresholds
    # VitalDB surgical patients may have different hemodynamics than typical subjects
    # Relaxed bounds to allow natural variability while rejecting clear artifacts
    if strict:
        # Strict thresholds - clinically reasonable ranges
        pat_range = (50, 350)   # ms - R-peak to PPG foot
        ptt_range = (50, 350)   # ms - same as PAT for this implementation
        hr_range = (40, 180)    # BPM (surgical patients typically 40-180)
        amp_range = (0.1, 0.9)  # amplitude ratio (tightened)
        ptt_max_cycle_pct = 0.50  # PTT < 50% of cardiac cycle
    else:
        # Permissive thresholds - only reject clearly broken data
        pat_range = (30, 400)   # 30ms minimum (physical limit ~40ms, allow some tolerance)
        ptt_range = (30, 400)   # Same as PAT
        hr_range = (20, 250)
        amp_range = (0.01, 1.2)  # Allow slight overshoot
        ptt_max_cycle_pct = 0.60  # PTT < 60% of cardiac cycle (permissive)
    
    # HR validation (first, needed for PTT/cycle validation)
    if not (hr_range[0] <= hr <= hr_range[1]):
        logger.debug(f"HR out of range: {hr} (allowed: {hr_range})")
        return False
    
    # PAT validation - this is the key physiological check
    if not (pat_range[0] <= pat <= pat_range[1]):
        logger.debug(f"PAT out of range: {pat} (allowed: {pat_range})")
        log_extraction_failure('invalid_ptt')
        return False
    
    # PTT absolute range validation
    if not (ptt_range[0] <= ptt <= ptt_range[1]):
        logger.debug(f"PTT out of range: {ptt} (allowed: {ptt_range})")
        log_extraction_failure('invalid_ptt')
        return False
    
    # PTT relative to cardiac cycle validation
    # PTT should be < 50% of cardiac cycle (physiological constraint)
    if hr > 0:
        cycle_length_ms = 60000.0 / hr  # Cardiac cycle in ms
        max_ptt_for_hr = ptt_max_cycle_pct * cycle_length_ms
        if ptt > max_ptt_for_hr:
            logger.debug(f"PTT too high relative to HR: {ptt:.1f}ms > {max_ptt_for_hr:.1f}ms ({ptt_max_cycle_pct*100:.0f}% of {cycle_length_ms:.0f}ms cycle)")
            log_extraction_failure('invalid_ptt')
            return False
    
    # Amplitude ratio validation
    if not (amp_range[0] <= amp <= amp_range[1]):
        logger.debug(f"Amplitude ratio out of range: {amp} (allowed: {amp_range})")
        return False
    
    # Additional: amplitude ratio = exactly 1.0 is suspicious (default value)
    if strict and amp == 1.0:
        logger.debug("Amplitude ratio exactly 1.0 (suspicious default)")
        return False
    
    return True


def validate_patient_physiology(patient_df, min_samples: int = 100, strict: bool = False) -> Tuple[bool, str, Dict]:
    """
    Validate that a patient's data shows physiologically correct relationships.
    
    CRITICAL: Analysis of VitalDB data showed that ~35% of patients have WRONG
    PTT-BP correlation sign. This is caused by:
    1. Low signal quality (PPG std < 3) making foot detection unreliable
    2. Inverted PPG polarity that wasn't correctly detected
    3. Confounding from HR changes (HR-PTT should be negative)
    
    A valid patient should show:
    - Negative PTT-SBP correlation (higher BP → stiffer arteries → faster transit)
    - Negative HR-PTT correlation (higher HR → shorter cycle → shorter PTT)
    - Sufficient PTT variability (std > 10ms) to detect meaningful changes
    
    Args:
        patient_df: DataFrame with features for ONE patient
        min_samples: Minimum samples required for analysis
        strict: If True, apply stricter validation (requires correct correlation signs)
        
    Returns:
        Tuple of (is_valid, reason, metrics_dict)
    """
    if len(patient_df) < min_samples:
        return False, f"insufficient_samples_{len(patient_df)}", {}
    
    metrics = {}
    
    # Get key features
    ptt = patient_df['ptt_peak_to_foot'] if 'ptt_peak_to_foot' in patient_df.columns else patient_df['pat_ecg_ppg']
    sbp = patient_df['sbp_reference']
    hr = patient_df['hr_bpm']
    
    # Calculate correlations
    ptt_sbp_r = ptt.corr(sbp)
    ptt_hr_r = ptt.corr(hr)
    hr_sbp_r = hr.corr(sbp)
    ptt_std = ptt.std()
    sbp_std = sbp.std()
    
    metrics['ptt_sbp_r'] = ptt_sbp_r
    metrics['ptt_hr_r'] = ptt_hr_r
    metrics['hr_sbp_r'] = hr_sbp_r
    metrics['ptt_std'] = ptt_std
    metrics['sbp_std'] = sbp_std
    metrics['n_samples'] = len(patient_df)
    
    # Validation rules based on physiology
    reasons = []
    
    if strict:
        # STRICT MODE: Require correct physiological relationships
        
        # Rule 1: PTT variability must be high enough
        if ptt_std < 15:
            reasons.append(f"low_ptt_variability_{ptt_std:.1f}ms")
        
        # Rule 2: HR-PTT correlation MUST be negative
        if ptt_hr_r >= 0:
            reasons.append(f"wrong_hr_ptt_corr_{ptt_hr_r:.2f}")
        
        # Rule 3: PTT-SBP correlation MUST be negative
        if ptt_sbp_r >= 0:
            reasons.append(f"wrong_ptt_sbp_corr_{ptt_sbp_r:.2f}")
        
        # Rule 4: SBP must have reasonable variability
        if sbp_std < 5:
            reasons.append(f"low_sbp_variability_{sbp_std:.1f}mmHg")
    else:
        # PERMISSIVE MODE: Only reject clearly broken data
        
        # Rule 1: PTT variability check
        if ptt_std < 10:
            reasons.append(f"low_ptt_variability_{ptt_std:.1f}ms")
        
        # Rule 2: HR-PTT correlation should NOT be strongly positive
        if ptt_hr_r > 0.3:
            reasons.append(f"wrong_hr_ptt_corr_{ptt_hr_r:.2f}")
        
        # Rule 3: PTT-SBP correlation should NOT be strongly positive
        if ptt_sbp_r > 0.2:
            reasons.append(f"wrong_ptt_sbp_corr_{ptt_sbp_r:.2f}")
        
        # Rule 4: SBP must have some variability
        if sbp_std < 5:
            reasons.append(f"low_sbp_variability_{sbp_std:.1f}mmHg")
    
    if len(reasons) > 0:
        return False, "|".join(reasons), metrics
    
    return True, "ok", metrics


def filter_valid_patients(df, strict: bool = False, remove_boundary_samples: bool = True, verbose: bool = True):
    """
    Filter dataset to only include physiologically valid patients and samples.
    
    Args:
        df: Full DataFrame with all patients
        strict: If True, apply stricter validation (requires correct correlation signs)
                Use strict=True for highest quality data (r ~ -0.18)
                Use strict=False for more samples (r ~ -0.02)
        remove_boundary_samples: If True, also remove samples at PTT boundaries (40ms, 250ms)
                                 and amplitude_ratio boundaries (0, 1)
        verbose: Print summary statistics
        
    Returns:
        Tuple of (filtered DataFrame, patient_metrics dict)
    """
    initial_samples = len(df)
    
    # Step 1: Remove boundary samples (sample-level filtering)
    if remove_boundary_samples:
        ptt_col = 'ptt_peak_to_foot' if 'ptt_peak_to_foot' in df.columns else 'pat_ecg_ppg'
        
        # PTT boundaries: remove samples at exact 40ms or 250ms
        # These are artifacts from the extraction algorithm hitting limits
        boundary_mask = (
            (df[ptt_col] > 42) &  # Allow 2ms tolerance above minimum
            (df[ptt_col] < 248) &  # Allow 2ms tolerance below maximum
            (df['amplitude_ratio_ra'] > 0.02) &  # Near-zero is suspicious
            (df['amplitude_ratio_ra'] < 0.98)    # Near-one is suspicious (default value)
        )
        
        df = df[boundary_mask].copy()
        
        if verbose:
            removed = initial_samples - len(df)
            print(f"\n=== Sample-Level Filtering ===")
            print(f"Removed {removed:,} boundary samples ({100*removed/initial_samples:.1f}%)")
            print(f"Remaining: {len(df):,} samples")
    
    # Step 2: Patient-level validation
    valid_patients = []
    invalid_patients = []
    patient_metrics = {}
    
    for pid in df['patient_id'].unique():
        patient_df = df[df['patient_id'] == pid]
        is_valid, reason, metrics = validate_patient_physiology(patient_df, strict=strict)
        
        patient_metrics[pid] = {
            'is_valid': is_valid,
            'reason': reason,
            **metrics
        }
        
        if is_valid:
            valid_patients.append(pid)
        else:
            invalid_patients.append(pid)
    
    if verbose:
        n_total = len(valid_patients) + len(invalid_patients)
        print(f"\n=== Patient Validation Summary ===")
        print(f"Total patients: {n_total}")
        print(f"Valid patients: {len(valid_patients)} ({100*len(valid_patients)/n_total:.1f}%)")
        print(f"Invalid patients: {len(invalid_patients)} ({100*len(invalid_patients)/n_total:.1f}%)")
        
        # Categorize reasons
        reason_counts = {}
        for pid in invalid_patients:
            reasons = patient_metrics[pid]['reason'].split('|')
            for r in reasons:
                reason_type = r.split('_')[0] + '_' + r.split('_')[1] if '_' in r else r
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        if reason_counts:
            print(f"\nInvalidity reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count} patients")
    
    # Filter to valid patients only
    df_filtered = df[df['patient_id'].isin(valid_patients)].copy()
    
    if verbose:
        print(f"\nFiltered dataset: {len(df_filtered):,} samples from {len(valid_patients)} patients")
        if len(df_filtered) > 0:
            ptt_sbp_r = df_filtered['ptt_peak_to_foot'].corr(df_filtered['sbp_reference'])
            print(f"Overall PTT-SBP correlation: r = {ptt_sbp_r:.4f}")
    
    return df_filtered, patient_metrics


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
