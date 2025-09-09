"""
Enhanced feature extraction for physiological signals (ECG/PPG).
Addresses limitations in the original feature extraction approach.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
import pywt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Advanced signal processing for ECG and PPG signals."""
    
    def __init__(self, sampling_rate: int = 125):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def apply_filters(self, signal_data: np.ndarray, filter_type: str = 'butterworth') -> np.ndarray:
        """Apply digital filters to clean the signal."""
        
        if filter_type == 'butterworth':
            # Bandpass filter for physiological signals
            low_freq = 0.5 / self.nyquist
            high_freq = 40 / self.nyquist
            
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return filtered_signal
        else:
            return signal_data
    
    def detect_peaks(self, signal_data: np.ndarray, signal_type: str) -> Tuple[np.ndarray, Dict]:
        """Advanced peak detection with signal-specific parameters."""
        
        if signal_type.upper() == 'ECG':
            # R-peak detection parameters
            height = np.std(signal_data) * 0.3
            distance = int(0.6 * self.sampling_rate)  # Minimum 600ms between R-peaks
            prominence = height * 0.5
        else:  # PPG
            # PPG peak detection parameters
            height = np.std(signal_data) * 0.2
            distance = int(0.4 * self.sampling_rate)  # Minimum 400ms between peaks
            prominence = height * 0.3
        
        peaks, properties = signal.find_peaks(
            signal_data,
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        return peaks, properties


class TimeDomainFeatures:
    """Extract comprehensive time-domain features."""
    
    @staticmethod
    def extract_statistical_features(signal_data: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal_data)
        
        # Distribution shape
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features[f'percentile_{p}'] = np.percentile(signal_data, p)
        
        # Inter-quartile range
        features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossings'] = len(zero_crossings)
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # Root mean square
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        return features
    
    @staticmethod
    def extract_morphological_features(signal_data: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
        """Extract morphological features based on detected peaks."""
        
        features = {}
        
        if len(peaks) == 0:
            # No peaks found, return zero features
            morphological_feature_names = [
                'num_peaks', 'mean_peak_amplitude', 'std_peak_amplitude',
                'mean_peak_interval', 'std_peak_interval', 'cv_peak_interval',
                'peak_amplitude_ratio', 'peak_prominence_mean', 'peak_prominence_std'
            ]
            for name in morphological_feature_names:
                features[name] = 0.0
            return features
        
        # Peak counts and amplitudes
        features['num_peaks'] = len(peaks)
        peak_amplitudes = signal_data[peaks]
        features['mean_peak_amplitude'] = np.mean(peak_amplitudes)
        features['std_peak_amplitude'] = np.std(peak_amplitudes) if len(peaks) > 1 else 0
        
        # Peak intervals (heart rate variability)
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / 125  # Convert to seconds (assuming 125 Hz)
            features['mean_peak_interval'] = np.mean(peak_intervals)
            features['std_peak_interval'] = np.std(peak_intervals)
            features['cv_peak_interval'] = features['std_peak_interval'] / features['mean_peak_interval'] if features['mean_peak_interval'] > 0 else 0
            
            # Heart rate estimate
            features['estimated_heart_rate'] = 60 / features['mean_peak_interval'] if features['mean_peak_interval'] > 0 else 0
        else:
            features['mean_peak_interval'] = 0
            features['std_peak_interval'] = 0
            features['cv_peak_interval'] = 0
            features['estimated_heart_rate'] = 0
        
        # Peak amplitude variability
        if len(peak_amplitudes) > 1:
            features['peak_amplitude_ratio'] = np.max(peak_amplitudes) / np.min(peak_amplitudes)
        else:
            features['peak_amplitude_ratio'] = 1
        
        # Additional morphological metrics
        features['peak_density'] = len(peaks) / len(signal_data)  # Peaks per sample
        
        return features


class FrequencyDomainFeatures:
    """Extract frequency domain features using spectral analysis."""
    
    def __init__(self, sampling_rate: int = 125):
        self.sampling_rate = sampling_rate
        
    def extract_spectral_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive spectral features."""
        
        features = {}
        
        # Power spectral density using Welch's method
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)//4))
        
        # Total power
        features['total_power'] = np.sum(psd)
        
        # Power in specific frequency bands
        freq_bands = {
            'vlf': (0.0033, 0.04),  # Very low frequency
            'lf': (0.04, 0.15),     # Low frequency
            'hf': (0.15, 0.4),      # High frequency
            'vhf': (0.4, 0.5)       # Very high frequency
        }
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            if np.any(idx_band):
                features[f'power_{band_name}'] = np.sum(psd[idx_band])
                features[f'relative_power_{band_name}'] = features[f'power_{band_name}'] / features['total_power']
            else:
                features[f'power_{band_name}'] = 0
                features[f'relative_power_{band_name}'] = 0
        
        # LF/HF ratio (important for autonomic nervous system analysis)
        if features['power_hf'] > 0:
            features['lf_hf_ratio'] = features['power_lf'] / features['power_hf']
        else:
            features['lf_hf_ratio'] = 0
        
        # Spectral centroid (center of mass of spectrum)
        if features['total_power'] > 0:
            features['spectral_centroid'] = np.sum(freqs * psd) / features['total_power']
        else:
            features['spectral_centroid'] = 0
        
        # Spectral spread (spread around centroid)
        if features['total_power'] > 0:
            features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / features['total_power'])
        else:
            features['spectral_spread'] = 0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_power = np.cumsum(psd)
        rolloff_threshold = 0.85 * features['total_power']
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = freqs[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = freqs[-1]
        
        # Dominant frequency (frequency with maximum power)
        features['dominant_frequency'] = freqs[np.argmax(psd)]
        features['dominant_power'] = np.max(psd)
        
        return features


class WaveletFeatures:
    """Extract wavelet-based features for time-frequency analysis."""
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels
        
    def extract_wavelet_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract features from wavelet decomposition."""
        
        features = {}
        
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.levels)
            
            # Features from each level
            for i, coeff in enumerate(coeffs):
                level_name = 'approximation' if i == 0 else f'detail_{i}'
                
                features[f'wavelet_{level_name}_energy'] = np.sum(coeff**2)
                features[f'wavelet_{level_name}_variance'] = np.var(coeff)
                features[f'wavelet_{level_name}_mean'] = np.mean(coeff)
                features[f'wavelet_{level_name}_std'] = np.std(coeff)
                
                # Relative energy
                total_energy = sum(np.sum(c**2) for c in coeffs)
                if total_energy > 0:
                    features[f'wavelet_{level_name}_relative_energy'] = features[f'wavelet_{level_name}_energy'] / total_energy
                else:
                    features[f'wavelet_{level_name}_relative_energy'] = 0
            
            # Shannon entropy of wavelet coefficients
            for i, coeff in enumerate(coeffs):
                level_name = 'approximation' if i == 0 else f'detail_{i}'
                # Calculate normalized energy for entropy
                energy = coeff**2
                if np.sum(energy) > 0:
                    normalized_energy = energy / np.sum(energy)
                    features[f'wavelet_{level_name}_entropy'] = entropy(normalized_energy + 1e-10)  # Add small value to avoid log(0)
                else:
                    features[f'wavelet_{level_name}_entropy'] = 0
                    
        except Exception as e:
            logger.warning(f"Wavelet feature extraction failed: {str(e)}")
            # Return zero features if wavelet analysis fails
            for i in range(self.levels + 1):
                level_name = 'approximation' if i == 0 else f'detail_{i}'
                for feature_type in ['energy', 'variance', 'mean', 'std', 'relative_energy', 'entropy']:
                    features[f'wavelet_{level_name}_{feature_type}'] = 0.0
        
        return features


class CrossSignalFeatures:
    """Extract features from the interaction between ECG and PPG signals."""
    
    @staticmethod
    def extract_cross_correlation_features(ppg_signal: np.ndarray, ecg_signal: np.ndarray, max_lag: int = 50) -> Dict[str, float]:
        """Extract cross-correlation features between PPG and ECG."""
        
        features = {}
        
        # Ensure signals are same length
        min_length = min(len(ppg_signal), len(ecg_signal))
        ppg_trimmed = ppg_signal[:min_length]
        ecg_trimmed = ecg_signal[:min_length]
        
        # Cross-correlation
        cross_corr = np.correlate(ppg_trimmed, ecg_trimmed, mode='full')
        
        # Find peak correlation and lag
        peak_idx = np.argmax(np.abs(cross_corr))
        features['max_cross_correlation'] = cross_corr[peak_idx]
        features['max_cross_correlation_abs'] = np.abs(cross_corr[peak_idx])
        
        # Calculate lag (in samples)
        lag = peak_idx - (len(cross_corr) // 2)
        features['optimal_lag'] = lag
        features['optimal_lag_ms'] = lag * 8  # Convert to milliseconds (assuming 125 Hz)
        
        # Cross-correlation at zero lag
        zero_lag_idx = len(cross_corr) // 2
        features['zero_lag_correlation'] = cross_corr[zero_lag_idx]
        
        # Cross-correlation statistics in limited lag range
        if max_lag < len(cross_corr) // 2:
            center = len(cross_corr) // 2
            limited_cross_corr = cross_corr[center-max_lag:center+max_lag+1]
            features['limited_max_correlation'] = np.max(np.abs(limited_cross_corr))
            features['limited_mean_correlation'] = np.mean(np.abs(limited_cross_corr))
        else:
            features['limited_max_correlation'] = features['max_cross_correlation_abs']
            features['limited_mean_correlation'] = np.mean(np.abs(cross_corr))
        
        return features
    
    @staticmethod
    def extract_pulse_transit_time_features(ppg_signal: np.ndarray, ecg_signal: np.ndarray, 
                                          ppg_peaks: np.ndarray, ecg_peaks: np.ndarray) -> Dict[str, float]:
        """Extract Pulse Transit Time (PTT) related features."""
        
        features = {}
        
        if len(ppg_peaks) == 0 or len(ecg_peaks) == 0:
            ptt_feature_names = [
                'mean_ptt', 'std_ptt', 'cv_ptt', 'min_ptt', 'max_ptt',
                'ptt_trend', 'ptt_variability'
            ]
            for name in ptt_feature_names:
                features[name] = 0.0
            return features
        
        # Calculate PTT for each valid pair
        ptt_values = []
        
        for ecg_peak in ecg_peaks:
            # Find the next PPG peak after this ECG peak
            next_ppg_peaks = ppg_peaks[ppg_peaks > ecg_peak]
            if len(next_ppg_peaks) > 0:
                ptt = next_ppg_peaks[0] - ecg_peak
                # Only include physiologically plausible PTT values (20-300 ms at 125 Hz)
                if 2.5 <= ptt <= 37.5:  # 20-300 ms converted to samples
                    ptt_values.append(ptt * 8)  # Convert to milliseconds
        
        if len(ptt_values) > 0:
            features['mean_ptt'] = np.mean(ptt_values)
            features['std_ptt'] = np.std(ptt_values)
            features['cv_ptt'] = features['std_ptt'] / features['mean_ptt'] if features['mean_ptt'] > 0 else 0
            features['min_ptt'] = np.min(ptt_values)
            features['max_ptt'] = np.max(ptt_values)
            
            # PTT trend (linear regression slope)
            if len(ptt_values) > 2:
                x = np.arange(len(ptt_values))
                features['ptt_trend'] = np.polyfit(x, ptt_values, 1)[0]
            else:
                features['ptt_trend'] = 0
            
            # PTT variability (coefficient of variation)
            features['ptt_variability'] = features['cv_ptt']
        else:
            for name in ['mean_ptt', 'std_ptt', 'cv_ptt', 'min_ptt', 'max_ptt', 'ptt_trend', 'ptt_variability']:
                features[name] = 0.0
        
        return features


class AdvancedFeatureExtractor:
    """Main feature extraction orchestrator with enhanced capabilities."""
    
    def __init__(self, sampling_rate: int = 125, config: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.config = config or {}
        
        # Initialize feature extractors
        self.signal_processor = SignalProcessor(sampling_rate)
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures(sampling_rate)
        self.wavelet_features = WaveletFeatures()
        self.cross_features = CrossSignalFeatures()
        
    def extract_signal_features(self, signal_data: np.ndarray, signal_type: str) -> Dict[str, float]:
        """Extract comprehensive features from a single signal."""
        
        features = {}
        
        # Apply preprocessing
        if self.config.get('filter_enabled', True):
            signal_data = self.signal_processor.apply_filters(signal_data)
        
        # Detect peaks
        peaks, peak_properties = self.signal_processor.detect_peaks(signal_data, signal_type)
        
        # Extract time domain features
        time_features = self.time_features.extract_statistical_features(signal_data)
        morphological_features = self.time_features.extract_morphological_features(signal_data, peaks)
        
        # Extract frequency domain features
        freq_features = self.freq_features.extract_spectral_features(signal_data)
        
        # Extract wavelet features
        if self.config.get('wavelet_features', True):
            wavelet_features = self.wavelet_features.extract_wavelet_features(signal_data)
        else:
            wavelet_features = {}
        
        # Combine all features with signal type prefix
        for feature_name, value in {**time_features, **morphological_features, **freq_features, **wavelet_features}.items():
            features[f'{signal_type.lower()}_{feature_name}'] = value
        
        # Store peaks for cross-signal analysis
        features[f'{signal_type.lower()}_peaks'] = peaks
        
        return features
    
    def extract_cross_signal_features(self, ppg_signal: np.ndarray, ecg_signal: np.ndarray,
                                    ppg_peaks: np.ndarray, ecg_peaks: np.ndarray) -> Dict[str, float]:
        """Extract features from interaction between PPG and ECG signals."""
        
        features = {}
        
        # Cross-correlation features
        cross_corr_features = self.cross_features.extract_cross_correlation_features(ppg_signal, ecg_signal)
        features.update(cross_corr_features)
        
        # PTT features
        ptt_features = self.cross_features.extract_pulse_transit_time_features(
            ppg_signal, ecg_signal, ppg_peaks, ecg_peaks
        )
        features.update(ptt_features)
        
        # Signal synchrony features
        if len(ppg_peaks) > 0 and len(ecg_peaks) > 0:
            features['peak_ratio'] = len(ppg_peaks) / len(ecg_peaks)
            features['peak_count_difference'] = abs(len(ppg_peaks) - len(ecg_peaks))
        else:
            features['peak_ratio'] = 0
            features['peak_count_difference'] = 0
        
        return features
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the dataset."""
        
        logger.info("Starting enhanced feature extraction...")
        
        feature_list = []
        
        for idx, row in df.iterrows():
            try:
                # Extract signals
                ppg_signal = row.iloc[6:206].values.astype(float)
                ecg_signal = row.iloc[206:406].values.astype(float)
                
                # Extract features from individual signals
                ppg_features = self.extract_signal_features(ppg_signal, 'PPG')
                ecg_features = self.extract_signal_features(ecg_signal, 'ECG')
                
                # Extract cross-signal features
                ppg_peaks = ppg_features.pop('ppg_peaks', np.array([]))
                ecg_peaks = ecg_features.pop('ecg_peaks', np.array([]))
                cross_features = self.extract_cross_signal_features(ppg_signal, ecg_signal, ppg_peaks, ecg_peaks)
                
                # Combine all features
                combined_features = {
                    'patient_id': row.iloc[0],
                    'sbp': row.iloc[3],
                    'dbp': row.iloc[4],
                    'mean_ptt_original': row.iloc[5],  # Original PTT from dataset
                    **ppg_features,
                    **ecg_features,
                    **cross_features
                }
                
                feature_list.append(combined_features)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} records")
                    
            except Exception as e:
                logger.error(f"Error extracting features for record {idx}: {str(e)}")
                continue
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(feature_list)
        
        logger.info(f"Feature extraction complete. Extracted {len(feature_df.columns) - 4} features from {len(feature_df)} records")
        
        return feature_df


def extract_enhanced_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Main entry point for enhanced feature extraction.
    
    Args:
        df: Cleaned dataset DataFrame
        config: Feature extraction configuration
        
    Returns:
        DataFrame with extracted features
    """
    extractor = AdvancedFeatureExtractor(sampling_rate=125, config=config)
    return extractor.extract_all_features(df)
