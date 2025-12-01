import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import yaml
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import joblib
from dataclasses import dataclass

# Import fixed feature extraction (addresses bugs identified in forensic analysis)
from fixed_feature_extraction import FixedFeatureExtractor, validate_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessedCycle:
    cycle_index: int
    ecg_segment: np.ndarray
    ppg_segment: np.ndarray
    r_peak_idx: int  # Index in the original signal (or relative if specified)
    sbp_ref: float
    dbp_ref: float
    timestamp: float
    hr_bpm: float
    quality_score: float

class AdaptiveRPeakDetector:
    def __init__(self, config: Dict):
        self.alpha = config.get('alpha', 0.4)
        self.beta = config.get('beta', 0.15)
        self.refractory_period_ms = config.get('refractory_period_ms', 250)
        
    def detect(self, ecg_signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Detect R-peaks using adaptive thresholding on the first derivative.
        """
        logger.info("Starting R-peak detection...")
        # 1. First Algebraic Derivative
        # dECG[n] = ECG[n] - ECG[n-1]
        d_ecg = np.diff(ecg_signal, prepend=ecg_signal[0])
        abs_d_ecg = np.abs(d_ecg)
        
        logger.info("Calculating adaptive thresholds...")
        # 2. Adaptive Threshold Detection
        # We'll implement a simplified version that updates threshold per window or iteratively
        # For efficiency on long signals, we can use a moving window or block processing.
        # Here, we'll use a block-based approach for the threshold to adapt.
        
        r_peaks = []
        
        # Initial estimates
        window_size = int(2.0 * fs) # 2 seconds window for threshold initialization
        
        # Refractory period in samples
        refractory_samples = int(self.refractory_period_ms / 1000 * fs)
        
        last_peak_idx = -refractory_samples
        
        # Iterate through signal
        # To make it truly adaptive as per formula: threshold = alpha * mean(|dECG|) + beta * max(|dECG|)
        # We can calculate this over a moving window.
        
        # Using a sliding window for threshold calculation might be slow in pure Python loop.
        # Vectorized approach: Calculate rolling mean and max.
        
        # Pandas rolling is efficient
        d_ecg_series = pd.Series(abs_d_ecg)
        # Window size for statistics (e.g., 2 seconds)
        stat_window = int(2 * fs)
        
        rolling_mean = d_ecg_series.rolling(window=stat_window, min_periods=1).mean().values
        rolling_max = d_ecg_series.rolling(window=stat_window, min_periods=1).max().values
        
        thresholds = self.alpha * rolling_mean + self.beta * rolling_max
        
        # Find peaks above threshold with dead zone
        # We can use find_peaks with height=thresholds, but thresholds is an array.
        # We'll iterate to apply the dead zone logic strictly.
        
        # Optimization: Find local maxima first, then filter
        potential_peaks, _ = signal.find_peaks(abs_d_ecg, distance=refractory_samples)
        
        final_peaks = []
        for p in potential_peaks:
            if abs_d_ecg[p] > thresholds[p]:
                final_peaks.append(p)
                
        r_peaks = np.array(final_peaks)
        
        if len(r_peaks) < 2:
            return np.array([]), np.array([]), np.array([]), 0.0
            
        # Calculate RR intervals
        rr_intervals_samples = np.diff(r_peaks)
        rr_intervals_ms = rr_intervals_samples / fs * 1000
        
        # Calculate HR
        # Handle division by zero if any RR is 0 (shouldn't happen with find_peaks distance)
        hr_bpm = 60000.0 / rr_intervals_ms
        
        # Quality Score: Consistency of RR intervals
        # Lower coefficient of variation => higher quality
        if len(rr_intervals_ms) > 1:
            cv_rr = np.std(rr_intervals_ms) / np.mean(rr_intervals_ms)
            quality_score = max(0, 1 - cv_rr) # Simple heuristic
        else:
            quality_score = 0.0
            
        return r_peaks, rr_intervals_ms, hr_bpm, quality_score # type: ignore

class BPEstimationPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.fs = self.config.get('sampling_rate', 500)
        self.rpeak_detector = AdaptiveRPeakDetector(self.config['rpeak'])
        self.wavelet_params = self.config['wavelet']
        
        # Initialize the FIXED feature extractor (addresses forensic analysis bugs)
        self.feature_extractor = FixedFeatureExtractor(fs=self.fs)
        
    def load_vitaldb_case(self, file_path: str) -> Dict:
        """Load a single .npz case file from VitalDB format."""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            print(f"Loaded {file_path} with keys: {list(data.keys())}")
            # Extract signals
            # Note: Keys might vary slightly, checking main.py for keys
            # main.py: wave_SNUADC_ECG_II, wave_SNUADC_PLETH, wave_SNUADC_ART
            
            ecg = data['wave_SNUADC_ECG_II']
            ppg = data['wave_SNUADC_PLETH']
            art = data['wave_SNUADC_ART'] # Arterial waveform
            
            # Numeric references (0.5 Hz)
            sbp_nums = data['num_Solar8000_ART_SBP']
            dbp_nums = data['num_Solar8000_ART_DBP']
            
            # Metadata
            metadata = {}
            if 'metadata' in data:
                import json
                metadata = json.loads(str(data['metadata']))
            
            return {
                'ecg_raw': ecg,
                'ppg_raw': ppg,
                'art_raw': art,
                'sbp_ref_series': sbp_nums,
                'dbp_ref_series': dbp_nums,
                'metadata': metadata,
                'filename': os.path.basename(file_path)
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None # type: ignore

    def wavelet_filter(self, signal_data: np.ndarray, hr_bpm: float, signal_type: str) -> np.ndarray:
        """
        Apply CWT-based filtering with adaptive frequency bands.
        Processes in chunks to save memory.
        """
        # Handle NaNs
        if np.isnan(signal_data).any():
            # Simple imputation for filtering
            mask = np.isnan(signal_data)
            signal_data[mask] = np.nanmean(signal_data)
            
        # Determine frequency bands
        if signal_type == 'ecg':
            low_freq = self.wavelet_params['ecg_band_low_mult'] * float(hr_bpm) / 60.0
            high_freq = self.wavelet_params['ecg_band_high_mult'] * float(hr_bpm) / 60.0
            wavelet_name = self.wavelet_params['family_ecg']
        else: # ppg
            low_freq = self.wavelet_params['ppg_band_low_mult'] * float(hr_bpm) / 60.0
            high_freq = self.wavelet_params['ppg_band_high_mult'] * float(hr_bpm) / 60.0
            wavelet_name = self.wavelet_params['family_ppg']
            
        # Ensure valid frequencies
        low_freq = max(0.5, low_freq)
        high_freq = min(self.fs / 2.1, high_freq)
        
        if low_freq >= high_freq:
            logger.warning(f"Invalid frequency band [{low_freq}, {high_freq}] for HR {hr_bpm}. Using defaults.")
            low_freq = 0.5 if signal_type == 'ecg' else 0.5
            high_freq = 40.0 if signal_type == 'ecg' else 10.0

        # Create scales for CWT
        freqs = np.linspace(low_freq, high_freq, 32)
        scales = pywt.frequency2scale(wavelet_name, freqs) * self.fs
        logger.info(f"Wavelet filtering {signal_type}: Freqs [{low_freq:.2f}, {high_freq:.2f}] Hz, Scales range [{min(scales):.1f}, {max(scales):.1f}]")
        
        # Chunk processing
        chunk_size = 10000 # Reduced chunk size
        n_samples = len(signal_data)
        reconstructed_signal = np.zeros_like(signal_data)
        
        icwt_func = getattr(pywt, 'icwt', None)

        for i in range(0, n_samples, chunk_size):
            start = i
            end = min(i + chunk_size, n_samples)
            chunk = signal_data[start:end]
            
            if len(chunk) < 2:
                continue
                
            # CWT
            try:
                coeffs, frequencies = pywt.cwt(chunk, scales, wavelet_name, sampling_period=1/self.fs, method='fft')
            except TypeError:
                coeffs, frequencies = pywt.cwt(chunk, scales, wavelet_name, sampling_period=1/self.fs)
            
            # Inverse CWT
            if icwt_func:
                rec_chunk = icwt_func(coeffs, scales, wavelet_name, sampling_period=1/self.fs)[0]
            else:
                # Fallback: Sum of real part of coefficients
                rec_chunk = np.sum(np.real(coeffs), axis=0)
                
            reconstructed_signal[start:end] = rec_chunk
            
        return reconstructed_signal

    def segment_and_sync_cycles(self, ecg_filtered: np.ndarray, ppg_filtered: np.ndarray, 
                               r_peaks: np.ndarray, sbp_series: np.ndarray, dbp_series: np.ndarray) -> List[ProcessedCycle]:
        """
        Segment PPG cycles, select representative ones, and sync with ECG.
        """
        cycles = []
        
        # Numeric BP is at 0.5 Hz (every 2 seconds). We need to interpolate or map to beats.
        # We'll map each beat to the nearest BP value in time.
        # BP time array:
        bp_times = np.arange(len(sbp_series)) * 2.0 # 0, 2, 4... seconds
        
        skipped_len = 0
        skipped_nan = 0
        
        for i in range(len(r_peaks) - 1):
            start_idx = r_peaks[i]
            end_idx = r_peaks[i+1]
            
            # Check cycle length
            cycle_len_ms = (end_idx - start_idx) / self.fs * 1000
            if not (self.config['segmentation']['min_cycle_len_ms'] <= cycle_len_ms <= self.config['segmentation']['max_cycle_len_ms']):
                skipped_len += 1
                continue
                
            ppg_cycle = ppg_filtered[start_idx:end_idx]
            ecg_cycle = ecg_filtered[start_idx:end_idx]
            
            # Get Reference BP
            beat_time_sec = start_idx / self.fs
            # Find nearest BP index
            bp_idx = int(round(beat_time_sec / 2.0))
            
            if 0 <= bp_idx < len(sbp_series):
                sbp = sbp_series[bp_idx]
                dbp = dbp_series[bp_idx]
                
                if np.isnan(sbp) or np.isnan(dbp):
                    skipped_nan += 1
                    continue
                
                # Validate BP range
                if not (self.config['qc']['valid_sbp_range'][0] <= sbp <= self.config['qc']['valid_sbp_range'][1]):
                    continue
                if not (self.config['qc']['valid_dbp_range'][0] <= dbp <= self.config['qc']['valid_dbp_range'][1]):
                    continue
                    
                # Create cycle object
                # We calculate HR for this cycle
                hr = 60000.0 / cycle_len_ms
                
                cycles.append(ProcessedCycle(
                    cycle_index=i,
                    ecg_segment=ecg_cycle,
                    ppg_segment=ppg_cycle,
                    r_peak_idx=start_idx,
                    sbp_ref=sbp,
                    dbp_ref=dbp,
                    timestamp=beat_time_sec,
                    hr_bpm=hr,
                    quality_score=1.0 # Placeholder, updated later
                ))
        
        logger.info(f"Segmented {len(cycles)} cycles. Skipped: {skipped_len} (len), {skipped_nan} (NaN BP).")
                
        # Representative Cycle Selection (Batch-wise or Window-wise)
        # For continuous estimation, we might want to keep all valid cycles.
        # The prompt suggests: "Select cycle with highest mean correlation as template... Reject outlier cycles"
        # This implies we filter out bad cycles.
        
        if not cycles:
            return []
            
        # Extract PPG segments for correlation
        # They are variable length, so we need to resample to fixed length for correlation matrix
        fixed_len = 200
        ppg_segments_resampled = []
        
        logger.info("Resampling cycles for correlation check...")
        for c in cycles:
            resampled = signal.resample(c.ppg_segment, fixed_len)
            ppg_segments_resampled.append(resampled)
            
        ppg_matrix = np.array(ppg_segments_resampled)
        logger.info("Computing correlation with template...")
        
        # Compute correlation with median template instead of full matrix to avoid O(N^2) memory/time
        # Median template is robust to outliers
        template = np.median(ppg_matrix, axis=0)
        
        # Normalize template
        template_mean = np.mean(template)
        template_std = np.std(template)
        template_norm = (template - template_mean) / (template_std + 1e-6)
        
        # Normalize all cycles
        means = np.mean(ppg_matrix, axis=1, keepdims=True)
        stds = np.std(ppg_matrix, axis=1, keepdims=True)
        ppg_norm = (ppg_matrix - means) / (stds + 1e-6)
        
        # Calculate correlation: dot product / N
        correlations = np.dot(ppg_norm, template_norm) / fixed_len
        
        # Filter
        threshold = self.config['segmentation']['correlation_threshold']
        final_cycles = []
        
        for idx, c in enumerate(cycles):
            corr = correlations[idx]
            if corr >= threshold:
                c.quality_score = float(corr)
                final_cycles.append(c)
                
        logger.info(f"Selected {len(final_cycles)} cycles after correlation filter (threshold {threshold}).")
        return final_cycles

    def extract_features_batch(self, cycles: List[ProcessedCycle]) -> pd.DataFrame:
        """
        Extract comprehensive features from synchronized cycles using FIXED extraction.
        
        This method uses the FixedFeatureExtractor which addresses the bugs identified
        in the forensic analysis:
        - ptt_peak_to_foot: Was 93.4% zeros due to argmin bug (now uses intersecting tangent)
        - amplitude_ratio_ra: Was 78.1% near-zero due to notch detection failure (now uses 2nd derivative)
        - time_to_maxslope_t1: Many zeros due to incorrect calculation
        
        Additionally extracts new predictive features:
        - max_upslope, max_downslope, slope_ratio
        - max_acceleration, min_acceleration  
        - systolic_area_ratio
        - pulse_width_50, crest_time, reflection_index
        """
        logger.info(f"Extracting features from {len(cycles)} cycles using FIXED extractor...")
        features_list = []
        
        # Statistics tracking
        stats = {
            'total': len(cycles),
            'valid': 0,
            'invalid_flat': 0,
            'invalid_features': 0,
            'invalid_ptt_range': 0
        }
        
        for i, cycle in enumerate(cycles):
            if i % 5000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(cycles)} cycles... Valid: {stats['valid']}")
            
            ppg = cycle.ppg_segment
            
            # Use the FIXED feature extractor with R-peak timing
            # The cycle starts at R-peak, so r_peak is at index 0 of the segment
            features = self.feature_extractor.extract_features(
                ppg, 
                hr_bpm=cycle.hr_bpm,
                r_peak_sample=cycle.r_peak_idx,
                cycle_start_sample=cycle.r_peak_idx
            )
            
            # Check if extraction succeeded
            if not features:
                stats['invalid_flat'] += 1
                continue
            
            # Validate features are within physiological bounds
            if not validate_features(features):
                stats['invalid_features'] += 1
                continue
            
            # NOTE: Removed redundant PTT range validation - validate_features handles this
            # Let the model learn from all valid data, outliers will be handled by robust training
            
            stats['valid'] += 1
            
            # Build feature record with cycle metadata
            feature_record = {
                'cycle_index': cycle.cycle_index,
                'timestamp': cycle.timestamp,
                'sbp_reference': cycle.sbp_ref,
                'dbp_reference': cycle.dbp_ref,
                'cycle_correlation': cycle.quality_score,
                # NEW: True ECG-PPG PTT (Pulse Arrival Time) - physiologically meaningful
                'pat_ecg_ppg': features.get('pat_ecg_ppg', np.nan),  # R-peak to PPG foot
                'pat_to_peak': features.get('pat_to_peak', np.nan),   # R-peak to PPG peak
                'pat_to_maxslope': features.get('pat_to_maxslope', np.nan),  # R-peak to max slope
                # Legacy PTT features (PPG-internal timing)
                'ptt_peak_to_peak': features.get('ptt_peak_to_peak', np.nan),
                'ptt_peak_to_foot': features.get('ptt_peak_to_foot', np.nan),
                'ptt_peak_to_maxslope': features.get('ptt_peak_to_maxslope', np.nan),
                'time_to_maxslope_t1': features.get('time_to_maxslope_t1', np.nan),
                # Core morphology features (FIXED)
                'amplitude_ratio_ra': features.get('amplitude_ratio_ra', np.nan),
                'systolic_duration_tsd': features.get('systolic_duration_tsd', np.nan),
                'diastolic_duration_tfd': features.get('diastolic_duration_tfd', np.nan),
                # Heart rate
                'hr_bpm': features.get('hr_bpm', cycle.hr_bpm),
                # NEW: Derivative features
                'max_upslope': features.get('max_upslope', np.nan),
                'max_downslope': features.get('max_downslope', np.nan),
                'slope_ratio': features.get('slope_ratio', np.nan),
                'max_acceleration': features.get('max_acceleration', np.nan),
                'min_acceleration': features.get('min_acceleration', np.nan),
                # NEW: Waveform features
                'systolic_area_ratio': features.get('systolic_area_ratio', np.nan),
                'pulse_width_50': features.get('pulse_width_50', np.nan),
                'crest_time': features.get('crest_time', np.nan),
                'reflection_index': features.get('reflection_index', np.nan),
                # NEW: Statistical features (Robust Fallback)
                'stat_mean': features.get('stat_mean', np.nan),
                'stat_std': features.get('stat_std', np.nan),
                'stat_skew': features.get('stat_skew', np.nan),
                'stat_kurtosis': features.get('stat_kurtosis', np.nan),
                'stat_power_cardiac': features.get('stat_power_cardiac', np.nan),
                'stat_power_low': features.get('stat_power_low', np.nan),
                'stat_power_ratio': features.get('stat_power_ratio', np.nan),
            }
            
            features_list.append(feature_record)
        
        # Log extraction statistics
        logger.info(f"Feature extraction complete:")
        logger.info(f"  Total cycles: {stats['total']}")
        logger.info(f"  Valid: {stats['valid']} ({100*stats['valid']/max(1,stats['total']):.1f}%)")
        logger.info(f"  Invalid (flat signal): {stats['invalid_flat']}")
        logger.info(f"  Invalid (out of bounds): {stats['invalid_features']}")
        logger.info(f"  Invalid (PTT range): {stats['invalid_ptt_range']}")
        
        return pd.DataFrame(features_list)

    def process_recording(self, file_path: str) -> pd.DataFrame:
        """Full pipeline for one recording."""
        # Stage 1: Load
        data = self.load_vitaldb_case(file_path)
        if data is None:
            return pd.DataFrame()
            
        ecg_raw = data['ecg_raw']
        ppg_raw = data['ppg_raw']
        sbp_ref = data['sbp_ref_series']
        dbp_ref = data['dbp_ref_series']
        
        # Check lengths
        min_len = min(len(ecg_raw), len(ppg_raw))
        ecg_raw = ecg_raw[:min_len]
        ppg_raw = ppg_raw[:min_len]
        
        # Stage 2: R-peak detection
        r_peaks, rr_intervals, hr_bpm_inst, quality = self.rpeak_detector.detect(ecg_raw, self.fs)
        logger.info(f"Detected {len(r_peaks)} R-peaks.")
        
        if len(r_peaks) < 10:
            logger.warning(f"Not enough R-peaks in {file_path}")
            return pd.DataFrame()
            
        mean_hr = np.mean(hr_bpm_inst) if len(hr_bpm_inst) > 0 else 70.0
        mean_hr = np.clip(mean_hr, 40.0, 180.0)
        logger.info(f"Mean HR: {mean_hr:.1f} BPM")
        
        # Stage 3: Wavelet filtering
        # Process in chunks if memory is an issue, but for 30min @ 500Hz (900k samples), it fits in RAM.
        ecg_filtered = self.wavelet_filter(ecg_raw, mean_hr, 'ecg')
        ppg_filtered = self.wavelet_filter(ppg_raw, mean_hr, 'ppg')
        
        # Stage 4: Segmentation & Sync
        cycles = self.segment_and_sync_cycles(ecg_filtered, ppg_filtered, r_peaks, sbp_ref, dbp_ref)
        
        # Stage 5: Feature Extraction
        df_features = self.extract_features_batch(cycles)
        
        # Add metadata
        if not df_features.empty:
            df_features['patient_id'] = data['metadata'].get('caseid', 'unknown')
            df_features['session_id'] = data['filename']
            
            # Clinical metadata
            meta = data['metadata']
            for col in ['age', 'sex', 'bmi', 'position', 'approach', 'aline1', 'dx', 'opname', 'preop_ecg']:
                df_features[col] = meta.get(col, np.nan if col in ['age', 'bmi'] else 'unknown')
            
        return df_features

    def run_batch(self, data_dir: str, max_files: Optional[int] = None):
        """Run pipeline on all .npz files in directory."""
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        if max_files:
            files = files[:max_files]
            
        logger.info(f"Processing {len(files)} files...")
        
        # Parallel processing
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.process_recording)(f) for f in files
        )
        
        # Combine results
        all_features = pd.concat(results, ignore_index=True)
        
        # Stage 6: Output
        output_path = os.path.join(self.config['output_dir'], 'bp_dataset_features.csv')
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Final validation
        # Remove records with quality scores < 0.7 (already done in segmentation mostly, but check features)
        # Check for NaN
        all_features.dropna(inplace=True)
        
        # SBP > DBP check
        all_features = all_features[all_features['sbp_reference'] > all_features['dbp_reference']]
        
        all_features.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path} with {len(all_features)} records.")
        
        return all_features

if __name__ == "__main__":
    # Example usage
    pipeline = BPEstimationPipeline('pipeline_config.yaml')
    
    # Assuming data is in ./data/raw
    raw_data_dir = './data/raw'
    if os.path.exists(raw_data_dir):
        df = pipeline.run_batch(raw_data_dir, max_files=5)
        print(df.head())
        print(df.describe())
    else:
        print(f"Data directory {raw_data_dir} not found.")
