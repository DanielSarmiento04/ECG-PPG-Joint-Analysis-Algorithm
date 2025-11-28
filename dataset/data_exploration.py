"""
VitalDB Data Exploration and Visualization Script
Author: Research Team
Description: Explores and visualizes downloaded VitalDB data for BP estimation research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VitalDBExplorer:
    """Class for exploring and visualizing VitalDB data"""
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize explorer
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'raw'
        self.metadata_dir = self.data_dir / 'metadata'
        self.figures_dir = self.data_dir / 'figures'
        
        # Create figures directory
        self.figures_dir.mkdir(exist_ok=True)
        
        # Sampling rates
        self.fs_wave = 500  # Hz for waveforms
        self.fs_numeric = 0.5  # Hz for numeric data (every 2 seconds)
    
    def load_case(self, caseid: int) -> Dict | None:
        """
        Load a single case
        
        Args:
            caseid: Case ID to load
            
        Returns:
            Dictionary with case data
        """
        case_file = self.raw_data_dir / f'case_{caseid:05d}.npz'
        
        if not case_file.exists():
            logger.error(f"Case {caseid} not found!")
            return None
        
        data = np.load(case_file, allow_pickle=True)
        
        # Parse data
        case_data = {
            'caseid': caseid,
            'waveforms': {},
            'numerics': {},
            'metadata': json.loads(str(data['metadata']))
        }
        
        # Load waveforms
        for key in data.files:
            if key.startswith('wave_'):
                # Convert: wave_SNUADC_ECG_II -> SNUADC/ECG_II
                # Only replace the first underscore (between device and track)
                param_name = key.replace('wave_', '', 1)
                if '_' in param_name:
                    parts = param_name.split('_', 1)  # Split only on first underscore
                    param_name = parts[0] + '/' + parts[1]
                case_data['waveforms'][param_name] = data[key]
            elif key.startswith('num_'):
                # Convert: num_Solar8000_ART_SBP -> Solar8000/ART_SBP
                param_name = key.replace('num_', '', 1)
                if '_' in param_name:
                    parts = param_name.split('_', 1)
                    param_name = parts[0] + '/' + parts[1]
                case_data['numerics'][param_name] = data[key]
        
        return case_data
    
    def get_dataset_overview(self) -> pd.DataFrame | None:
        """
        Get overview of all downloaded cases
        
        Returns:
            DataFrame with case statistics
        """
        logger.info("Generating dataset overview...")
        
        # Load case list
        cases_file = self.metadata_dir / 'available_cases.csv'
        if not cases_file.exists():
            logger.error("Case list not found!")
            return None
        
        df_cases = pd.read_csv(cases_file)
        
        # Get list of downloaded cases
        downloaded_cases = [
            int(f.stem.split('_')[1]) 
            for f in self.raw_data_dir.glob('case_*.npz')
        ]
        
        df_downloaded = df_cases[df_cases['caseid'].isin(downloaded_cases)]
        
        logger.info(f"Total downloaded cases: {len(df_downloaded)}")
        
        return df_downloaded
    
    def plot_case_waveforms(
        self, 
        caseid: int, 
        duration_sec: float = 30, 
        start_time: float | None = None
    ) -> None:
        """
        Plot waveforms for a single case
        
        Args:
            caseid: Case ID
            duration_sec: Duration to plot in seconds
            start_time: Start time in seconds (None = auto-detect first valid data)
        """
        logger.info(f"Plotting waveforms for case {caseid}...")
        
        # Load case
        case_data = self.load_case(caseid)
        if not case_data:
            return
        
        waveforms = case_data['waveforms']
        
        # Auto-detect start time if not provided (skip NaN values)
        if start_time is None:
            # Find first valid data point across all signals
            first_valid_idx = len(waveforms.get('SNUADC/ECG_II', []))
            for signal_data in waveforms.values():
                if len(signal_data) > 0:
                    non_nan = np.where(~np.isnan(signal_data))[0]
                    if len(non_nan) > 0:
                        first_valid_idx = min(first_valid_idx, non_nan[0])
            
            # Start 10 seconds after first valid data
            start_time = (first_valid_idx / self.fs_wave) + 10
            logger.info(f"Auto-detected start time: {start_time:.1f}s")
        
        # Calculate sample indices
        start_idx = int(start_time * self.fs_wave)
        end_idx = int((start_time + duration_sec) * self.fs_wave)
        
        # Create time array
        n_samples = end_idx - start_idx
        time = np.arange(n_samples) / self.fs_wave
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Case {caseid} - Waveform Signals ({duration_sec}s window)', 
                     fontsize=16, fontweight='bold')
        
        # Plot ECG
        if 'SNUADC/ECG_II' in waveforms:
            ecg = waveforms['SNUADC/ECG_II'][start_idx:end_idx]
            axes[0].plot(time, ecg, 'b-', linewidth=0.8)
            axes[0].set_ylabel('ECG II (mV)', fontsize=12, fontweight='bold')
            axes[0].set_title('Electrocardiogram (ECG)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        # Plot PPG
        if 'SNUADC/PLETH' in waveforms:
            ppg = waveforms['SNUADC/PLETH'][start_idx:end_idx]
            # Normalize PPG for better visualization
            ppg_norm = (ppg - np.mean(ppg)) / np.std(ppg)
            axes[1].plot(time, ppg_norm, 'r-', linewidth=0.8)
            axes[1].set_ylabel('PPG (normalized)', fontsize=12, fontweight='bold')
            axes[1].set_title('Photoplethysmogram (PPG)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        # Plot Arterial BP
        if 'SNUADC/ART' in waveforms:
            art = waveforms['SNUADC/ART'][start_idx:end_idx]
            axes[2].plot(time, art, 'g-', linewidth=0.8)
            axes[2].set_ylabel('ABP (mmHg)', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            axes[2].set_title('Arterial Blood Pressure', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Add BP reference lines (use nanpercentile to handle NaN values)
            if len(art) > 0 and not np.all(np.isnan(art)):
                sbp = np.nanpercentile(art, 95)
                dbp = np.nanpercentile(art, 5)
                axes[2].axhline(y=sbp, color='darkgreen', linestyle='--', 
                               alpha=0.5, label=f'SBP ≈ {sbp:.0f}')
                axes[2].axhline(y=dbp, color='darkred', linestyle='--', 
                               alpha=0.5, label=f'DBP ≈ {dbp:.0f}')
                axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / f'case_{caseid:05d}_waveforms.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {fig_path}")
        
        plt.show()
    
    def plot_single_beat(self, caseid: int, beat_number: int = 10) -> None:
        """
        Plot detailed view of a single cardiac cycle
        
        Args:
            caseid: Case ID
            beat_number: Which beat to plot
        """
        logger.info(f"Plotting single beat analysis for case {caseid}...")
        
        # Load case
        case_data = self.load_case(caseid)
        if not case_data:
            return
        
        waveforms = case_data['waveforms']
        
        # Get ECG and find R peaks
        if 'SNUADC/ECG_II' not in waveforms:
            logger.error("ECG not available!")
            return
        
        ecg = waveforms['SNUADC/ECG_II']
        
        # Simple R-peak detection
        ecg_filt = signal.medfilt(ecg, kernel_size=5)
        peaks, _ = signal.find_peaks(ecg_filt, distance=self.fs_wave*0.5, 
                                      height=np.percentile(ecg_filt[~np.isnan(ecg_filt)], 75))
        
        if len(peaks) < 2:
            logger.error(f"Not enough heart beats detected ({len(peaks)} peaks found)")
            return
        
        if beat_number >= len(peaks) - 1:
            beat_number = len(peaks) - 2
            logger.info(f"Adjusted beat_number to {beat_number}")
        
        # Extract single beat
        r_idx = peaks[beat_number]
        r_idx_next = peaks[beat_number + 1]
        
        # Get data for one beat (with some padding)
        padding = int(0.1 * self.fs_wave)  # 100ms padding
        start_idx = max(0, r_idx - padding)
        end_idx = min(len(ecg), r_idx_next + padding)
        
        beat_ecg = ecg[start_idx:end_idx]
        beat_ppg = waveforms.get('SNUADC/PLETH', np.zeros(len(ecg)))[start_idx:end_idx]
        beat_art = waveforms.get('SNUADC/ART', np.zeros(len(ecg)))[start_idx:end_idx]
        
        # Normalize
        beat_ppg_norm = (beat_ppg - np.min(beat_ppg)) / (np.max(beat_ppg) - np.min(beat_ppg) + 1e-8)
        
        # Create time array
        time_ms = np.arange(len(beat_ecg)) / self.fs_wave * 1000
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Case {caseid} - Single Cardiac Cycle (Beat #{beat_number})', 
                     fontsize=16, fontweight='bold')
        
        # ECG with R-peak marked
        axes[0].plot(time_ms, beat_ecg, 'b-', linewidth=1.5)
        r_time = (r_idx - start_idx) / self.fs_wave * 1000
        axes[0].plot(r_time, beat_ecg[r_idx - start_idx], 'ro', 
                    markersize=10, label='R-peak')
        axes[0].set_ylabel('ECG (mV)', fontsize=11, fontweight='bold')
        axes[0].set_title('ECG Signal', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PPG with features
        axes[1].plot(time_ms, beat_ppg_norm, 'r-', linewidth=1.5)
        
        # Find PPG peaks (systolic and dicrotic)
        ppg_peaks, _ = signal.find_peaks(beat_ppg_norm, distance=int(0.2*self.fs_wave))
        if len(ppg_peaks) >= 2:
            axes[1].plot(time_ms[ppg_peaks[0]], beat_ppg_norm[ppg_peaks[0]], 
                        'go', markersize=10, label='Systolic peak')
            axes[1].plot(time_ms[ppg_peaks[1]], beat_ppg_norm[ppg_peaks[1]], 
                        'mo', markersize=8, label='Dicrotic peak')
            
            # Calculate PTT
            ptt_ms = time_ms[ppg_peaks[0]] - r_time
            axes[1].annotate(f'PTT = {ptt_ms:.1f} ms', 
                           xy=(time_ms[ppg_peaks[0]], beat_ppg_norm[ppg_peaks[0]]),
                           xytext=(time_ms[ppg_peaks[0]] + 50, beat_ppg_norm[ppg_peaks[0]] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                           fontsize=10, fontweight='bold')
        
        axes[1].set_ylabel('PPG (normalized)', fontsize=11, fontweight='bold')
        axes[1].set_title('Photoplethysmogram', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Arterial BP
        axes[2].plot(time_ms, beat_art, 'g-', linewidth=1.5)
        
        # Mark systolic and diastolic points
        sbp_idx = np.argmax(beat_art)
        dbp_idx = np.argmin(beat_art)
        axes[2].plot(time_ms[sbp_idx], beat_art[sbp_idx], 'ro', 
                    markersize=10, label=f'SBP = {beat_art[sbp_idx]:.0f} mmHg')
        axes[2].plot(time_ms[dbp_idx], beat_art[dbp_idx], 'bo', 
                    markersize=10, label=f'DBP = {beat_art[dbp_idx]:.0f} mmHg')
        
        axes[2].set_ylabel('ABP (mmHg)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        axes[2].set_title('Arterial Blood Pressure', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        fig_path = self.figures_dir / f'case_{caseid:05d}_single_beat.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {fig_path}")
        
        plt.show()
    
    def plot_dataset_statistics(self, df_cases: pd.DataFrame) -> None:
        """
        Plot dataset statistics
        
        Args:
            df_cases: DataFrame with case information
        """
        logger.info("Plotting dataset statistics...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('VitalDB Dataset Overview', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0, 0].hist(df_cases['age'].dropna(), bins=30, color='skyblue', 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Age (years)', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title(f'Age Distribution (n={df_cases["age"].notna().sum()})', 
                            fontsize=12)
        axes[0, 0].axvline(df_cases['age'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {df_cases["age"].mean():.1f}')
        axes[0, 0].legend()
        
        # BMI distribution
        axes[0, 1].hist(df_cases['bmi'].dropna(), bins=30, color='lightgreen', 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('BMI (kg/m²)', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title(f'BMI Distribution (n={df_cases["bmi"].notna().sum()})', 
                            fontsize=12)
        axes[0, 1].axvline(df_cases['bmi'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {df_cases["bmi"].mean():.1f}')
        axes[0, 1].legend()
        
        # Sex distribution
        sex_counts = df_cases['sex'].value_counts()
        axes[0, 2].bar(sex_counts.index, sex_counts.values, 
                      color=['lightcoral', 'lightskyblue'], edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Sex', fontsize=11)
        axes[0, 2].set_ylabel('Count', fontsize=11)
        axes[0, 2].set_title('Sex Distribution', fontsize=12)
        for i, (idx, val) in enumerate(sex_counts.items()):
            axes[0, 2].text(i, val + 5, str(val), ha='center', fontweight='bold')
        
        # Duration distribution
        axes[1, 0].hist(df_cases['duration_min'].dropna(), bins=30, 
                       color='plum', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Duration (minutes)', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Surgery Duration', fontsize=12)
        axes[1, 0].axvline(df_cases['duration_min'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {df_cases["duration_min"].mean():.0f} min')
        axes[1, 0].legend()
        
        # ASA classification
        asa_counts = df_cases['asa'].value_counts().sort_index()
        axes[1, 1].bar(asa_counts.index.astype(str), asa_counts.values, 
                      color='wheat', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('ASA Class', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('ASA Physical Status', fontsize=12)
        
        # Department distribution
        dept_counts = df_cases['department'].value_counts().head(8)
        axes[1, 2].barh(range(len(dept_counts)), dept_counts.values, 
                       color='lightsteelblue', edgecolor='black', alpha=0.7)
        axes[1, 2].set_yticks(range(len(dept_counts)))
        axes[1, 2].set_yticklabels(dept_counts.index, fontsize=9)
        axes[1, 2].set_xlabel('Count', fontsize=11)
        axes[1, 2].set_title('Top Surgical Departments', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        fig_path = self.figures_dir / 'dataset_statistics.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {fig_path}")
        
        plt.show()
    
    def analyze_signal_quality(self, caseid: int, duration_min: float = 5) -> Dict | None:
        """
        Analyze signal quality for a case
        
        Args:
            caseid: Case ID
            duration_min: Duration to analyze in minutes
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"Analyzing signal quality for case {caseid}...")
        
        case_data = self.load_case(caseid)
        if not case_data:
            return None
        
        waveforms = case_data['waveforms']
        
        # Analyze up to duration_min
        n_samples = int(duration_min * 60 * self.fs_wave)
        
        quality_metrics = {}
        
        for signal_name, signal_data in waveforms.items():
            if len(signal_data) < n_samples:
                data = signal_data
            else:
                data = signal_data[:n_samples]
            
            # Calculate metrics
            metrics = {
                'mean': np.mean(data),
                'std': np.std(data),
                'snr_db': 10 * np.log10(np.var(data) / (np.var(np.diff(data)) + 1e-10)),
                'missing_ratio': np.sum(np.isnan(data)) / len(data),
                'flat_segments': self._detect_flat_segments(data),
            }
            
            quality_metrics[signal_name] = metrics
        
        return quality_metrics
    
    def _detect_flat_segments(self, data: np.ndarray, threshold: float = 1e-6, 
                             min_length: int = 100) -> int:
        """Detect flat segments in signal"""
        diff = np.abs(np.diff(data))
        flat = diff < threshold
        
        # Count continuous flat segments
        count = 0
        current_length = 0
        for is_flat in flat:
            if is_flat:
                current_length += 1
            else:
                if current_length >= min_length:
                    count += 1
                current_length = 0
        
        return count


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("VitalDB Data Exploration and Visualization")
    print("=" * 80)
    
    # Initialize explorer
    explorer = VitalDBExplorer(data_dir='./data')
    
    # Check if data exists
    if not explorer.raw_data_dir.exists() or not list(explorer.raw_data_dir.glob('*.npz')):
        print("\n❌ No data found! Please run main.py first to download data.")
        return
    
    # Get dataset overview
    print("\n[Step 1] Loading dataset overview...")
    df_cases = explorer.get_dataset_overview()
    
    if df_cases is None or len(df_cases) == 0:
        print("❌ No cases found!")
        return
    
    print(f"\n✓ Found {len(df_cases)} downloaded cases")
    print("\nDataset Statistics:")
    print(f"  Age: {df_cases['age'].mean():.1f} ± {df_cases['age'].std():.1f} years")
    print(f"  BMI: {df_cases['bmi'].mean():.1f} ± {df_cases['bmi'].std():.1f} kg/m²")
    print(f"  Duration: {df_cases['duration_min'].mean():.1f} ± {df_cases['duration_min'].std():.1f} min")
    print(f"  Sex: {df_cases['sex'].value_counts().to_dict()}")
    
    # Plot dataset statistics
    print("\n[Step 2] Generating dataset overview plots...")
    explorer.plot_dataset_statistics(df_cases)
    
    # Select a representative case
    case_to_visualize = df_cases.iloc[0]['caseid']
    print(f"\n[Step 3] Visualizing case {case_to_visualize}...")
    
    # Plot waveforms
    print("  - Plotting 30-second waveform window...")
    explorer.plot_case_waveforms(case_to_visualize, duration_sec=30)
    
    # Plot single beat
    print("  - Plotting single cardiac cycle analysis...")
    explorer.plot_single_beat(case_to_visualize, beat_number=50)
    
    # Analyze signal quality
    print("\n[Step 4] Analyzing signal quality...")
    quality = explorer.analyze_signal_quality(case_to_visualize, duration_min=5)
    
    if quality:
        print("\nSignal Quality Metrics:")
        for signal_name, metrics in quality.items():
            print(f"\n  {signal_name}:")
            print(f"    SNR: {metrics['snr_db']:.1f} dB")
            print(f"    Missing: {metrics['missing_ratio']*100:.2f}%")
            print(f"    Flat segments: {metrics['flat_segments']}")
    
    print("\n" + "=" * 80)
    print("Exploration Complete!")
    print("=" * 80)
    print(f"\nFigures saved to: {explorer.figures_dir}")
    print("\nNext steps:")
    print("  1. Review the generated visualizations")
    print("  2. Implement feature extraction pipeline")
    print("  3. Train BP estimation model")


if __name__ == "__main__":
    main()