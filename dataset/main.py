"""
VitalDB Data Download Script for BP Estimation Research
Author: Research Team
Description: Downloads ECG, PPG, and arterial BP data from VitalDB for cuffless BP estimation

VitalDB API version 1.5.8 compatibility
"""

import vitaldb
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vitaldb_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VitalDBDownloader:
    """Class to handle VitalDB data downloading and preprocessing"""
    
    def __init__(self, output_dir: str = './data'):
        """
        Initialize the downloader
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = output_dir
        self.raw_data_dir = os.path.join(output_dir, 'raw')
        self.processed_data_dir = os.path.join(output_dir, 'processed')
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        
        # Create directories
        for directory in [self.raw_data_dir, self.processed_data_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Define required parameters
        # For load_case API: use track names without device prefix
        self.waveform_tracks = ['ECG_II', 'PLETH', 'ART']
        self.numeric_tracks = ['ART_SBP', 'ART_DBP', 'ART_MBP', 'HR', 'PLETH_SPO2']
        
        # For reference/labeling: include device prefix
        self.waveform_params = [
            'SNUADC/ECG_II',      # ECG Lead II (500 Hz)
            'SNUADC/PLETH',       # Plethysmography (500 Hz)
            'SNUADC/ART',         # Arterial pressure waveform (500 Hz)
        ]
        
        self.numeric_params = [
            'Solar8000/ART_SBP',  # Systolic BP
            'Solar8000/ART_DBP',  # Diastolic BP
            'Solar8000/ART_MBP',  # Mean BP
            'Solar8000/HR',       # Heart Rate
            'Solar8000/PLETH_SPO2',  # SpO2
        ]
        
        # Enhanced parameters for improved model
        self.enhanced_tracks = ['SVV', 'SV', 'CO', 'SVR']
        self.enhanced_params = [
            'Vigileo/SVV',        # Stroke Volume Variation
            'Vigileo/SV',         # Stroke Volume
            'Vigileo/CO',         # Cardiac Output
            'EV1000/SVR',         # Systemic Vascular Resistance
        ]
        
        # Clinical metadata
        self.clinical_params = [
            'caseid', 'age', 'sex', 'height', 'weight', 'bmi',
            'preop_htn', 'preop_dm', 'asa', 'optype',
            'ane_type', 'department'
        ]
    
    def get_available_cases(self, min_duration_min: int = 30) -> pd.DataFrame:
        """
        Get list of available cases with required signals
        
        Args:
            min_duration_min: Minimum duration in minutes
            
        Returns:
            DataFrame with case information
        """
        logger.info("Fetching available cases from VitalDB...")
        
        # Method 1: Use find_cases to get cases with required tracks
        # This is much faster than checking each case individually
        logger.info("Finding cases with required ECG, PPG, and ART signals...")
        
        try:
            # Find cases that have all required tracks
            # Note: track names in find_cases don't include device prefix
            cases_with_ecg = set(vitaldb.find_cases('ECG_II'))
            cases_with_ppg = set(vitaldb.find_cases('PLETH'))
            cases_with_art = set(vitaldb.find_cases('ART'))
            
            # Get intersection - cases that have all three signals
            suitable_caseids = cases_with_ecg & cases_with_ppg & cases_with_art
            
            logger.info(f"Found {len(suitable_caseids)} cases with all required signals")
            
            # Get case metadata from API
            url = 'https://api.vitaldb.net/cases'
            df_all_cases = pd.read_csv(url)
            
            # Filter to only suitable cases
            df_cases = df_all_cases[df_all_cases['caseid'].isin(suitable_caseids)].copy()
            
            # Calculate duration and filter by minimum duration
            df_cases['duration_min'] = (df_cases['caseend'] - df_cases['casestart']) / 60
            df_cases = df_cases[df_cases['duration_min'] >= min_duration_min]
            
            logger.info(f"After filtering by duration (>={min_duration_min} min): {len(df_cases)} cases")
            
            # Save case list
            cases_file = os.path.join(self.metadata_dir, 'available_cases.csv')
            df_cases.to_csv(cases_file, index=False)
            logger.info(f"Saved case list to {cases_file}")
            
            return df_cases
            
        except Exception as e:
            logger.error(f"Error fetching cases: {e}")
            raise
    
    def download_case(self, caseid: int, use_enhanced: bool = False) -> Optional[Dict]:
        """
        Download data for a single case
        
        Args:
            caseid: Case ID to download
            use_enhanced: Whether to download enhanced parameters
            
        Returns:
            Dictionary with downloaded data
        """
        logger.info(f"Downloading case {caseid}...")
        
        try:
            # Load waveforms at 500 Hz (interval = 1/500 = 0.002 seconds)
            logger.info(f"Loading waveform data at 500 Hz...")
            waveform_vals = vitaldb.load_case(caseid, self.waveform_tracks, 1/500)
            
            waveform_data = {}
            if waveform_vals is not None and len(waveform_vals) > 0:
                for i, param in enumerate(self.waveform_params):
                    waveform_data[param] = waveform_vals[:, i]
                    logger.info(f"  {param}: {len(waveform_vals[:, i])} samples")
            else:
                logger.warning(f"Case {caseid}: No waveform data")
            
            # Load numeric data at 0.5 Hz (interval = 2 seconds)
            logger.info(f"Loading numeric data at 0.5 Hz...")
            tracks_to_load = self.numeric_tracks + (self.enhanced_tracks if use_enhanced else [])
            params_to_label = self.numeric_params + (self.enhanced_params if use_enhanced else [])
            
            numeric_vals = vitaldb.load_case(caseid, tracks_to_load, 2)
            
            numeric_data = {}
            if numeric_vals is not None and len(numeric_vals) > 0:
                for i, param in enumerate(params_to_label):
                    numeric_data[param] = numeric_vals[:, i]
                    logger.debug(f"  {param}: {len(numeric_vals[:, i])} samples")
            else:
                logger.warning(f"Case {caseid}: No numeric data")
            
            # Get case metadata from API
            import pandas as pd
            url = 'https://api.vitaldb.net/cases'
            df_cases = pd.read_csv(url)
            case_row = df_cases[df_cases['caseid'] == caseid]
            case_info = case_row.to_dict('records')[0] if len(case_row) > 0 else {}
            
            # Package data
            data_package = {
                'caseid': caseid,
                'waveforms': waveform_data,
                'numerics': numeric_data,
                'metadata': case_info,
                'download_timestamp': datetime.now().isoformat()
            }
            
            # Save raw data
            output_file = os.path.join(self.raw_data_dir, f'case_{caseid:05d}.npz')
            
            # Prepare arrays for npz
            save_dict = {'metadata': json.dumps(case_info)}
            
            # Add waveforms
            for key, val in waveform_data.items():
                save_dict[f'wave_{key.replace("/", "_")}'] = val
            
            # Add numerics
            for key, val in numeric_data.items():
                save_dict[f'num_{key.replace("/", "_")}'] = val
            
            np.savez_compressed(output_file, **save_dict)
            logger.info(f"Saved case {caseid} to {output_file}")
            
            return data_package
            
        except Exception as e:
            logger.error(f"Error downloading case {caseid}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def download_multiple_cases(self, case_list: List[int], use_enhanced: bool = False, 
                                max_cases: Optional[int] = None) -> None:
        """
        Download multiple cases
        
        Args:
            case_list: List of case IDs
            use_enhanced: Whether to download enhanced parameters
            max_cases: Maximum number of cases to download (None = all)
        """
        if max_cases:
            case_list = case_list[:max_cases]
        
        logger.info(f"Starting download of {len(case_list)} cases...")
        
        successful = 0
        failed = 0
        
        for i, caseid in enumerate(case_list, 1):
            logger.info(f"Processing case {i}/{len(case_list)}: {caseid}")
            
            # Check if already downloaded
            output_file = os.path.join(self.raw_data_dir, f'case_{caseid:05d}.npz')
            if os.path.exists(output_file):
                logger.info(f"Case {caseid} already downloaded, skipping...")
                successful += 1
                continue
            
            result = self.download_case(caseid, use_enhanced)
            
            if result:
                successful += 1
            else:
                failed += 1
            
            # Progress update
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(case_list)} - Success: {successful}, Failed: {failed}")
        
        logger.info(f"Download complete! Success: {successful}, Failed: {failed}")
        
        # Save download summary
        summary = {
            'total_cases': len(case_list),
            'successful': successful,
            'failed': failed,
            'timestamp': datetime.now().isoformat(),
            'use_enhanced': use_enhanced
        }
        
        with open(os.path.join(self.metadata_dir, 'download_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main execution function"""
    
    # Configuration
    MIN_DURATION_MIN = 30  # Minimum case duration in minutes
    MAX_CASES_TO_DOWNLOAD = 50  # Set to None to download all available cases
    USE_ENHANCED_PARAMS = False  # Set True to include hemodynamic parameters
    
    print("=" * 80)
    print("VitalDB Data Download for BP Estimation Research")
    print("=" * 80)
    
    # Initialize downloader
    downloader = VitalDBDownloader(output_dir='./data')
    
    # Step 1: Get available cases
    print("\n[Step 1] Fetching available cases...")
    df_cases = downloader.get_available_cases(min_duration_min=MIN_DURATION_MIN)
    
    if len(df_cases) == 0:
        logger.error("No suitable cases found!")
        return
    
    print(f"\nFound {len(df_cases)} suitable cases")
    print("\nCase Statistics:")
    print(f"  Age: {df_cases['age'].mean():.1f} ± {df_cases['age'].std():.1f} years")
    print(f"  Sex: {df_cases['sex'].value_counts().to_dict()}")
    print(f"  BMI: {df_cases['bmi'].mean():.1f} ± {df_cases['bmi'].std():.1f}")
    print(f"  Duration: {df_cases['duration_min'].mean():.1f} ± {df_cases['duration_min'].std():.1f} min")
    
    # Step 2: Select cases to download
    case_list = df_cases['caseid'].tolist()
    
    if MAX_CASES_TO_DOWNLOAD:
        print(f"\n[Step 2] Selecting {MAX_CASES_TO_DOWNLOAD} cases for download...")
        # Stratified sampling by age groups
        df_cases['age_group'] = pd.cut(df_cases['age'], bins=[0, 40, 60, 100], labels=['young', 'middle', 'old'])
        sampled = df_cases.groupby('age_group', group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_CASES_TO_DOWNLOAD // 3))
        )
        case_list = sampled['caseid'].tolist()[:MAX_CASES_TO_DOWNLOAD]
    else:
        print(f"\n[Step 2] Preparing to download all {len(case_list)} cases...")
    
    # Step 3: Download cases
    print(f"\n[Step 3] Downloading {len(case_list)} cases...")
    print(f"  Enhanced parameters: {USE_ENHANCED_PARAMS}")
    
    user_input = input("\nProceed with download? (y/n): ")
    if user_input.lower() != 'y':
        print("Download cancelled.")
        return
    
    downloader.download_multiple_cases(
        case_list=case_list,
        use_enhanced=USE_ENHANCED_PARAMS,
        max_cases=None
    )
    
    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"\nData saved to: {downloader.output_dir}")
    print(f"  Raw data: {downloader.raw_data_dir}")
    print(f"  Metadata: {downloader.metadata_dir}")
    print("\nNext steps:")
    print("  1. Run 'python data_exploration.py' to explore the downloaded data")
    print("  2. Run feature extraction pipeline")
    print("  3. Train BP estimation model")


if __name__ == "__main__":
    main()