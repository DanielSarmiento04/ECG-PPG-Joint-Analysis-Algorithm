"""
Data validation and quality assurance for physiological signals.
Addresses the lack of data validation in the original pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import logging
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class SignalQualityValidator:
    """Validates the quality of ECG and PPG signals."""
    
    def __init__(self, sampling_rate: int = 125):
        self.sampling_rate = sampling_rate
        self.min_signal_length = 50  # Minimum required signal length
        self.max_outlier_ratio = 0.15  # Maximum allowed outlier ratio
        
    def validate_signal_quality(self, signal_data: np.ndarray, signal_type: str) -> Dict[str, Any]:
        """
        Comprehensive signal quality assessment.
        
        Args:
            signal_data: 1D array of signal values
            signal_type: 'ECG' or 'PPG'
            
        Returns:
            Dict with quality metrics and pass/fail status
        """
        quality_metrics = {
            'signal_type': signal_type,
            'length': len(signal_data),
            'is_valid': True,
            'issues': []
        }
        
        # Check signal length
        if len(signal_data) < self.min_signal_length:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append(f'Signal too short: {len(signal_data)} < {self.min_signal_length}')
        
        # Check for NaN/infinite values
        nan_count = np.sum(np.isnan(signal_data))
        inf_count = np.sum(np.isinf(signal_data))
        
        if nan_count > 0:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append(f'Contains {nan_count} NaN values')
            
        if inf_count > 0:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append(f'Contains {inf_count} infinite values')
        
        # Check signal range (physiological plausibility)
        if not np.isnan(signal_data).all():
            signal_range = np.ptp(signal_data)  # Peak-to-peak
            signal_std = np.std(signal_data)
            
            quality_metrics['range'] = signal_range
            quality_metrics['std'] = signal_std
            quality_metrics['mean'] = np.mean(signal_data)
            
            # Signal-specific validations
            if signal_type.upper() == 'ECG':
                if signal_range < 0.1 or signal_range > 10:  # mV range
                    quality_metrics['issues'].append(f'ECG range suspicious: {signal_range:.3f}')
                    
            elif signal_type.upper() == 'PPG':
                if signal_std < 0.01:  # Too flat
                    quality_metrics['issues'].append(f'PPG signal too flat: std={signal_std:.4f}')
        
        # Check for saturation/clipping
        unique_values = len(np.unique(signal_data))
        if unique_values < len(signal_data) * 0.1:  # Less than 10% unique values
            quality_metrics['issues'].append(f'Signal may be saturated: only {unique_values} unique values')
        
        # Detect outliers using IQR method
        q1, q3 = np.percentile(signal_data, [25, 75])
        iqr = q3 - q1
        outlier_mask = (signal_data < q1 - 1.5 * iqr) | (signal_data > q3 + 1.5 * iqr)
        outlier_ratio = np.sum(outlier_mask) / len(signal_data)
        
        quality_metrics['outlier_ratio'] = outlier_ratio
        
        if outlier_ratio > self.max_outlier_ratio:
            quality_metrics['issues'].append(f'High outlier ratio: {outlier_ratio:.3f}')
        
        # Final validation
        if len(quality_metrics['issues']) > 0:
            quality_metrics['is_valid'] = False
        
        return quality_metrics
    
    def clean_signal(self, signal_data: np.ndarray, signal_type: str) -> np.ndarray:
        """
        Clean signal by removing outliers and interpolating missing values.
        """
        cleaned_signal = signal_data.copy()
        
        # Handle NaN values by interpolation
        if np.any(np.isnan(cleaned_signal)):
            # Linear interpolation for small gaps
            mask = np.isnan(cleaned_signal)
            if np.sum(mask) < len(cleaned_signal) * 0.3:  # Less than 30% missing
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 1:
                    cleaned_signal[mask] = np.interp(
                        np.where(mask)[0], 
                        valid_indices, 
                        cleaned_signal[valid_indices]
                    )
                else:
                    # If too few valid points, fill with median
                    cleaned_signal[mask] = np.nanmedian(cleaned_signal)
            else:
                logger.warning(f"Too many missing values ({np.sum(mask)}/{len(cleaned_signal)}), filling with median")
                cleaned_signal[mask] = np.nanmedian(cleaned_signal)
        
        # Remove extreme outliers (beyond 3 standard deviations)
        if not np.all(np.isnan(cleaned_signal)):
            mean_val = np.mean(cleaned_signal)
            std_val = np.std(cleaned_signal)
            outlier_mask = np.abs(cleaned_signal - mean_val) > 3 * std_val
            
            if np.any(outlier_mask):
                # Replace outliers with clipped values
                cleaned_signal[outlier_mask] = np.clip(
                    cleaned_signal[outlier_mask],
                    mean_val - 3 * std_val,
                    mean_val + 3 * std_val
                )
        
        return cleaned_signal


class PhysiologicalRangeValidator:
    """Validates physiological plausibility of blood pressure values."""
    
    def __init__(self):
        # Physiological ranges for blood pressure (mmHg)
        self.sbp_range = (70, 250)   # Systolic
        self.dbp_range = (40, 150)   # Diastolic
        self.pulse_pressure_range = (20, 120)  # SBP - DBP
        
    def validate_bp_values(self, sbp: float, dbp: float) -> Dict[str, Any]:
        """
        Validate blood pressure measurements for physiological plausibility.
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check individual ranges
        if not (self.sbp_range[0] <= sbp <= self.sbp_range[1]):
            validation_result['errors'].append(f'SBP out of range: {sbp} mmHg')
            validation_result['is_valid'] = False
            
        if not (self.dbp_range[0] <= dbp <= self.dbp_range[1]):
            validation_result['errors'].append(f'DBP out of range: {dbp} mmHg')
            validation_result['is_valid'] = False
        
        # Check pulse pressure
        if validation_result['is_valid']:
            pulse_pressure = sbp - dbp
            
            if pulse_pressure < self.pulse_pressure_range[0]:
                validation_result['warnings'].append(f'Low pulse pressure: {pulse_pressure} mmHg')
            elif pulse_pressure > self.pulse_pressure_range[1]:
                validation_result['warnings'].append(f'High pulse pressure: {pulse_pressure} mmHg')
            
            # SBP should be higher than DBP
            if sbp <= dbp:
                validation_result['errors'].append(f'SBP ({sbp}) <= DBP ({dbp})')
                validation_result['is_valid'] = False
        
        return validation_result


class DatasetValidator:
    """Main dataset validation orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.signal_validator = SignalQualityValidator()
        self.bp_validator = PhysiologicalRangeValidator()
        
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive dataset validation and cleaning.
        
        Args:
            df: DataFrame with structure [patient_id, ?, ?, SBP, DBP, PTT, PPG_signals, ECG_signals]
            
        Returns:
            Tuple of (cleaned_dataframe, validation_report)
        """
        logger.info("Starting comprehensive dataset validation...")
        
        validation_report = {
            'original_records': len(df),
            'removed_records': 0,
            'signal_issues': [],
            'bp_issues': [],
            'cleaned_records': 0
        }
        
        cleaned_records = []
        
        for idx, row in df.iterrows():
            record_valid = True
            record_issues = []
            
            try:
                # Extract signals (assuming columns 6-205 are PPG, 206-405 are ECG)
                ppg_signal = row.iloc[6:206].values.astype(float)
                ecg_signal = row.iloc[206:406].values.astype(float)
                
                # Extract BP values
                sbp = float(row.iloc[3])
                dbp = float(row.iloc[4])
                
                # Validate signals
                ppg_quality = self.signal_validator.validate_signal_quality(ppg_signal, 'PPG')
                ecg_quality = self.signal_validator.validate_signal_quality(ecg_signal, 'ECG')
                
                if not ppg_quality['is_valid']:
                    record_issues.extend([f"PPG: {issue}" for issue in ppg_quality['issues']])
                    record_valid = False
                    
                if not ecg_quality['is_valid']:
                    record_issues.extend([f"ECG: {issue}" for issue in ecg_quality['issues']])
                    record_valid = False
                
                # Validate BP values
                bp_validation = self.bp_validator.validate_bp_values(sbp, dbp)
                if not bp_validation['is_valid']:
                    record_issues.extend(bp_validation['errors'])
                    record_valid = False
                
                # If record is valid or can be cleaned, process it
                if record_valid or len(record_issues) <= 2:  # Allow minor issues
                    cleaned_row = row.copy()
                    
                    # Clean signals if needed
                    if not ppg_quality['is_valid'] and 'NaN' not in str(ppg_quality['issues']):
                        cleaned_ppg = self.signal_validator.clean_signal(ppg_signal, 'PPG')
                        cleaned_row.iloc[6:206] = cleaned_ppg
                        
                    if not ecg_quality['is_valid'] and 'NaN' not in str(ecg_quality['issues']):
                        cleaned_ecg = self.signal_validator.clean_signal(ecg_signal, 'ECG')
                        cleaned_row.iloc[206:406] = cleaned_ecg
                    
                    cleaned_records.append(cleaned_row)
                    validation_report['cleaned_records'] += 1
                else:
                    validation_report['removed_records'] += 1
                    validation_report['signal_issues'].append({
                        'record_id': idx,
                        'patient_id': row.iloc[0],
                        'issues': record_issues
                    })
                    
            except Exception as e:
                logger.error(f"Error processing record {idx}: {str(e)}")
                validation_report['removed_records'] += 1
                record_issues.append(f"Processing error: {str(e)}")
                validation_report['signal_issues'].append({
                    'record_id': idx,
                    'patient_id': row.iloc[0] if len(row) > 0 else 'unknown',
                    'issues': record_issues
                })
        
        # Create cleaned dataframe
        if cleaned_records:
            cleaned_df = pd.DataFrame(cleaned_records)
            cleaned_df.reset_index(drop=True, inplace=True)
        else:
            cleaned_df = pd.DataFrame()
            logger.warning("No valid records found after validation!")
        
        # Log validation summary
        logger.info(f"Validation complete: {validation_report['cleaned_records']}/{validation_report['original_records']} records retained")
        
        return cleaned_df, validation_report
    
    def generate_validation_report(self, validation_report: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        
        report = f"""
=== DATA VALIDATION REPORT ===

Summary:
- Original records: {validation_report['original_records']}
- Clean records: {validation_report['cleaned_records']}
- Removed records: {validation_report['removed_records']}
- Success rate: {validation_report['cleaned_records']/validation_report['original_records']*100:.1f}%

Signal Quality Issues:
"""
        
        if validation_report['signal_issues']:
            for issue in validation_report['signal_issues'][:10]:  # Show first 10
                report += f"- Patient {issue['patient_id']}: {', '.join(issue['issues'])}\n"
            
            if len(validation_report['signal_issues']) > 10:
                report += f"... and {len(validation_report['signal_issues']) - 10} more issues\n"
        else:
            report += "- No signal quality issues found\n"
        
        return report


def validate_and_clean_data(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for data validation and cleaning.
    
    Args:
        df: Raw dataset DataFrame
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (cleaned_dataframe, validation_report)
    """
    validator = DatasetValidator(config)
    return validator.validate_dataset(df)
