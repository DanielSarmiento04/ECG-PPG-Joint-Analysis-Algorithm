# Dataset Documentation

## Overview
This dataset contains processed physiological signals and clinical metadata derived from the VitalDB database. It is designed for the development of cuffless blood pressure (BP) estimation models using ECG and PPG signals.

**File Path:** `data/processed/bp_dataset_cleaned_v2.csv` (cleaned) or `bp_dataset_raw.csv` (raw)  
**Total Samples:** ~106,000 (cleaned), ~267,000 (raw)  
**Patients:** 52  
**Format:** CSV

## Pipeline Version History

### v2.0 (December 2025) - Major Feature Extraction Fix
- **Fixed foot detection algorithm**: Replaced single-method approach with multi-method consensus (second derivative, intersecting tangent, threshold crossing, minimum detection)
- **Eliminated 52% PTT zeros**: Previous version had 52.4% of `ptt_peak_to_peak` = 0 due to detection failures
- **Improved data retention**: From 1.5% to 39.7% after cleaning
- **Relaxed amplitude ratio filtering**: Dicrotic notch detection (amplitude_ratio = 1.0) no longer filters samples

### v1.0 (November 2025) - Initial Release
- Basic wavelet filtering and R-peak detection
- Known issue: 93% of PTT values were invalid

## Feature Descriptions

### 1. Target Variables (Ground Truth)
| Column | Unit | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| `sbp_reference` | mmHg | Systolic Blood Pressure (invasive arterial line) | 60 - 220 |
| `dbp_reference` | mmHg | Diastolic Blood Pressure (invasive arterial line) | 30 - 130 |

### 2. Timing Features (PTT/PAT)
| Column | Unit | Description | Expected Range |
| :--- | :--- | :--- | :--- |
| `pat_ecg_ppg` | ms | Pulse Arrival Time (R-peak to PPG foot) | 10 - 300 |
| `pat_to_peak` | ms | Time from R-peak to PPG systolic peak | 50 - 400 |
| `ptt_peak_to_foot` | ms | Same as `pat_ecg_ppg` (legacy name) | 10 - 300 |
| `ptt_peak_to_peak` | ms | Same as `pat_to_peak` (legacy name) | 50 - 400 |
| `ptt_peak_to_maxslope` | ms | Time from R-peak to PPG max slope point | 5 - 200 |
| `time_to_maxslope_t1` | ms | Time from PPG foot to max slope point | 0 - 100 |

### 3. Morphology Features
| Column | Unit | Description |
| :--- | :--- | :--- |
| `amplitude_ratio_ra` | ratio | Ratio of dicrotic notch to systolic peak (0-1). **Note:** Value of 1.0 indicates detection failure |
| `systolic_duration_tsd` | ms | Duration of the systolic phase (foot to dicrotic notch) |
| `diastolic_duration_tfd` | ms | Duration of the diastolic phase (dicrotic notch to next foot) |
| `max_upslope` | units/s | Maximum upward slope of PPG |
| `max_downslope` | units/s | Maximum downward slope of PPG |
| `slope_ratio` | ratio | Ratio of upslope to downslope |
| `reflection_index` | ratio | Dicrotic wave reflection index |
| `pulse_width_50` | ms | Width of PPG pulse at 50% amplitude |
| `crest_time` | ms | Time to peak from foot |

### 4. Heart Rate & Quality
| Column | Unit | Description |
| :--- | :--- | :--- |
| `hr_bpm` | bpm | Instantaneous Heart Rate derived from RR interval |
| `cycle_correlation` | 0-1 | Quality metric: correlation of current PPG cycle with template |

### 5. Statistical Features
| Column | Description |
| :--- | :--- |
| `stat_mean`, `stat_std` | PPG signal statistics |
| `stat_skew`, `stat_kurtosis` | Higher-order moments |
| `stat_power_cardiac`, `stat_power_low` | Frequency domain power |
| `stat_power_ratio` | Cardiac/low frequency power ratio |

### 6. Clinical Metadata
| Column | Type | Description |
| :--- | :--- | :--- |
| `patient_id` | int | Unique patient identifier |
| `session_id` | string | Recording session (case file) |
| `age` | years | Patient age |
| `sex` | M/F | Biological sex |
| `bmi` | kg/m² | Body Mass Index |
| `position` | categorical | Patient position (Supine, Lithotomy, etc.) |
| `approach` | categorical | Surgical approach (Open, Videoscopic, Robotic) |
| `aline1` | categorical | Arterial line location |
| `dx` | categorical | Primary diagnosis |
| `opname` | categorical | Operation name |
| `preop_ecg` | categorical | Pre-operative ECG status |

## Data Quality Summary

### Current Dataset Statistics (v2.0)
| Metric | Raw | Cleaned |
| :--- | :--- | :--- |
| Total Samples | 267,267 | 106,164 |
| Patients | 52 | 52 |
| PTT zeros | 52.4% | 0% |
| Amplitude = 1.0 | 76.5% | 70.4% |
| PTT median | 80 ms | 100 ms |
| Mean SBP | 120 ± 18 mmHg | 123 ± 21 mmHg |
| Mean DBP | 67 ± 11 mmHg | 68 ± 12 mmHg |

### Known Limitations
1. **Weak PTT-BP Correlation**: The overall PTT-SBP correlation is weak (r ≈ -0.01 to -0.09). This is likely due to:
   - Surgical patients under anesthesia (affects vascular tone)
   - Medications that decouple the PTT-BP relationship
   - Limited within-patient BP variation during stable anesthesia
   
2. **Amplitude Ratio Detection**: 70% of samples have `amplitude_ratio_ra = 1.0` indicating dicrotic notch detection failure. These samples are retained as this doesn't affect PTT validity.

3. **Per-Patient Variability**: Some patients show expected negative PTT-BP correlation (up to r = -0.53), while others show positive correlation, suggesting individual calibration is needed.

## Usage for Machine Learning

### Recommended Preprocessing
```python
# Load data
df = pd.read_csv('data/processed/bp_dataset_cleaned_v2.csv')

# Key features for BP estimation
timing_features = ['pat_ecg_ppg', 'pat_to_peak', 'time_to_maxslope_t1']
morphology_features = ['max_upslope', 'slope_ratio', 'pulse_width_50']
hr_features = ['hr_bpm']

# Always split by patient to test generalization
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['patient_id']):
    # Train/test split
    pass
```

### Important Notes
- **Patient-wise splitting is critical**: Do not use random splits as this causes data leakage
- **Consider per-patient calibration**: Due to weak global correlation, individual calibration may be necessary
- **HR is a strong predictor**: Heart rate often correlates more strongly with BP than PTT in this dataset

## Files Description

| File | Description |
| :--- | :--- |
| `bp_dataset_raw.csv` | All extracted features before cleaning (267K samples) |
| `bp_dataset_cleaned_v2.csv` | Cleaned dataset with invalid samples removed (106K samples) |
| `bp_dataset_features.csv` | Legacy file (same as raw) |
| `quality_report_before.json` | Quality metrics before cleaning |
| `quality_report_after.json` | Quality metrics after cleaning |
