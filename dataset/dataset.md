# Dataset Documentation

## Overview
This dataset contains processed physiological signals and clinical metadata derived from the VitalDB database. It is designed for the development of cuffless blood pressure (BP) estimation models using ECG and PPG signals.

**File Path:** `data/processed/bp_dataset_cleaned_v2.csv` (cleaned) or `bp_dataset_features.csv` (raw)  
**Total Samples:** ~97,000+ (from 20 files), scalable to 800K+ (200 files)  
**Format:** CSV

## Pipeline Version History

### v2.1 (December 2025) - PPG Polarity & Artifact Fix
- **Critical fix: PPG polarity detection**: VitalDB PPG signals can be **inverted** (minimum = systolic peak). Added automatic polarity detection.
- **ECG artifact avoidance**: Skip first 25-50ms of each cycle to avoid R-peak electrical artifact bleeding into PPG.
- **Achieved physiological PTT-BP correlations**: 
  - `pat_ecg_ppg` vs SBP: **r = -0.23** (was near 0)
  - `pat_ecg_ppg` vs DBP: **r = -0.36** (was near 0)
- **100% valid extraction rate**: All cycles now produce valid features (no PTT range failures)

### v2.0 (December 2025) - Multi-Method Foot Detection
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
| Column | Unit | Description | Expected Range | BP Correlation |
| :--- | :--- | :--- | :--- | :--- |
| `pat_ecg_ppg` | ms | Pulse Arrival Time (R-peak to PPG foot) | 20 - 200 | **r = -0.23 (SBP), -0.36 (DBP)** |
| `pat_to_peak` | ms | Time from R-peak to PPG systolic peak | 50 - 500 | **r = -0.16 (SBP), -0.25 (DBP)** |
| `ptt_peak_to_foot` | ms | Same as `pat_ecg_ppg` (legacy name) | 20 - 200 | Same as pat_ecg_ppg |
| `ptt_peak_to_peak` | ms | Same as `pat_to_peak` (legacy name) | 50 - 500 | Same as pat_to_peak |
| `pat_to_maxslope` | ms | Time from R-peak to PPG max slope point | 5 - 200 | **r = +0.20 (SBP), +0.26 (DBP)** |
| `time_to_maxslope_t1` | ms | Time from PPG foot to max slope point | 0 - 100 | - |

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

### Current Dataset Statistics (v2.1 - 20 files test)
| Metric | Value |
| :--- | :--- |
| Total Samples | 97,848 |
| Unique Patients | 18 |
| Valid Extraction Rate | **100%** (no PTT range failures) |
| PTT zeros | **0%** |
| Mean SBP | 122 ± 22 mmHg |
| Mean DBP | 62 ± 11 mmHg |
| Cycle Correlation (quality) | 0.946 mean |

### PTT-BP Correlations Achieved (v2.1)
| Feature | SBP (r) | DBP (r) | SBP_delta (r) |
| :--- | :--- | :--- | :--- |
| `pat_ecg_ppg` | **-0.231** | **-0.360** | -0.342 |
| `pat_to_peak` | **-0.158** | **-0.245** | -0.254 |
| `pat_to_maxslope` | +0.196 | +0.256 | +0.174 |
| `hr_bpm` | -0.100 | -0.030 | +0.244 |

**Note:** Negative correlations for PAT features are physiologically expected (higher BP → stiffer arteries → faster pulse transit → lower PTT).

### Known Limitations
1. **Amplitude Ratio Detection**: Many samples have `amplitude_ratio_ra = 0.5` (fallback) indicating dicrotic notch detection is challenging. These samples are retained as PTT validity is unaffected.

2. **VitalDB-Specific Considerations**:
   - Surgical patients under anesthesia (affects vascular tone)
   - Active vasoactive medications may decouple PTT-BP relationship
   - PPG signals may be **inverted** depending on recording hardware
   - ECG artifact can bleed into PPG in first 25-50ms of cycle
   
3. **Per-Patient Variability**: Individual calibration is recommended for best results.

## Usage for Machine Learning

### Recommended Preprocessing
```python
import pandas as pd
from sklearn.model_selection import GroupKFold

# Load data
df = pd.read_csv('data/processed/bp_dataset_features.csv')

# Key features for BP estimation (based on correlation analysis)
timing_features = ['pat_ecg_ppg', 'pat_to_peak']  # Strong negative correlation with BP
morphology_features = ['max_upslope', 'slope_ratio', 'pulse_width_50']
hr_features = ['hr_bpm']

# Use per-patient normalized features for cross-patient models
norm_features = ['pat_ecg_ppg_norm', 'pat_to_peak_norm', 'hr_bpm_norm']

# Always split by patient to test generalization
X = df[timing_features + morphology_features + hr_features]
y = df['sbp_reference']

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['patient_id']):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train your model here
```

### Important Notes
- **Patient-wise splitting is critical**: Do not use random splits as this causes data leakage
- **PAT features show expected inverse relationship**: `pat_ecg_ppg` (r=-0.23 SBP, r=-0.36 DBP) is the strongest predictor
- **Per-patient normalized features**: Use `*_norm` columns for cross-patient generalization
- **Delta features**: Use `*_delta` columns for within-patient BP change prediction

## Files Description

| File | Description |
| :--- | :--- |
| `bp_dataset_raw.csv` | All extracted features before cleaning (267K samples) |
| `bp_dataset_cleaned_v2.csv` | Cleaned dataset with invalid samples removed (106K samples) |
| `bp_dataset_features.csv` | Legacy file (same as raw) |
| `quality_report_before.json` | Quality metrics before cleaning |
| `quality_report_after.json` | Quality metrics after cleaning |
