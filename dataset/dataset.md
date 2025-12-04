# Dataset Documentation

## Overview
This dataset contains processed physiological signals and clinical metadata derived from the VitalDB database. It is designed for the development of cuffless blood pressure (BP) estimation models using ECG and PPG signals.

**File Path:** `data/processed/bp_dataset_cleaned_v2.csv` (cleaned) or `bp_dataset_features.csv` (raw)  
**Total Samples:** ~900K+ (from 200 files)  
**Format:** CSV

## Pipeline Version History

### v2.4 (December 2025) - Per-Patient Correlation Analysis

**Critical Discovery:** The near-zero overall PTT-SBP correlation (r=0.027) hides a **bimodal pattern**:
- **41 patients (23%)** have the **correct negative correlation** (r < -0.1)
- **48 patients (27%)** have the **wrong positive correlation** (r > +0.1)
- Remaining patients have near-zero correlation

**Best vs Worst Patient Comparison:**
| Metric | Correct (r < -0.2) | Wrong (r > +0.2) |
|:---|:---|:---|
| Count | 15 patients | 17 patients |
| PTT mean | 69.0 ± 17.3 ms | 63.0 ± 15.8 ms |
| **PTT std** | **24.1 ms** | **14.5 ms** |
| Amp mean | 0.539 | 0.568 |

**Key Insight:** Patients with **wrong positive correlation** have **40% lower PTT variability** (std=14.5ms vs 24.1ms). This suggests the foot detection is stable but detecting the **wrong feature** (possibly due to PPG polarity issues).

**Top 5 Patients with Best Correlations:**
| Patient ID | PTT-SBP r | Samples |
|:---|:---|:---|
| 3825 | -0.452 | 1,750 |
| 733 | -0.450 | 5,520 |
| 5717 | -0.427 | 1,989 |
| 524 | -0.405 | 1,337 |
| 69 | -0.369 | 3,391 |

**Recommendations:**
1. **Filter patients by PTT-SBP correlation** during training - use patients with r < -0.1
2. **Use per-patient sign detection** to correct polarity issues  
3. **Use patients with PTT std > 20ms** for better signal quality
4. **Quality-filtered dataset**: Use `bp_dataset_quality_filtered.csv` which contains only patients with correct negative PTT-SBP correlation

**Data Filtering Code:**
```python
import pandas as pd

df = pd.read_csv('data/processed/bp_dataset_features.csv')

# Compute per-patient PTT-SBP correlation
patient_corrs = {}
for pid in df['patient_id'].unique():
    p_df = df[df['patient_id'] == pid]
    if len(p_df) >= 100:
        patient_corrs[pid] = p_df['ptt_peak_to_foot'].corr(p_df['sbp_reference'])

# Filter to patients with correct negative correlation
good_pids = [pid for pid, r in patient_corrs.items() if r < -0.1]
df_quality = df[df['patient_id'].isin(good_pids)]

# Expected result: ~41 patients, ~185K samples, r ~ -0.115
```

### v2.3 (December 2025) - Analysis & Insights

**Key Finding:** Absolute PTT values have **near-zero cross-patient correlation** with BP because each patient has a different baseline PTT due to:
- Individual arterial stiffness and compliance
- Arm length and sensor placement variations
- Hemodynamic state differences (anesthesia, medications)

**What Works:**
| Feature Type | Best Feature | Correlation | Use Case |
|:---|:---|:---|:---|
| Per-patient normalized | `hr_bpm_norm` | r=+0.130 SBP | Cross-patient models |
| Per-patient normalized | `systolic_duration_tsd_norm` | r=-0.099 SBP | Cross-patient models |
| Within-patient delta | `pat_to_maxslope_delta` | r=-0.092 SBP_delta | BP change detection |
| Within-patient raw | Patient 2656: PTT-SBP | r=-0.340 | Individual calibration |

**Recommendations:**
1. **Use normalized features** (`*_norm`) for cross-patient generalization
2. **Use delta features** (`*_delta`) for BP change prediction
3. **Per-patient calibration** is essential for absolute BP estimation
4. Focus on **relative changes** rather than absolute PTT values

### v2.2 (December 2025) - Artifact Removal & Boundary Fix
**Goal:** Fix boundary clustering in PTT and amplitude ratio distributions.

**Techniques Applied:**
1. **Spike artifact removal (`remove_spikes()`)**: VitalDB PPG signals contain ECG R-peak electrical artifacts that bleed into the PPG channel. These appear as sharp spikes (-198 to +91 range vs normal ~30-60). Using MAD-based outlier detection (threshold=4 MAD), spikes are identified and replaced with interpolated values.

2. **Reduced minimum PTT boundary**: Changed from 80ms to 40ms to allow natural variability. Physical minimum is ~40ms (arm length 0.6m ÷ max pulse wave velocity 15 m/s).

3. **Reduced ECG artifact skip region**: Changed from 80ms to 40ms. Previous setting was too aggressive and forcing all foot detections to hit the boundary.

4. **Relaxed validation bounds**: PAT range 30-400ms (permissive) to allow VitalDB surgical patient variability.

**Remaining Challenges:**
- 19.6% of PTT at exactly 40ms (boundary clustering persists)
- 5.7% amplitude ratio near 0, 12.4% near 1
- **Root cause:** Foot detection consensus algorithm over-smooths natural variability in some signals

### v2.1 (December 2025) - PPG Polarity & Artifact Fix
- **Critical fix: PPG polarity detection**: VitalDB PPG signals can be **inverted** (minimum = systolic peak). Added automatic polarity detection by comparing first quarter mean vs second quarter mean.
- **ECG artifact avoidance**: Skip first 25-50ms of each cycle to avoid R-peak electrical artifact bleeding into PPG.
- **Multi-method foot detection with consensus**: 5 algorithms (second derivative, intersecting tangent, threshold crossing, minimum detection, 5% rise method) with median consensus.
- **Amplitude ratio fix**: `find_systolic_peak()` now returns value from normalized signal; amplitude_ratio calculation uses consistent `ppg_norm` values and `abs()` for inverted signals.

### v2.0 (December 2025) - Multi-Method Foot Detection
- **Fixed foot detection algorithm**: Replaced single-method approach with multi-method consensus
- **Eliminated 52% PTT zeros**: Previous version had 52.4% of `ptt_peak_to_peak` = 0 due to detection failures
- **Boundary check**: If foot_idx at minimum boundary, use alternative 5% rise method

### v1.0 (November 2025) - Initial Release
- Basic wavelet filtering and R-peak detection
- Known issue: 93% of PTT values were invalid

## Feature Descriptions

### 1. Target Variables (Ground Truth)
| Column | Unit | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| `sbp_reference` | mmHg | Systolic Blood Pressure (invasive arterial line) | 60 - 220 |
| `dbp_reference` | mmHg | Diastolic Blood Pressure (invasive arterial line) | 30 - 130 |

### 2. Timing Features (PTT/PAT) - USE NORMALIZED VERSIONS
| Column | Unit | Description | Expected Range | Cross-Patient r |
| :--- | :--- | :--- | :--- | :--- |
| `pat_ecg_ppg` | ms | Pulse Arrival Time (R-peak to PPG foot) | 40 - 250 | ~0 (use `_norm`) |
| `pat_to_peak` | ms | Time from R-peak to PPG systolic peak | 60 - 500 | ~0 (use `_norm`) |
| `ptt_peak_to_foot` | ms | Same as `pat_ecg_ppg` (legacy name) | 40 - 250 | ~0 (use `_norm`) |
| `pat_to_maxslope` | ms | Time from R-peak to PPG max slope point | 5 - 200 | ~0 (use `_norm`) |
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
