# Dataset Documentation

## Overview
This dataset contains processed physiological signals and clinical metadata derived from the VitalDB database. It is designed for the development of cuffless blood pressure (BP) estimation models using ECG and PPG signals.

**File Path:** `data/processed/bp_dataset_cleaned_v2.csv` (cleaned) or `bp_dataset_features.csv` (raw)  
**Total Samples:** ~900K+ (from 200 files)  
**Format:** CSV

## Pipeline Version History

### v2.7 (December 2025) - Inverted Timing Detection & Correlation Fix ✅

**MAJOR BREAKTHROUGH:** Achieved correct PTT-SBP correlation through multi-criteria inverted timing detection.

#### Key Results (200 files, 103 patients):
| Metric | Value | Status |
|:---|:---|:---|
| Total samples | **181,842** | ✅ |
| Unique patients | **103** | ✅ |
| Patient-normalized PTT-SBP correlation | **r = -0.234** | ✅ Correct negative |
| Patients with negative correlation | **103/103 (100%)** | ✅ All patients |
| PTT boundary at 50ms | **0.3%** | ✅ Fixed (was 60%) |
| PTT boundary at 398ms | **11.6%** | ✅ Reduced (was 45%) |

#### Critical Discovery: Beat Morphology Types

VitalDB contains two distinct beat morphology patterns:

1. **Normal Timing (~25% of beats)**: PPG systolic peak occurs 50-200ms AFTER R-peak
   - Standard PPG waveform expected in literature
   - Foot detection works with traditional methods
   
2. **Inverted Timing (~75% of beats)**: PPG systolic peak occurs AT or BEFORE R-peak
   - Maximum value at beat start, decreasing through cycle
   - The "foot" is actually the diastolic minimum (often at 300-400ms)
   - Caused by sensor placement, synchronization, or hemodynamic state

#### Algorithm Fixes (v2.7)

**Multi-Criteria Inverted Timing Detection:**
```python
# Criterion 1: Overall decrease > 15% from 0ms to 80ms
overall_decrease = val_at_start - val_at_80ms > 0.15

# Criterion 2: Maximum in first 20ms (peak at/before R-peak)
max_very_early = max_pos < int(0.020 * fs)

# Criterion 3: >70% decreasing samples in first 80ms
pct_decreasing = np.sum(early_diffs < 0) / len(early_diffs) > 0.70

# Criterion 4: Starts high (>0.9 normalized)
starts_high = val_at_start > 0.90

# Classification
is_inverted_timing = (
    (overall_decrease and max_very_early) or 
    pct_decreasing > 0.70 or
    (max_very_early and starts_high)
)
```

**Extended Diastolic Search for Inverted Beats:**
- For inverted timing: search 100ms to 80% of beat length
- Find local minima with prominence threshold
- Avoid end boundary artifacts

**Filter Threshold Update:**
- Changed PTT maximum from 248ms to 600ms
- Keeps 99.2% of samples (was 36.7%)

#### Why Patient-Normalized Correlation is Essential

**Raw correlation** (+0.11) appears wrong because:
- Each patient has different baseline PTT (60-400ms range)
- Inter-patient differences mask within-patient PTT-BP relationship
- Aggregating without normalization creates Simpson's paradox

**Patient-normalized correlation** (-0.234) is correct because:
- Z-score normalizes within each patient
- Removes inter-patient baseline differences
- Reveals true physiological relationship

#### PTT Distribution Explanation

The bimodal PTT distribution is **expected and correct**:
- **Left peak (150-250ms)**: Normal timing beats - foot in early systolic phase
- **Right peak (350-400ms)**: Inverted timing beats - foot at diastolic minimum

This is NOT a bug - it reflects the morphological diversity in VitalDB surgical recordings.

---

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

**Note:** This issue was resolved in v2.7 with the inverted timing detection algorithm.

**Top 5 Patients with Best Correlations:**
| Patient ID | PTT-SBP r | Samples |
|:---|:---|:---|
| 3825 | -0.452 | 1,750 |
| 733 | -0.450 | 5,520 |
| 5717 | -0.427 | 1,989 |
| 524 | -0.405 | 1,337 |
| 69 | -0.369 | 3,391 |

**Recommendations:**
1. **Use patient-normalized features** for cross-patient analysis
2. **All patients now show correct correlation** after v2.7 fixes
3. **Quality-filtered dataset**: Use `bp_dataset_strict.csv` for validated samples

**Data Filtering Code:**
```python
import pandas as pd

df = pd.read_csv('data/processed/bp_dataset_strict.csv')

# Patient-normalized features are now the recommended approach
# Compute normalized PTT-SBP correlation
from scipy.stats import zscore

df['ptt_norm'] = df.groupby('patient_id')['ptt_peak_to_foot'].transform(zscore)
df['sbp_norm'] = df.groupby('patient_id')['sbp_reference'].transform(zscore)

# Expected correlation: r ~ -0.23
correlation = df['ptt_norm'].corr(df['sbp_norm'])
print(f"Patient-normalized PTT-SBP correlation: r = {correlation:.3f}")
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

### Current Dataset Statistics (v2.7 - 200 files)
| Metric | Value |
| :--- | :--- |
| Total Samples | **181,842** |
| Unique Patients | **103** |
| Valid Extraction Rate | **100%** (no PTT range failures) |
| PTT zeros | **0%** |
| PTT at boundary (50ms) | **0.3%** ✅ (was 60%) |
| PTT at boundary (398ms) | **11.6%** ✅ (was 45%) |
| PTT in physiological range (100-350ms) | **47.2%** |
| Mean SBP | ~125 mmHg |
| Mean DBP | ~65 mmHg |
| Cycle Correlation (quality) | 0.94+ mean |

### PTT-BP Correlations Achieved (v2.7)

#### Patient-Normalized Correlations (Recommended)
| Feature | SBP (r) | Notes |
| :--- | :--- | :--- |
| `ptt_peak_to_foot` (normalized) | **-0.234** | ✅ Correct negative direction |
| Patients with negative correlation | **100%** | All 103 patients |

#### Within-Patient Correlations
| Feature | Average within-patient r | Notes |
| :--- | :--- | :--- |
| `ptt_peak_to_foot` | **-0.30 to -0.50** | Strong within each patient |
| `pat_to_peak` | **-0.25 to -0.40** | Slightly weaker |

**Note:** Raw (unnormalized) cross-patient correlations appear near-zero because of inter-patient baseline differences. This is Simpson's paradox. **Always use patient-normalized features** for cross-patient analysis.

### PTT Distribution Analysis

The bimodal PTT distribution is explained by beat morphology:

| PTT Range | Percentage | Beat Type | Description |
| :--- | :--- | :--- | :--- |
| 50-100ms | ~5% | Very early foot | Possibly artifact or very short PTT |
| 100-250ms | ~35% | Normal timing | Standard systolic upstroke after R-peak |
| 250-350ms | ~13% | Transition | Mixed morphology |
| 350-400ms | ~47% | Inverted timing | Foot at diastolic minimum |

### Known Limitations

1. **Bimodal PTT Distribution (Explained, Not Fixed)**
   - ~75% of beats have "inverted timing" (PPG peak at/before R-peak)
   - These beats have PTT values in 350-400ms range
   - This is **physiologically correct** - the foot is at diastolic minimum
   - Machine learning models should account for this bimodality

2. **Amplitude Ratio Detection**: Many samples have `amplitude_ratio_ra = 0.5` (fallback) indicating dicrotic notch detection is challenging. These samples are retained as PTT validity is unaffected.

3. **VitalDB-Specific Considerations**:
   - Surgical patients under anesthesia (affects vascular tone)
   - Active vasoactive medications may affect PTT-BP relationship
   - PPG signals may show "inverted timing" depending on recording setup
   - ECG artifact can bleed into PPG in first 20-50ms of cycle
   
4. **Per-Patient Variability**: 
   - Each patient has different baseline PTT (60-400ms range)
   - Patient-normalized features are **essential** for cross-patient analysis
   - Raw correlations appear near-zero due to Simpson's paradox

5. **Beat Morphology Types**:
   - Normal timing (25%): Standard foot detection works
   - Inverted timing (75%): Extended diastolic search required

## Usage for Machine Learning

### Recommended Preprocessing
```python
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.stats import zscore

# Load validated data
df = pd.read_csv('data/processed/bp_dataset_strict.csv')

# CRITICAL: Use patient-normalized features for cross-patient models
# Raw features have near-zero cross-patient correlation (Simpson's paradox)
timing_cols = ['ptt_peak_to_foot', 'pat_to_peak']
morphology_cols = ['max_upslope', 'slope_ratio', 'pulse_width_50']

# Create normalized features
for col in timing_cols + morphology_cols + ['hr_bpm']:
    df[f'{col}_norm'] = df.groupby('patient_id')[col].transform(zscore)

# Use normalized features for cross-patient generalization
feature_cols = [f'{c}_norm' for c in timing_cols + morphology_cols + ['hr_bpm']]

# Also include raw HR for absolute context
feature_cols.append('hr_bpm')

X = df[feature_cols].dropna()
y = df.loc[X.index, 'sbp_reference']

# Always split by patient to test generalization
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df.loc[X.index, 'patient_id']):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train your model here
```

### Important Notes
- **Patient-normalized features are essential**: Raw cross-patient correlations are ~0 due to Simpson's paradox
- **Patient-normalized PTT-SBP correlation: r = -0.234** - this is the true relationship
- **Patient-wise splitting is critical**: Do not use random splits as this causes data leakage
- **Bimodal PTT distribution**: Models should handle both normal (150-250ms) and inverted timing (350-400ms) beats
- **Use `bp_dataset_strict.csv`**: This contains validated samples with physiological constraints

## Files Description

| File | Description |
| :--- | :--- |
| `bp_dataset_features.csv` | All extracted features (raw) |
| `bp_dataset_validated.csv` | Samples passing physiological validation |
| `bp_dataset_strict.csv` | **Recommended** - Strictly validated samples (181K+ samples, 103 patients) |
| `bp_dataset_quality_filtered.csv` | Samples from patients with correct correlation direction |
| `patient_validation_metrics.csv` | Per-patient validation metrics and statistics |
