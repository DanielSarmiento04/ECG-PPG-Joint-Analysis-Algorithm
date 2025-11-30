# Dataset Documentation

## Overview
This dataset contains processed physiological signals and clinical metadata derived from the VitalDB database. It is designed for the development of cuffless blood pressure (BP) estimation models using ECG and PPG signals.

**File Path:** `data/processed/bp_dataset_features.csv`
**Total Samples:** ~100,000+ (in current batch)
**Format:** CSV

## Feature Descriptions

### 1. Target Variables (Ground Truth)
| Column | Unit | Description | Valid Range (approx) |
| :--- | :--- | :--- | :--- |
| `sbp_reference` | mmHg | Systolic Blood Pressure (invasive arterial line) | 50 - 250 |
| `dbp_reference` | mmHg | Diastolic Blood Pressure (invasive arterial line) | 30 - 150 |

### 2. Physiological Features (Derived from PPG/ECG)
| Column | Unit | Description |
| :--- | :--- | :--- |
| `ptt_peak_to_peak` | ms | Pulse Transit Time (R-peak to PPG systolic peak) |
| `ptt_peak_to_foot` | ms | Pulse Transit Time (R-peak to PPG foot) |
| `ptt_peak_to_maxslope` | ms | Pulse Transit Time (R-peak to PPG max slope point) |
| `amplitude_ratio_ra` | ratio | Ratio of systolic peak height to dicrotic notch height |
| `systolic_duration_tsd` | ms | Duration of the systolic phase (foot to dicrotic notch) |
| `diastolic_duration_tfd` | ms | Duration of the diastolic phase (dicrotic notch to next foot) |
| `time_to_maxslope_t1` | ms | Time from PPG foot to max slope point |
| `hr_bpm` | bpm | Instantaneous Heart Rate derived from RR interval |
| `cycle_correlation` | 0-1 | Quality metric: correlation of current PPG cycle with template |

### 3. Clinical Metadata (Categorical & Demographics)
These variables provide context about the patient's state and surgical environment, which significantly influence BP hemodynamics.

#### Demographics
- **`age`** (years): Patient age.
- **`sex`** (M/F): Biological sex.
- **`bmi`** (kg/m²): Body Mass Index.

#### Surgical Context (Categorical)
| Column | Unique Values | Description |
| :--- | :--- | :--- |
| **`position`** | 5 | Patient position during surgery. Affects hydrostatic pressure. <br>Values: `Supine`, `Left lateral decubitus`, `Lithotomy`, `Right lateral decubitus`, `Reverse Trendelenburg` |
| **`approach`** | 3 | Surgical approach. <br>Values: `Open`, `Videoscopic`, `Robotic` |
| **`aline1`** | 3 | Location of the arterial line catheter (BP measurement site). <br>Values: `Right radial`, `Left radial`, `Right dorsalis pedis` |
| **`preop_ecg`** | 4+ | Pre-operative ECG diagnosis. <br>Values: `Normal Sinus Rhythm`, `Incomplete right bundle branch block`, etc. |
| **`dx`** | 50+ | Primary diagnosis (e.g., `Early gastric cancer`, `Lung cancer`). |
| **`opname`** | 40+ | Name of the operation (e.g., `Lung lobectomy`, `Gastrectomy`). |

## Data Quality Notes
- **Missing Values:** The processed dataset has been filtered to remove rows with missing critical values (`NaN` counts are 0 in the sampled batch).
- **Signal Quality:** Only cycles with a `cycle_correlation` > 0.7 (configurable) are included.
- **BP Range:** Extreme BP values (outliers) may still be present and should be filtered during model training (e.g., SBP > 300 or < 20).

## Usage for Machine Learning
- **Normalization:** 
    - Numerical features (`ptt`, `age`, `bmi`) should be scaled (e.g., `StandardScaler`).
    - Target variables (`sbp`, `dbp`) generally do not need scaling for regression, but it can help convergence.
- **Encoding:** 
    - Categorical features (`sex`, `position`, etc.) must be encoded (e.g., `OneHotEncoder` or `Embedding` layers).
- **Splitting:** 
    - **Crucial:** Split data by `patient_id` (GroupKFold) to ensure the model generalizes to new patients and doesn't just memorize a specific patient's physiology.

## Troubleshooting
- **Training Freeze:** If training freezes, it is likely due to the large dataset size overloading the DataLoader or memory.
    - **Solution:** Reduce `BATCH_SIZE`, set `num_workers=0` in DataLoader, or use a subset of the data for debugging.
