# Dataset Analysis: PTT-Based Blood Pressure Estimation

## Executive Summary

**This is a DATASET problem, not a TRAINING problem.**

The VitalDB surgical dataset has fundamental limitations that prevent achieving AAMI-compliant blood pressure estimation using PTT/PAT features alone. However, **with per-patient calibration**, the dataset can achieve **61-83% AAMI compliance** for SBP and **91-98% for DBP**.

| Problem Type | Verdict |
|-------------|---------|
| **Dataset Task** | ✅ The core issue - features lack cross-patient signal |
| **Training Task** | ❌ Cannot fix fundamental data limitations |
| **Calibration Task** | ✅ Per-patient calibration enables AAMI compliance |

### Key Results (Updated Dec 8, 2025)

| Approach | SBP AAMI | DBP AAMI | Notes |
|----------|----------|----------|-------|
| Global Model (no calibration) | **0%** | ~13% | Cross-patient R² = -0.01 |
| EMA Calibration (α=0.1) | **61%** | 91% | Recommended baseline |
| EMA Calibration (α=0.2) | **79%** | 97% | Better for stable patients |
| EMA Calibration (α=0.3) | **83%** | 98% | Fastest adaptation |
| Hybrid (EMA + Features) | **47%** | - | Features hurt more than help! |

**Critical Insight**: Pure EMA calibration (using recent BP values) **outperforms** feature-based prediction. The features add noise rather than signal.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Normalization Fix (Dec 8, 2025)](#2-normalization-fix-dec-8-2025)
3. [The Five Fundamental Problems](#3-the-five-fundamental-problems)
4. [Detailed Analysis](#4-detailed-analysis)
5. [Calibration Results](#5-calibration-results)
6. [Why This is a Dataset Problem](#6-why-this-is-a-dataset-problem)
7. [Recommendations](#7-recommendations)
8. [Running the Analysis](#8-running-the-analysis)

---

## 1. Dataset Overview

### Source
- **Database**: VitalDB (surgical patients)
- **File**: `src/data/bp_dataset_strict.csv`

### Statistics
| Metric | Value |
|--------|-------|
| Total samples | 181,842 |
| Number of patients | 103 |
| Samples per patient (mean) | 1,765 |
| Samples per patient (range) | 106 - 10,270 |
| SBP range | 50 - 250 mmHg |
| SBP mean ± std | 119.6 ± 21.0 mmHg |
| DBP mean ± std | 65.2 ± 13.0 mmHg |

### Context
Data collected during **surgery** with:
- Invasive arterial BP monitoring (reference)
- ECG and PPG waveforms
- Extracted timing features (PTT, PAT, HR, etc.)

---

## 2. Normalization Fix (Dec 8, 2025)

### What Changed
The `_norm` features have been **corrected**. Previously, per-patient normalization was computed before filtering, causing biased statistics.

### Before vs After
| Metric | Before | After |
|--------|--------|-------|
| Per-patient _norm mean range | [0.00, 1.13] | **[0.00, 0.00]** ✅ |
| Per-patient _norm std range | [0.26, 1.27] | **[1.00, 1.00]** ✅ |

### Verification
```python
# All _norm features now have exact mean=0, std=1 per patient
pat_ecg_ppg_norm: mean=[-0.0000,0.0000], std=[1.000,1.000]
ptt_peak_to_foot_norm: mean=[-0.0000,0.0000], std=[1.000,1.000]
hr_bpm_norm: mean=[-0.0000,0.0000], std=[1.000,1.000]
```

### Impact on Training
- Removes data leakage from improper normalization
- `*_delta` features have also been recomputed
- **Cross-patient R² remains -0.01 to -0.04** (this is NOT a bug—it reflects fundamental weak signal)

---

## 3. The Five Fundamental Problems

### Problem 1: Weak Cross-Patient Signal (R² ≈ 0.07)

**Features explain only 7% of BP variation across patients.**

| Feature | Correlation with SBP Dev | R² |
|---------|-------------------------|-----|
| hr_bpm_norm | r = +0.246 | 0.061 |
| pat_to_peak_norm | r = -0.227 | 0.052 |
| ptt_peak_to_peak_norm | r = -0.227 | 0.052 |
| ptt_peak_to_foot_norm | r = -0.222 | 0.049 |
| diastolic_duration_tfd_norm | r = -0.193 | 0.037 |

**Linear Model Performance (Group K-Fold)**:
- All _norm features: **R² = 0.071 ± 0.026**
- Top 5 features: **R² = 0.063 ± 0.020**

**AAMI Requirement**: R² ≥ 0.72 for SBP → **10x gap**

---

### Problem 2: Temporal Autocorrelation (r = 0.94)

**Adjacent samples share 94% of their information.**

| Lag | Autocorrelation | Interpretation |
|-----|-----------------|----------------|
| 1 | 0.944 | Nearly identical |
| 5 | 0.837 | Very similar |
| 10 | 0.734 | Still correlated |
| 20 | 0.604 | Moderate |
| 50 | 0.394 | Weak |
| 100 | 0.210 | Near independent |

**Effective Sample Size**:
```
Nominal:   181,842 samples
Effective:   5,624 samples (32x redundancy!)
```

**Impact**: Creates illusion of large dataset. Causes severe overfitting if not handled.

---

### Problem 3: Non-Stationary Relationships

**The PTT-BP relationship CHANGES during surgery.**

Example patients showing correlation by time quarter:

| Patient | Q1 | Q2 | Q3 | Q4 | Issue |
|---------|-----|-----|-----|-----|-------|
| 5844 | -0.29 | -0.37 | -0.02 | **+0.18** | Sign flip |
| 4596 | -0.05 | **+0.18** | -0.21 | -0.12 | Sign flip |
| 2859 | **+0.20** | -0.20 | -0.16 | -0.34 | Sign flip |
| 2130 | -0.39 | **+0.13** | +0.01 | -0.35 | Sign flip |

**Causes**:
- Anesthesia (changes vascular tone)
- Medications (vasopressors, vasodilators)
- Blood loss / fluid infusion
- Surgical stimulation
- Patient positioning

**Impact**: Calibration on early data **fails** for later data. This is why calibration makes predictions **worse**, not better.

---

### Problem 4: No Universal PTT-BP Relationship

**Each patient has a different PTT-BP slope.**

| Metric | Value |
|--------|-------|
| Mean slope | -4.15 mmHg/std |
| Std of slopes | 3.63 mmHg/std |
| Range | [-23.53, -0.10] |
| Coefficient of Variation | **88%** |

**Impact**: A global model assumes ONE slope for all patients. This is fundamentally wrong.

---

### Problem 5: Patient Identity Dominates Features

**77-89% of feature variance is between-patient, not within-patient.**

| Feature | Between-Patient | Within-Patient |
|---------|-----------------|----------------|
| hr_bpm | 79.1% | 59.1% |
| crest_time | 63.4% | 52.3% |
| ptt_peak_to_foot | 59.8% | 42.8% |
| pat_ecg_ppg | 59.8% | 42.8% |

**Impact**: 
- Models learn to identify patients, not predict BP
- High training R², negative test R² (Simpson's Paradox)
- The `_norm` features are **NOT** properly patient-normalized

---

## 4. Detailed Analysis

### 4.1 Variance Decomposition

Blood pressure variance can be decomposed into:

**SBP**:
- Total variance: 440.9 mmHg²
- Between-patient: 179.8 mmHg² (40.8%)
- Within-patient: 311.0 mmHg² (70.6%) ← **This is what we need to predict**

**DBP**:
- Total variance: 169.4 mmHg²
- Between-patient: 74.6 mmHg² (44.0%)
- Within-patient: 97.4 mmHg² (57.5%)

### 4.2 The _norm Features (FIXED Dec 8, 2025)

The `_norm` features are now **correctly normalized** per patient.

| Feature | Patient Mean Range | Patient Std Range | Status |
|---------|-------------------|-------------------|--------|
| ptt_peak_to_foot_norm | [-0.0000, 0.0000] | [1.000, 1.000] | ✅ Fixed |
| pat_ecg_ppg_norm | [-0.0000, 0.0000] | [1.000, 1.000] | ✅ Fixed |
| hr_bpm_norm | [-0.0000, 0.0000] | [1.000, 1.000] | ✅ Fixed |

**Impact**: Normalization leakage removed, but cross-patient R² still negative.

### 4.3 Cross-Patient vs Within-Patient Correlations

| Feature | Global r | Within-Patient r | Match? |
|---------|----------|------------------|--------|
| ptt_peak_to_foot | -0.038 | -0.235 | ✓ Same sign |
| hr_bpm | +0.104 | +0.245 | ✓ Same sign |
| pat_ecg_ppg | -0.038 | -0.235 | ✓ Same sign |

**Note**: Global correlations are **weaker** than within-patient because between-patient variance dilutes the signal.

### 4.4 Delta Features for BP Change Prediction

The `*_delta` features capture **changes** in features between consecutive beats.

| Feature | Correlation with SBP Change |
|---------|----------------------------|
| hr_bpm_delta | r = +0.016 |
| pat_to_peak_delta | r = -0.015 |
| ptt_peak_to_peak_delta | r = -0.015 |

**Conclusion**: Delta features have **near-zero correlation** with BP changes. Feature changes do NOT predict BP changes.

---

## 5. Calibration Results

### Why Calibration Works

The key insight: **Within a patient, BP is highly autocorrelated (r=0.94)**. Simply using recent BP values as prediction works well.

### EMA Calibration Performance

Exponential Moving Average (EMA) prediction: `pred(t) = α × BP(t-1) + (1-α) × pred(t-1)`

| α (adaptation rate) | SBP AAMI | SBP ME | SBP STD | DBP AAMI |
|---------------------|----------|--------|---------|----------|
| 0.05 (slow) | 50% | 0.92 | 9.13 | 85% |
| **0.10** | **61%** | 0.54 | 7.56 | **91%** |
| 0.20 | 79% | 0.28 | 6.24 | 97% |
| 0.30 (fast) | 83% | 0.19 | 5.62 | 98% |

**Best for clinical use**: α=0.1 (balances stability and responsiveness)

### Per-Patient Calibrated Linear Model

Training a linear model on first N samples per patient:

| Calibration Size | SBP AAMI | SBP STD |
|-----------------|----------|---------|
| 50 samples | 1% | 16.5 |
| 100 samples | 1% | 16.3 |
| 200 samples | 0% | 16.3 |
| 500 samples | 0% | 15.9 |

**Conclusion**: Feature-based prediction **fails completely** even with per-patient calibration. The features do not generalize within a patient's surgery.

### Hybrid Approach (EMA + Features)

| Approach | SBP AAMI | SBP STD |
|----------|----------|---------|
| EMA only (α=0.1) | **61%** | 7.6 |
| Features only (calibrated) | 1% | 16.3 |
| Hybrid (EMA + features) | 47% | 8.4 |

**Critical Finding**: Adding features to EMA **hurts** performance! Features add noise, not signal.

### Interpretation

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   The features contain NO useful predictive information        │
│   beyond what's already in the recent BP history.              │
│                                                                │
│   EMA calibration achieves 61-83% AAMI compliance              │
│   by exploiting autocorrelation, NOT features.                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. Why This is a Dataset Problem

### AAMI/ISO 81060-2 Requirements

| Metric | Requirement |
|--------|-------------|
| Mean Error (ME) | ≤ 5 mmHg |
| Standard Deviation (STD) | ≤ 8 mmHg |

### Required R² for AAMI Compliance

Given:
- SBP deviation std = 15.23 mmHg
- DBP deviation std = 8.64 mmHg
- Required STD ≤ 8 mmHg

Formula: `Error_STD = BP_STD × √(1 - R²)`

Solving for R²:

| Target | BP_STD | Required R² | Achievable R² | Verdict |
|--------|--------|-------------|---------------|---------|
| SBP | 15.23 | **0.72** | ~0.10-0.15 | ❌ Cannot meet AAMI |
| DBP | 8.64 | **0.14** | ~0.10-0.15 | ✓ Can meet AAMI |

### What's Actually Achievable

With current features (R² ≈ 0.10-0.15):

| Target | Achievable STD | AAMI Limit | Gap |
|--------|---------------|------------|-----|
| SBP | ~14 mmHg | 8 mmHg | 1.75x too high |
| DBP | ~8 mmHg | 8 mmHg | Just meets |

---

## 6. Why This is a Dataset Problem

### It's NOT a Training Problem Because:

| Training "Solution" | Why It Won't Work |
|--------------------|-------------------|
| Bigger model | Features only contain R²=0.07 signal max |
| Different architecture | Same features = same limit |
| More epochs | Memorizes patients, doesn't generalize |
| Data augmentation | Can't create signal that doesn't exist |
| Better optimizer | Can't find patterns that aren't there |
| Ensemble methods | Average of poor predictions = poor |

### It IS a Dataset Problem Because:

| Issue | Dataset Root Cause |
|-------|-------------------|
| Weak signal | **Feature extraction** didn't capture BP-relevant info |
| Non-stationarity | **Surgical context** confounds PTT-BP relationship |
| Patient specificity | **Physiology varies** too much between patients |
| Autocorrelation | **Sampling rate** creates redundant data |

### The Proof: EMA Beats Features

The strongest evidence this is a dataset problem:

1. **EMA achieves 61% AAMI compliance** using ONLY recent BP values
2. **Adding features REDUCES performance** to 47% AAMI
3. **Features are noise, not signal**

If this were a training problem, features would help at least a little. They don't.

---

## 7. Recommendations

### Option 1: Different Dataset (Recommended)

PTT-based BP estimation works better for:
- **Ambulatory monitoring** (stable subjects)
- **Resting conditions** (no surgical confounders)
- **Healthy subjects** (predictable physiology)

Consider: MIMIC-III PPG segments, UCI BP dataset, custom collection

### Option 2: Different Task - Trend Prediction

Instead of predicting BP value, predict **direction**:

```
Classification Task:
- Class 0: BP decreasing (Δ < -5 mmHg)
- Class 1: BP stable (-5 ≤ Δ ≤ +5 mmHg)
- Class 2: BP increasing (Δ > +5 mmHg)
```

**Advantages**:
- Clinically useful (alerts for BP changes)
- More achievable with weak features
- Doesn't require AAMI compliance

### Option 3: Different Features

Current features (PTT, PAT, HR) have weak signal. Consider:

1. **Raw waveform deep learning** - End-to-end learning from PPG/ECG
2. **Second derivative features** - Acceleration plethysmogram (APG)
3. **Morphology features** - Area under curve, peak ratios
4. **Multi-modal fusion** - Combine more signals

### Option 4: EMA Calibration Baseline (Recommended for This Dataset)

For this dataset specifically, use EMA as baseline:

```python
# Per-patient EMA prediction
alpha = 0.1  # Adaptation rate
warmup = 50  # Initial samples to establish baseline

ema = np.mean(bp_values[:warmup])
for t in range(warmup, len(patient_data)):
    prediction = ema  # Use current EMA as prediction
    actual = bp_values[t]
    ema = alpha * actual + (1 - alpha) * ema  # Update EMA
```

**Expected**: 61% SBP AAMI, 91% DBP AAMI

### Option 5: Accept Limitations

Report honestly:
- EMA calibration achieves 61-83% AAMI without features
- Features do not improve predictions
- Position as "BP trend monitoring" for surgical awareness

### Option 6: Advanced Training Strategy (New)

If you must train a model, follow the specific strategy outlined in **[training_strategy.md](./training_strategy.md)**.
This document details how to target **BP changes (deltas)** and **residual learning** to potentially beat the EMA baseline.

---

## 8. Running the Analysis

### Prerequisites

```bash
conda activate paper_ecg_ppg
pip install pandas numpy matplotlib seaborn scipy
```

### Run Full Analysis

```bash
cd /path/to/train
python dataset_problem_analysis.py
```

### Output
- Console: Detailed statistics for all 10 analysis sections
- File: `analysis_outputs/dataset_problem_analysis.png` (visualization)

### Key Functions in Analysis Script

| Function | Purpose |
|----------|---------|
| `analyze_autocorrelation()` | Compute lag correlations, effective sample size |
| `analyze_variance_decomposition()` | Between/within patient variance |
| `analyze_correlations()` | Feature-BP correlations |
| `analyze_nonstationarity()` | PTT-BP stability over time |
| `analyze_patient_specificity()` | Slope distribution across patients |
| `analyze_theoretical_limits()` | AAMI compliance math |

---

## Summary Table

| Aspect | Before Fix | After Fix | Implication |
|--------|------------|-----------|-------------|
| _norm Features | ❌ Not normalized | ✅ Fixed (mean=0, std=1) | Leakage removed |
| Feature Signal | R² = 0.06 | R² = 0.07 | Still fundamentally weak |
| Required for AAMI | R² = 0.72 | R² = 0.72 | **10x gap remains** |
| Temporal Correlation | r = 0.94 | r = 0.94 | 32x data redundancy |
| EMA Calibration | - | **61% AAMI** | Best approach |
| Features + Calibration | - | **1% AAMI** | Features hurt! |

**Final Verdict**: 

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   This is a DATASET limitation, not a MODEL limitation.            │
│                                                                    │
│   Key Findings (Dec 8, 2025):                                      │
│   1. Normalization fix did NOT improve cross-patient R²            │
│   2. EMA calibration achieves 61% SBP AAMI, 91% DBP AAMI           │
│   3. Adding features HURTS performance (61% → 47%)                 │
│   4. Features contain NO useful signal beyond autocorrelation      │
│                                                                    │
│   Recommendation: Use EMA calibration, ignore PTT/PAT features     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## References

1. AAMI/ISO 81060-2: Non-invasive sphygmomanometers — Clinical investigation
2. VitalDB: A high-fidelity multi-parameter vital signs database
3. Mukkamala et al., "Toward Ubiquitous Blood Pressure Monitoring via Pulse Transit Time", IEEE TBME 2015
