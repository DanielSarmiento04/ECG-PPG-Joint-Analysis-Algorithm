# VitalDB Dataset: BP Estimation Pipeline

This directory contains a complete pipeline for downloading, processing, and analyzing physiological waveform data from VitalDB for cuffless blood pressure estimation research.

## Overview

The project implements a robust signal processing pipeline to extract high-quality features from ECG and PPG signals. It includes:
1.  **Data Acquisition**: Downloading raw waveforms from VitalDB.
2.  **Signal Processing**: Wavelet-based denoising, adaptive R-peak detection, and beat segmentation.
3.  **Feature Extraction**: Robust multi-method PPG foot detection and morphological feature extraction.
4.  **Data Cleaning**: Physiological validation and quality-based filtering.
5.  **Analysis**: Statistical analysis and visualization of the extracted dataset.

## Recent Updates (December 2025)

### v2.0 - Major Pipeline Improvements
- **Fixed critical feature extraction bug**: Previous version had 93% invalid PTT values
- **New multi-method foot detection**: Consensus of 4 algorithms (second derivative, intersecting tangent, threshold crossing, minimum detection)
- **Improved data retention**: From 1.5% to 39.7% sample retention after cleaning
- **New modules added**:
  - `fixed_feature_extraction.py`: Robust feature extraction with quality checks
  - `data_cleaning.py`: Configurable physiological validation
  - `quality_assessment.py`: Automated quality reporting
  - `run_pipeline.py`: Complete pipeline orchestration

## Directory Structure

```
dataset/
├── main.py                      # Main data download script
├── bp_pipeline.py               # Core signal processing pipeline class
├── fixed_feature_extraction.py  # [NEW] Robust feature extraction
├── data_cleaning.py             # [NEW] Data cleaning and validation
├── quality_assessment.py        # [NEW] Quality metrics and reporting
├── run_pipeline.py              # [NEW] Complete pipeline runner
├── run_full_batch.py            # Batch processing script for all cases
├── feature_analysis.py          # Feature visualization and analysis
├── analyze_categorical.py       # Categorical variable analysis script
├── pipeline_config.yaml         # Configuration parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── dataset.md                   # Detailed dataset documentation
│
├── data/                        # Data directory
│   ├── raw/                     # Raw downloaded data (.npz files)
│   └── processed/               # Extracted features
│       ├── bp_dataset_raw.csv           # All extracted samples (~267K)
│       ├── bp_dataset_cleaned_v2.csv    # Cleaned samples (~106K)
│       ├── quality_report_before.json   # Pre-cleaning metrics
│       └── quality_report_after.json    # Post-cleaning metrics
│
└── logs/
    └── vitaldb_download.log     # Download logs
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies: `vitaldb`, `numpy`, `pandas`, `scipy`, `pywavelets`, `seaborn`, `matplotlib`, `joblib`.

### Step-by-Step Execution

#### Step 1: Download Data from VitalDB

First, download the raw signal data from VitalDB:

```bash
python main.py
```

**What it does:**
- Connects to VitalDB API
- Downloads ~50 cases with ECG, PPG, and arterial BP signals
- Saves as `.npz` files to `./data/raw/`

*This step only needs to be run once. It will prompt for confirmation before downloading.*

#### Step 2: Extract Features (Raw)

Process all downloaded cases to extract features:

```bash
python run_full_batch.py
```

**What it does:**
- Loads each `.npz` file from `./data/raw/`
- Applies **Wavelet Denoising** to ECG and PPG signals
- Detects R-peaks using an **adaptive threshold** algorithm
- Segments signals into individual cardiac cycles
- Filters low-quality beats using **Template Correlation** (threshold > 0.85)
- Extracts features using **multi-method consensus** for robust foot detection
- Adds per-patient normalized features
- Saves raw features to `./data/processed/bp_dataset_features.csv`

**Output:** `bp_dataset_features.csv` (~267,000 samples before cleaning)

#### Step 3: Clean and Validate Data

Apply physiological validation filters to remove invalid samples:

```bash
python data_cleaning.py
```

**What it does:**
- Loads raw features from `bp_dataset_features.csv`
- Filters samples with PTT = 0 (detection failure)
- Removes PTT values outside physiological range (20-600 ms)
- Removes impossible BP values (SBP < 60 or > 220, DBP < 30 or > 130)
- Removes SBP ≤ DBP violations
- Removes NaN values
- Saves cleaned data to `./data/processed/bp_dataset_cleaned_v2.csv`

**Output:** `bp_dataset_cleaned_v2.csv` (~106,000 samples after cleaning)

#### Step 4: Analyze Results

Generate visualizations and statistics:

```bash
python feature_analysis.py
```

**Outputs (in `./data/figures/`):**
*   `bp_distributions.png`: Histograms of SBP/DBP.
*   `feature_distributions.png`: Distributions of PTT and other features.
*   `correlation_matrix.png`: Feature correlation heatmap.
*   `ptt_vs_bp.png`: Scatter plots of PTT vs. Blood Pressure.

#### Step 5: Analyze Categorical Variables (Optional)

Evaluate the impact of clinical metadata (Position, Approach, etc.) on Blood Pressure:

```bash
python analyze_categorical.py
```

**Outputs (in `./data/figures/`):**
*   `bp_by_position.png`: Boxplots of BP by patient position.
*   `bp_by_approach.png`: Boxplots of BP by surgical approach.
*   `bp_by_aline1.png`: Boxplots of BP by arterial line location.
*   `bp_by_preop_ecg.png`: Boxplots of BP by pre-op ECG status.
*   **Console Output**: ANOVA statistical test results for each category.

## Pipeline Methodology

The `BPEstimationPipeline` (in `bp_pipeline.py`) implements the following steps:

1.  **Preprocessing**:
    *   **ECG**: Continuous Wavelet Transform (CWT) filtering (0.67-5 Hz passband).
    *   **PPG**: CWT filtering (0.5-2.1 Hz passband) to remove baseline wander and high-frequency noise.
2.  **Beat Detection**:
    *   Derivative-based adaptive thresholding for robust R-peak detection.
3.  **Quality Control**:
    *   **Cycle Segmentation**: Extracts windows around each R-peak.
    *   **Template Matching**: Computes a median template beat for the case. Rejects cycles with correlation < 0.85 to the template.
    *   **Physiological Limits**: Rejects HR < 40 or > 180 bpm, and invalid BP values.
4.  **Feature Extraction**:
    *   **PTT_Peak_to_Peak**: Time delay between ECG R-peak and PPG systolic peak.
    *   **PTT_Peak_to_Foot**: Time delay between ECG R-peak and PPG foot (diastolic trough).
    *   **Amplitude_Ratio**: Ratio of systolic to diastolic amplitude.
    *   **Durations**: Systolic and diastolic phase durations.

### Multi-Method Foot Detection (v2)

The `fixed_feature_extraction.py` module implements a robust **multi-method consensus** approach for PPG foot detection:

1.  **Second Derivative Maximum**: Finds the point of maximum acceleration, indicating rapid systolic upstroke.
2.  **Intersecting Tangent Method**: Projects tangent from max slope to baseline (classic literature method).
3.  **Threshold Crossing**: Finds where signal first exceeds 10% of cycle amplitude.
4.  **Minimum Before Peak**: Simple fallback - minimum value in first half of cycle.

**Consensus Logic:**
- Collects valid candidates from all methods (within physiological range: 10-200 samples at 500 Hz)
- Returns median of valid candidates if ≥2 methods agree
- Falls back to minimum-in-first-40% if methods disagree

This approach solved the previous issue where 93% of PTT values were invalid (returning 2ms due to index 0 detection).

## Dataset Statistics (Current Batch)

*   **Total Processed Cases**: 52
*   **Total Valid Cycles**: ~106,000
*   **Unique Patients**: 52
*   **Mean SBP**: ~115 ± 17 mmHg
*   **Mean DBP**: ~66 ± 10 mmHg
*   **Mean PTT (peak-to-foot)**: ~100 ms (valid range: 20-300 ms)
*   **Sample Retention Rate**: 39.7% (after quality filtering)
*   **Signal Quality**: High-quality cycles with correlation > 0.85.

### Key Improvements (v2)

| Metric | Before | After |
|--------|--------|-------|
| PTT Zero Rate | 52.4% | 0% |
| Valid Samples | 4,040 | 106,164 |
| Median PTT (foot) | 2 ms | ~100 ms |
| Sample Retention | 1.5% | 39.7% |

## Data Format

### Processed Features (`bp_dataset_cleaned_v2.csv`)

| Column | Description | Units |
|--------|-------------|-------|
| `patient_id` | Unique case identifier | - |
| `sbp_reference` | Systolic Blood Pressure | mmHg |
| `dbp_reference` | Diastolic Blood Pressure | mmHg |
| `ptt_peak_to_peak` | Pulse Transit Time (R-peak to PPG peak) | ms |
| `ptt_peak_to_foot` | Pulse Transit Time (R-peak to PPG foot) | ms |
| `amplitude_ratio_ra` | PPG Amplitude Ratio | - |
| `hr_bpm` | Heart Rate | BPM |
| `cycle_correlation` | Correlation with patient's template beat | 0-1 |

### Categorical Metadata

The dataset now includes detailed categorical metadata to improve model robustness:

| Column | Description | Examples |
|--------|-------------|----------|
| `position` | Patient body position during surgery | Supine, Lithotomy, Trendelenburg |
| `approach` | Surgical approach | Open, Videoscopic, Robotic |
| `aline1` | Arterial line location (affects waveform shape) | Right radial, Left radial, Femoral |
| `dx` | Primary diagnosis | Hypertension, Trauma, Cancer |
| `opname` | Name of the operation | Laparoscopic Cholecystectomy |
| `preop_ecg` | Pre-operative ECG status | Normal Sinus Rhythm, Atrial Fibrillation |

## Known Issues & Important Notes

### PTT-BP Correlation in Surgical Data

**Finding:** The expected strong inverse PTT-BP correlation (r ≈ -0.5 to -0.8) is weak in this VitalDB surgical dataset (r ≈ -0.01 to -0.09).

**Why this happens in VitalDB surgical data:**
- Patients are under **general anesthesia** which affects BP regulation
- Active **vasoactive medications** (pressors, vasodilators) alter vascular tone
- Limited within-patient **BP variation** (controlled surgical environment)
- Arterial line placement and **calibration artifacts**

**Implications:**
- The data extraction is now correct (valid PTT values, proper foot detection)
- ML models may have lower accuracy than expected on this specific dataset
- Consider per-patient normalization or patient-specific models

### Issue 1: PyWavelets Freezing
**Problem:** `cwt` with `method='conv'` freezes on large arrays.
**Solution:** Implemented chunked processing with `method='fft'` in `bp_pipeline.py`.

### Issue 2: Memory Crashes
**Problem:** `np.corrcoef` on large matrices ($N^2$ complexity) caused crashes.
**Solution:** Replaced with $O(N)$ dot-product correlation against a median template.

## References

- **VitalDB:** https://vitaldb.net/
- **VitalDB Paper:** Lee HC, Jung CW. Vital Recorder—a free research tool for automatic recording of high-resolution time-synchronised physiological data from multiple anaesthesia devices. Sci Rep. 2018;8:1527.
- **API Documentation:** https://vitaldb.net/dataset/?query=lib

## Citation

If you use this data in your research, please cite:

```bibtex
@article{lee2018vital,
  title={Vital Recorder—a free research tool for automatic recording of high-resolution time-synchronised physiological data from multiple anaesthesia devices},
  author={Lee, Hyung-Chul and Jung, Chang-Wook},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={1527},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## Contributing

For issues, questions, or contributions:
1. Check existing issues
2. Open new issue with detailed description
3. Include error logs and system information

## License

This code is for research purposes. VitalDB data is subject to VitalDB's terms of use.

---

**Last Updated:** November 28, 2025  
**VitalDB API Version:** 1.5.8  
**Python Version:** 3.10+
