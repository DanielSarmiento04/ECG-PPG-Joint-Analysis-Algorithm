# VitalDB Dataset: BP Estimation Pipeline

This directory contains a complete pipeline for downloading, processing, and analyzing physiological waveform data from VitalDB for cuffless blood pressure estimation research.

## 📋 Overview

The project implements a robust signal processing pipeline to extract high-quality features from ECG and PPG signals. It includes:
1.  **Data Acquisition**: Downloading raw waveforms from VitalDB.
2.  **Signal Processing**: Wavelet-based denoising, adaptive R-peak detection, and beat segmentation.
3.  **Feature Extraction**: Calculation of Pulse Transit Time (PTT) and morphological features.
4.  **Analysis**: Statistical analysis and visualization of the extracted dataset.

## 🗂️ Directory Structure

```
dataset/
├── main.py                    # Main data download script
├── data_exploration.py        # Initial data visualization
├── bp_pipeline.py             # Core signal processing pipeline class
├── run_full_batch.py          # Batch processing script for all cases
├── feature_analysis.py        # Feature visualization and analysis
├── pipeline_config.yaml       # Configuration parameters
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Data directory
│   ├── raw/                   # Raw downloaded data (.npz files)
│   ├── processed/             # Extracted features
│   │   └── bp_dataset_features.csv
│   ├── metadata/              # Case metadata
│   │   ├── available_cases.csv
│   │   └── download_summary.json
│   └── figures/               # Generated visualizations
│
└── logs/
    └── vitaldb_download.log   # Download logs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies: `vitaldb`, `numpy`, `pandas`, `scipy`, `pywavelets`, `seaborn`, `matplotlib`, `joblib`.

### 1. Download Data

Run the main download script to fetch cases from VitalDB:

```bash
python main.py
```
*Downloads ~50 cases (configurable) to `./data/raw/`.*

### 2. Run Processing Pipeline

Process all downloaded cases to extract features:

```bash
python run_full_batch.py
```

**What it does:**
*   Loads each `.npz` file.
*   Applies **Wavelet Denoising** to ECG and PPG signals.
*   Detects R-peaks using an **adaptive threshold** algorithm.
*   Segments signals into individual cardiac cycles.
*   Filters low-quality beats using **Template Correlation** (threshold > 0.85).
*   Extracts features (PTT, Amplitudes, Durations) and aligns with BP labels.
*   Saves results to `./data/processed/bp_dataset_features.csv`.

### 3. Analyze Results

Generate visualizations and statistics:

```bash
python feature_analysis.py
```

**Outputs (in `./data/figures/`):**
*   `bp_distributions.png`: Histograms of SBP/DBP.
*   `feature_distributions.png`: Distributions of PTT and other features.
*   `correlation_matrix.png`: Feature correlation heatmap.
*   `ptt_vs_bp.png`: Scatter plots of PTT vs. Blood Pressure.

## ⚙️ Pipeline Methodology

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

## 📊 Dataset Statistics (Current Batch)

*   **Total Processed Cases**: 48
*   **Total Valid Cycles**: ~45,300
*   **Unique Patients**: 44
*   **Mean SBP**: 115.1 ± 17.3 mmHg
*   **Mean DBP**: 66.3 ± 9.9 mmHg
*   **Mean HR**: 88.3 ± 12.1 BPM
*   **Signal Quality**: Mean cycle correlation of **0.94**.

## 🔧 Data Format

### Processed Features (`bp_dataset_features.csv`)

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

## ⚠️ Known Issues & Solutions

### Issue 1: PyWavelets Freezing
**Problem:** `cwt` with `method='conv'` freezes on large arrays.
**Solution:** Implemented chunked processing with `method='fft'` in `bp_pipeline.py`.

### Issue 2: Memory Crashes
**Problem:** `np.corrcoef` on large matrices ($N^2$ complexity) caused crashes.
**Solution:** Replaced with $O(N)$ dot-product correlation against a median template.

## 📚 References

- **VitalDB:** https://vitaldb.net/
- **VitalDB Paper:** Lee HC, Jung CW. Vital Recorder—a free research tool for automatic recording of high-resolution time-synchronised physiological data from multiple anaesthesia devices. Sci Rep. 2018;8:1527.
- **API Documentation:** https://vitaldb.net/dataset/?query=lib

## 📝 Citation

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

## 🤝 Contributing

For issues, questions, or contributions:
1. Check existing issues
2. Open new issue with detailed description
3. Include error logs and system information

## 📄 License

This code is for research purposes. VitalDB data is subject to VitalDB's terms of use.

---

**Last Updated:** November 24, 2025  
**VitalDB API Version:** 1.5.8  
**Python Version:** 3.8+
