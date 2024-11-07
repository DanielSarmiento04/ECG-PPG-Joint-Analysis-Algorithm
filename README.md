# ECG-PPG-Joint-Analysis-Algorithm

This project contains algorithms for joint analysis of ECG (Electrocardiogram) and PPG (Photoplethysmogram) signals. The analysis includes signal processing, feature extraction, and visualization using Continuous Wavelet Transform (CWT).

## Project Structure


## Requirements

To install the required packages, run:

```sh
    pip install -r requirements.txt
```

## Usage
Loading and Processing ECG Data
The function load_and_process_ecg in main.ipynb loads and processes ECG data from WAV files in the ECG folder.

## Continuous Wavelet Transform (CWT)
The function apply_cwt in main.ipynb applies the Continuous Wavelet Transform to the ECG signal.

## Plotting CWT Analysis
The function plot_cwt_analysis in main.ipynb creates plots for the original signal and CWT scalogram.

## Analyzing CWT Peaks
The function analyze_cwt_peaks in main.ipynb analyzes peaks in the CWT coefficients.

## Example
To run the analysis, execute the cells in main.ipynb in order. The notebook will load the ECG data, apply CWT, and visualize the results.

