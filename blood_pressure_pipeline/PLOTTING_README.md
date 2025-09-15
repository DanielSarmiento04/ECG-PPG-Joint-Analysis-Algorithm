# M3 Pro Results Visualization

This directory contains scripts to generate comprehensive visualizations from the trained M3 Pro optimized blood pressure prediction models.

## Quick Start

### Option 1: Using the Shell Script (Recommended)
```bash
./run_plots.sh
```

### Option 2: Direct Python Execution
```bash
python generate_plots.py
```

## Requirements

The plotting scripts require the following packages:
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.11.0
- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `joblib` >= 1.0.0

Install with:
```bash
pip install matplotlib seaborn numpy pandas joblib
```

## Prerequisites

Before running the visualization scripts, you must have:
1. ✅ Trained models saved in the `models/` directory
2. ✅ Evaluation reports saved in the `reports/` directory (optional but recommended)

To generate these, run the full training pipeline first:
```bash
python demo_m3_pro.py
```

## Generated Visualizations

The plotting script generates the following visualizations in the `plots/` directory:

### 1. Model Performance Plots
- **`sbp_model_performance.png`**: SBP model comparison (R², RMSE, training time, performance trade-offs)
- **`dbp_model_performance.png`**: DBP model comparison (R², RMSE, training time, performance trade-offs)

### 2. Clinical Validation Plots
- **`clinical_validation.png`**: BHS grade distribution and clinical accuracy thresholds

### 3. Performance Summary
- **`performance_summary.png`**: Overall performance summary with best model metrics

### 4. Feature Importance
- **`*_feature_importance.png`**: Top 20 feature importances for each model

### 5. Hardware Utilization
- **`hardware_utilization.png`**: M3 Pro hardware utilization summary

## Visualization Features

### Model Performance Analysis
- **R² Score Comparison**: Higher values indicate better model fit
- **RMSE Comparison**: Lower values indicate better accuracy
- **Training Time Analysis**: Optimization efficiency on M3 Pro hardware
- **Performance Trade-offs**: Scatter plots showing accuracy vs speed

### Clinical Validation
- **BHS Grade Distribution**: Clinical validation according to British Hypertension Society standards
- **Accuracy Thresholds**: Percentage of predictions within ±5, ±10, ±15 mmHg
- **AAMI Compliance**: Association for the Advancement of Medical Instrumentation standards

### Hardware Utilization
- **CPU Core Usage**: M3 Pro core utilization during training
- **Memory Utilization**: Available system memory
- **Training Time Breakdown**: Time spent per model
- **Platform Information**: Hardware configuration details

## Understanding the Results

### R² Score Interpretation
- **0.80-0.85**: Good model performance
- **0.85-0.90**: Excellent model performance
- **0.90+**: Outstanding model performance

### BHS Grade Classification
- **Grade A**: Excellent (≥60% within ±5 mmHg, ≥85% within ±10 mmHg, ≥95% within ±15 mmHg)
- **Grade B**: Good (≥50% within ±5 mmHg, ≥75% within ±10 mmHg, ≥90% within ±15 mmHg)
- **Grade C**: Acceptable (≥40% within ±5 mmHg, ≥65% within ±10 mmHg, ≥85% within ±15 mmHg)
- **Grade D**: Poor (Below Grade C requirements)

### RMSE Targets
- **SBP**: <8 mmHg is considered good, <6 mmHg is excellent
- **DBP**: <5 mmHg is considered good, <4 mmHg is excellent

## File Structure

```
plots/
├── sbp_model_performance.png      # SBP models comparison
├── dbp_model_performance.png      # DBP models comparison
├── clinical_validation.png        # Clinical accuracy plots
├── performance_summary.png        # Overall performance summary
├── *_feature_importance.png       # Feature importance plots
└── hardware_utilization.png       # M3 Pro hardware usage
```

## Troubleshooting

### No Models Found
```
❌ No saved models found. Please run the training pipeline first.
```
**Solution**: Run `python demo_m3_pro.py` to train and save models.

### Missing Packages
```
❌ Missing packages: ['matplotlib', 'seaborn']
```
**Solution**: Install missing packages with `pip install matplotlib seaborn numpy pandas joblib`

### No Plots Generated
- Check that the `models/` directory contains `.pkl` files
- Verify that the models were saved with performance metrics
- Check the console output for specific error messages

## Tips

1. **View Results**: Use `open plots/` (macOS) or your file manager to view the generated plots
2. **High Resolution**: All plots are saved at 300 DPI for publication quality
3. **Custom Analysis**: Modify `generate_plots.py` to add custom visualizations
4. **Batch Processing**: The script processes all saved models automatically

## Performance Expectations

On M3 Pro hardware, the visualization generation typically takes:
- **Model Loading**: 5-10 seconds
- **Plot Generation**: 15-30 seconds
- **Total Time**: <1 minute

The generated plots provide comprehensive insights into:
- Model accuracy and clinical validation
- Hardware optimization effectiveness
- Feature importance analysis
- Performance trade-offs between different algorithms
