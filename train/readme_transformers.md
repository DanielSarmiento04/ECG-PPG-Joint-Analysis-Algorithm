# Transformer-based Blood Pressure Estimation

This directory contains a Transformer-based deep learning approach for estimating Systolic (SBP) and Diastolic (DBP) Blood Pressure from physiological signal features derived from ECG and PPG signals.

## 1. Overview

The goal is to leverage the self-attention mechanism of Transformers to capture complex, non-linear relationships between hemodynamic features and blood pressure values. Unlike traditional Feed-Forward Neural Networks (FFNN), this architecture treats the input features as a sequence, allowing the model to learn dependencies between different physiological markers (e.g., how Pulse Transit Time relates to Systolic Duration).

## 2. Model Architecture: `PhysiologicalTransformer`

The model is defined in `src/models/transformers.py` and is adapted from the Vision Transformer (ViT) architecture for 1D data.

### Key Components:
*   **Input Embedding**: The 7 input features are treated as a sequence of tokens. A linear projection maps them to a hidden vector space.
*   **Positional Encoding**: Learnable position embeddings are added to retain the identity/order of each feature.
*   **Transformer Encoder Blocks**: A stack of layers (default: 6) containing:
    *   **Multi-Head Self-Attention (MSA)**: Allows the model to weigh the importance of different features relative to each other.
    *   **Feed-Forward Networks (MLP)**: Processes the information extracted by attention.
    *   **Layer Normalization & Residual Connections**: Ensures stable training and gradient flow.
*   **Regression Head**: A final linear layer maps the encoded representation (from the class token) to the 2 output values (SBP, DBP).

## 3. Dataset

The model is designed to train on the processed feature dataset: `src/data/bp_dataset_features.csv`.

### Input Features (7 variables)
The pipeline extracts these specific features from the raw signals:
1.  `ptt_peak_to_peak`: Pulse Transit Time (R-peak to PPG peak).
2.  `ptt_peak_to_foot`: Pulse Transit Time (R-peak to PPG foot).
3.  `ptt_peak_to_maxslope`: Pulse Transit Time (R-peak to PPG max slope).
4.  `amplitude_ratio_ra`: Ratio of systolic to diastolic amplitude.
5.  `systolic_duration_tsd`: Duration of the systolic phase.
6.  `diastolic_duration_tfd`: Duration of the diastolic phase.
7.  `time_to_maxslope_t1`: Time from foot to max slope.

### Targets
*   `sbp_reference`: Systolic Blood Pressure (mmHg)
*   `dbp_reference`: Diastolic Blood Pressure (mmHg)

## 4. Training Pipeline

The training script `transformers.py` implements a robust training loop with modern deep learning practices.

### Key Features
*   **Data Normalization**: Uses `TorchStandardScaler` to Z-score normalize inputs and targets (crucial for regression).
*   **Mixed Precision (AMP)**: Uses FP16 training on GPUs to reduce memory usage and speed up training.
*   **Learning Rate Scheduler**: Implements `OneCycleLR` for faster convergence and super-convergence properties.
*   **Early Stopping**: Monitors validation R² score to stop training if the model stops improving.
*   **Checkpointing**: Automatically saves the best model (`best_model.pt`) based on the highest R² score.

## 5. How to Run

### Prerequisites
Ensure you have the necessary Python packages installed:
```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm torchinfo
```

### Training Command
Run the training script from the root of the workspace:

```bash
python train/transformers.py --data_path train/src/data/bp_dataset_features.csv --epochs 100 --batch_size 64
```

### Configuration Arguments
You can customize the training via command-line arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | `src/data/bp_dataset_features.csv` | Path to the input CSV file. |
| `--epochs` | `200` | Maximum number of training epochs. |
| `--batch_size` | `128` | Number of samples per batch. |
| `--lr` | `0.001` | Initial learning rate. |
| `--depth` | `6` | Number of Transformer encoder blocks. |
| `--num_heads` | `8` | Number of attention heads. |
| `--hidden_size` | `64` | Dimension of the hidden layer. |
| `--save_dir` | `checkpoints` | Directory to save model and plots. |

## 6. Outputs

After training, the script produces:
1.  **Console Logs**: Real-time metrics (Loss, RMSE, MAE, R²) for training and validation.
2.  **`checkpoints/best_model.pt`**: The saved model weights with the best validation performance.
3.  **`checkpoints/training_curves.png`**: A visualization of the Loss and R² score evolution over epochs.
