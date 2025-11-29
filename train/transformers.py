# -*- coding: utf-8 -*-
"""
Transformer for Physiological Signal Analysis

This script implements a Transformer model for predicting physiological
parameters (e.g., blood pressure) from ECG and PPG signals or extracted features.

Key Features:
- Custom Transformer architecture for physiological signals
- Mixed precision training support (FP16)
- Early stopping with patience
- Model checkpointing (best model saving)
- Gradient clipping for stability
- Comprehensive evaluation metrics (R², MAE, RMSE)
- Learning rate scheduling with OneCycleLR
- Data normalization and augmentation
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
# from torchinfo import summary

from src.models.transformers import PhysiologicalTransformer
from src.utils.hardware import get_device

from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_optimizer_and_scheduler(
    model: nn.Module, 
    args: argparse.Namespace, 
    num_training_steps: int
) -> Tuple[AdamW, OneCycleLR]:
    """
    Create optimizer with weight decay and learning rate scheduler.
    
    Uses AdamW optimizer with differential weight decay:
    - Applies weight decay to weights but not to biases and LayerNorm parameters
    - OneCycleLR scheduler for cyclical learning rate with warm-up
    
    Args:
        model: Neural network model
        args: Training arguments containing lr and weight_decay
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        total_steps=num_training_steps, 
        pct_start=0.1,  # 10% warm-up
        anneal_strategy='cos',
        div_factor=25.0,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = max_lr/1e4
    )
    return optimizer, scheduler

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for Blood Pressure Prediction.
    Emphasizes clinically critical ranges (Hypotensive < 90, Hypertensive > 140).
    """
    def __init__(self, device, alpha=1.0, beta=1.0):
        super().__init__()
        self.device = device
        self.alpha = alpha # Weight for SBP
        self.beta = beta   # Weight for DBP
        
    def forward(self, pred, target, y_scaler=None):
        # pred, target are scaled values
        # We need unscaled values to determine weights
        
        if y_scaler is not None:
            # Unscale to get mmHg
            # Note: inverse_transform might detach gradients if not careful, 
            # but we only need it for weight calculation which doesn't require gradients on weights
            with torch.no_grad():
                target_mmhg = y_scaler.inverse_transform(target)
        else:
            target_mmhg = target
            
        # Calculate weights based on target_mmhg
        # SBP is index 0, DBP is index 1
        sbp = target_mmhg[:, 0]
        dbp = target_mmhg[:, 1]
        
        # Weights for SBP
        w_sbp = torch.ones_like(sbp)
        w_sbp[sbp < 90] = 2.0
        w_sbp[sbp > 140] = 2.0
        w_sbp[sbp < 70] = 3.0
        w_sbp[sbp > 160] = 3.0
        
        # Weights for DBP
        w_dbp = torch.ones_like(dbp)
        w_dbp[dbp < 60] = 2.0
        w_dbp[dbp > 90] = 2.0
        
        # Combine weights
        weights = torch.stack([w_sbp, w_dbp], dim=1)
        
        # Apply global task weights (alpha for SBP, beta for DBP)
        weights[:, 0] *= self.alpha
        weights[:, 1] *= self.beta
        
        # Calculate MSE
        loss = (pred - target) ** 2
        
        # Apply weights
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

class TorchStandardScaler:
    """
    PyTorch-based standard scaler for normalizing tensors.
    
    Performs z-score normalization: (x - mean) / std
    All operations are performed on GPU/CPU tensors for efficiency.
    
    Attributes:
        device: Device for tensor operations (CPU or CUDA)
        mean: Feature-wise mean values
        std: Feature-wise standard deviation values
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize scaler.
        
        Args:
            device: PyTorch device (torch.device('cuda') or torch.device('cpu'))
        """
        self.device = device
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> 'TorchStandardScaler':
        """
        Compute mean and std from training data.
        
        Args:
            x: Input tensor of shape (n_samples, n_features)
            
        Returns:
            self for method chaining
        """
        self.mean = x.mean(0, keepdim=True).to(self.device)
        self.std = x.std(0, unbiased=False, keepdim=True).to(self.device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using fitted parameters.
        
        Args:
            x: Input tensor to normalize
            
        Returns:
            Normalized tensor
            
        Raises:
            ValueError: If scaler hasn't been fitted
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        x = x.to(self.device)
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-7)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            x: Normalized tensor
            
        Returns:
            Denormalized tensor in original scale
            
        Raises:
            ValueError: If scaler hasn't been fitted
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse_transform. Call fit() first.")
        x = x.to(self.device)
        return x * self.std.to(x.device) + self.mean.to(x.device)

def prepare_data(
    data_path: str, 
    device: torch.device, 
    batch_size: int, 
    input_length: int,
    train_split: float = 0.9,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, TorchStandardScaler, TorchStandardScaler, TorchStandardScaler, list]:
    """
    Load and prepare ECG/PPG data for training with hybrid inputs.
    
    Loads data from Excel/CSV file, normalizes features and targets,
    encodes categorical variables, and creates train/test data loaders.
    
    Returns:
        Tuple of (train_loader, test_loader, y_scaler, X_scaler, num_scaler, cardinalities)
    """
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # --- 1. Signal Features ---
    feature_cols = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'ptt_peak_to_maxslope', 'amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd', 'time_to_maxslope_t1']
    target_cols = ['sbp_reference', 'dbp_reference']
    
    # --- 2. Numerical Metadata ---
    # age, bmi, hr_bpm
    num_meta_cols = ['age', 'bmi', 'hr_bpm']
    
    # --- 3. Categorical Metadata ---
    # sex, position, approach, aline1, preop_ecg
    cat_meta_cols = ['sex', 'position', 'approach', 'aline1', 'preop_ecg']
    
    # Check columns exist
    all_cols = feature_cols + target_cols + num_meta_cols + cat_meta_cols
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}. Filling with defaults.")
        for c in missing_cols:
            if c in num_meta_cols:
                df[c] = 0.0
            elif c in cat_meta_cols:
                df[c] = 'Unknown'
    
    # --- Feature Engineering: Interaction Features ---
    # PTT / Height (using BMI as proxy for body size if height not available)
    # PTT * Age
    if 'age' in df.columns and 'ptt_peak_to_peak' in df.columns:
        df['interaction_ptt_age'] = df['ptt_peak_to_peak'] * df['age']
        num_meta_cols.append('interaction_ptt_age')
        
    if 'bmi' in df.columns and 'ptt_peak_to_peak' in df.columns:
        df['interaction_ptt_bmi'] = df['ptt_peak_to_peak'] / (df['bmi'] + 1e-6)
        num_meta_cols.append('interaction_ptt_bmi')

    # Extract Tensors
    X_signal = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
    y = torch.tensor(df[target_cols].values, dtype=torch.float32, device=device)
    X_num = torch.tensor(df[num_meta_cols].fillna(0).values, dtype=torch.float32, device=device)
    
    # Process Categorical
    X_cat_list = []
    cardinalities = []
    for col in cat_meta_cols:
        le = LabelEncoder()
        # Fill NaNs with 'Unknown' and convert to string
        col_data = df[col].fillna('Unknown').astype(str).values
        encoded = le.fit_transform(col_data)
        X_cat_list.append(torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(1))
        cardinalities.append(len(le.classes_))
        logger.info(f"Categorical '{col}': {len(le.classes_)} classes")
        
    X_cat = torch.cat(X_cat_list, dim=1)
    
    # Handle NaNs/Infs in numerical data
    for tensor_name, tensor in [('X_signal', X_signal), ('y', y), ('X_num', X_num)]:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"Found NaN/Inf in {tensor_name}, replacing with zeros")
            tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scalers
    X_scaler = TorchStandardScaler(device).fit(X_signal)
    y_scaler = TorchStandardScaler(device).fit(y)
    num_scaler = TorchStandardScaler(device).fit(X_num)
    
    # Normalize
    X_signal_norm = X_scaler.transform(X_signal).unsqueeze(1)
    y_norm = y_scaler.transform(y)
    X_num_norm = num_scaler.transform(X_num)
    
    # Split
    total_size = X_signal_norm.size(0)
    train_size = int(train_split * total_size)
    
    torch.manual_seed(random_seed)
    indices = torch.randperm(total_size, device=device)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_dataset = TensorDataset(
        X_signal_norm[train_idx], 
        X_num_norm[train_idx], 
        X_cat[train_idx], 
        y_norm[train_idx]
    )
    test_dataset = TensorDataset(
        X_signal_norm[test_idx], 
        X_num_norm[test_idx], 
        X_cat[test_idx], 
        y_norm[test_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, y_scaler, X_scaler, num_scaler, cardinalities

def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    y_scaler: Optional[TorchStandardScaler] = None
) -> Dict[str, float]:
    """
    Evaluate model on test set with comprehensive metrics.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, x_num, x_cat, targets in test_loader:
            inputs = inputs.to(device)
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs, x_num, x_cat)
            
            # Pass y_scaler to criterion if it's WeightedMSELoss
            if isinstance(criterion, WeightedMSELoss):
                loss = criterion(predictions, targets, y_scaler)
            else:
                loss = criterion(predictions, targets)
                
            total_loss += loss.item()
            
            # Inverse transform if scaler provided
            if y_scaler is not None:
                targets = y_scaler.inverse_transform(targets)
                predictions = y_scaler.inverse_transform(predictions)
                
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Aggregate predictions
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    # Note: MSE/Loss here is on the scaled data if y_scaler is None, or unscaled if provided
    # But total_loss above is always on scaled data (as per training objective)
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # Per-output R² scores (useful for multi-output regression)
    r2_per_output = r2_score(all_targets, all_predictions, multioutput='raw_values')
    
    return {
        'loss': total_loss / len(test_loader), # Keep loss on scaled data for consistency
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'r2_output_0': r2_per_output[0],
        'r2_output_1': r2_per_output[1] if len(r2_per_output) > 1 else 0.0
    }



def main():
    """Main training loop with advanced techniques."""
    parser = argparse.ArgumentParser(description='Train Vision Transformer for ECG/PPG signal analysis')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='src/data/bp_dataset_features.csv',
                       help='Path to the dataset file (CSV or Excel)')
    parser.add_argument('--input_length', type=int, default=7,
                       help='Length of input signal or number of features')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Dimension of hidden representations')
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=512,
                       help='Dimension of MLP layer')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--att_dropout', type=float, default=0.1,
                       help='Attention dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='Weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training')
    parser.add_argument('--use_film', action='store_true',
                       help='Use FiLM layers for demographic conditioning')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Skip training and only evaluate the best model')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # MPS-specific reproducibility
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and prepare data
    train_loader, test_loader, y_scaler, X_scaler, num_scaler, cardinalities = prepare_data(
        args.data_path, device, args.batch_size, args.input_length, random_seed=args.seed
    )
    
    # Initialize model
    # Determine number of numerical features dynamically from the scaler
    # num_scaler.mean is a tensor of shape (1, num_features)
    if num_scaler.mean is not None:
        num_numerical_features = num_scaler.mean.shape[1]
    else:
        num_numerical_features = 3 # Fallback default
        
    model = PhysiologicalTransformer(
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_outputs=2,  # SBP and DBP
        dropout=args.dropout,
        attention_dropout=args.att_dropout,
        input_length=args.input_length,
        num_numerical_features=num_numerical_features, 
        categorical_cardinalities=cardinalities,
        use_film=args.use_film
    ).to(device)
    
    # Print model summary
    logger.info("\nModel Architecture:")
    # summary(model, input_size=(args.batch_size, 1, args.input_length)) # Summary might fail with multiple inputs
    
    # Loss function and optimizer
    # Use Weighted MSE Loss for clinical relevance
    criterion = WeightedMSELoss(device=device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, len(train_loader) * args.epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Training metrics tracking
    train_losses = []
    test_losses = []
    r2_scores = []
    best_r2 = -float('inf')
    patience_counter = 0
    
    if args.evaluate_only:
        logger.info("Evaluation mode enabled: Skipping training loop.")
        args.epochs = 0

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, x_num, x_cat, targets in pbar:
            inputs = inputs.to(device)
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = model(inputs, x_num, x_cat)
                    loss = criterion(predictions, targets, y_scaler)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(inputs, x_num, x_cat)
                loss = criterion(predictions, targets, y_scaler)
                loss.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation phase
        metrics = evaluate_model(model, test_loader, criterion, device, y_scaler)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(metrics['loss'])
        r2_scores.append(metrics['r2'])
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {metrics['loss']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f} mmHg, "
            f"MAE: {metrics['mae']:.4f} mmHg, "
            f"R²: {metrics['r2']:.4f}"
        )
        
        # Save best model
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_r2': best_r2,
                'args': vars(args),
                'y_scaler_mean': y_scaler.mean,
                'y_scaler_std': y_scaler.std,
                'num_scaler_mean': num_scaler.mean,
                'num_scaler_std': num_scaler.std,
                'cardinalities': cardinalities
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            logger.info(f"  ✓ New best R² score: {best_r2:.4f} - Model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {args.patience})")
            break
    
    # Plot training curves
    if not args.evaluate_only:
        plot_training_curves(train_losses, test_losses, r2_scores, args.save_dir)
    
    # --- Final Evaluation & Prediction Saving ---
    logger.info("\nLoading best model for final evaluation and prediction saving...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load scaler stats from checkpoint to ensure consistency with training
    if 'y_scaler_mean' in checkpoint:
        y_scaler.mean = checkpoint['y_scaler_mean'].to(device)
        y_scaler.std = checkpoint['y_scaler_std'].to(device)
    if 'num_scaler_mean' in checkpoint:
        num_scaler.mean = checkpoint['num_scaler_mean'].to(device)
        num_scaler.std = checkpoint['num_scaler_std'].to(device)
        
    if 'best_r2' in checkpoint:
        best_r2 = checkpoint['best_r2']
    
    # Get predictions
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, x_num, x_cat, targets in test_loader:
            inputs = inputs.to(device)
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs, x_num, x_cat)
            
            # Inverse transform
            targets = y_scaler.inverse_transform(targets)
            predictions = y_scaler.inverse_transform(predictions)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Save predictions to CSV for ensembling
    results_df = pd.DataFrame({
        'sbp_true': all_targets[:, 0],
        'dbp_true': all_targets[:, 1],
        'sbp_pred_transformer': all_predictions[:, 0],
        'dbp_pred_transformer': all_predictions[:, 1]
    })
    results_path = os.path.join(args.save_dir, 'predictions_transformer.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Predictions saved to: {results_path}")
    
    # Save Test Data for GBDT (to ensure same split)
    # We need to reconstruct the test dataframe from the loader or indices
    # Since we didn't save indices, we can't easily reconstruct the exact features row-by-row 
    # aligned with the original CSV unless we saved them.
    # However, for ensembling, we just need the targets to match.
    # But for GBDT training, we need the FEATURES of the test set.
    
    # Let's save the full test set (features + targets) to a CSV
    # We need to iterate the loader again and collect features
    logger.info("Saving test set features for GBDT benchmarking...")
    all_features_signal = []
    all_features_num = []
    all_features_cat = []
    
    with torch.no_grad():
        for inputs, x_num, x_cat, _ in test_loader:
            # Inverse transform features
            inputs = X_scaler.inverse_transform(inputs.squeeze(1))
            x_num = num_scaler.inverse_transform(x_num)
            
            all_features_signal.append(inputs.cpu().numpy())
            all_features_num.append(x_num.cpu().numpy())
            all_features_cat.append(x_cat.cpu().numpy())
            
    X_signal_test = np.vstack(all_features_signal)
    X_num_test = np.vstack(all_features_num)
    X_cat_test = np.vstack(all_features_cat)
    
    # Create DataFrame
    # Note: We need column names. We'll hardcode them based on prepare_data
    feature_cols = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'ptt_peak_to_maxslope', 'amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd', 'time_to_maxslope_t1']
    # num_meta_cols depends on feature engineering, but we can just name them num_0, num_1...
    # Actually, we know them: age, bmi, hr_bpm, interaction_ptt_age, interaction_ptt_bmi
    # But let's just use generic names to be safe or try to match
    
    test_df = pd.DataFrame(X_signal_test, columns=feature_cols)
    
    # Add numerical metadata
    # We know there are 5 numerical features usually
    num_cols = ['age', 'bmi', 'hr_bpm', 'interaction_ptt_age', 'interaction_ptt_bmi']
    if X_num_test.shape[1] == len(num_cols):
        for i, col in enumerate(num_cols):
            test_df[col] = X_num_test[:, i]
    else:
        for i in range(X_num_test.shape[1]):
            test_df[f'num_{i}'] = X_num_test[:, i]
            
    # Add categorical
    cat_cols = ['sex', 'position', 'approach', 'aline1', 'preop_ecg']
    for i, col in enumerate(cat_cols):
        test_df[col] = X_cat_test[:, i]
        
    # Add targets
    test_df['sbp_reference'] = all_targets[:, 0]
    test_df['dbp_reference'] = all_targets[:, 1]
    
    test_data_path = os.path.join(args.save_dir, 'test_data_split.csv')
    test_df.to_csv(test_data_path, index=False)
    logger.info(f"Test data split saved to: {test_data_path}")
    
    logger.info(f"\nTraining Complete! Best R² Score: {best_r2:.4f}")
    logger.info(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")


def plot_training_curves(
    train_losses: list,
    test_losses: list,
    r2_scores: list,
    save_dir: str
):
    """
    Plot and save training curves.
    
    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        r2_scores: List of R² scores
        save_dir: Directory to save plots
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', marker='s', markersize=3, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Loss Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # R² plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, r2_scores, label='R² Score', marker='d', markersize=3, 
             linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('R² Score Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")
    plt.show()

if __name__ == '__main__':
    main()
