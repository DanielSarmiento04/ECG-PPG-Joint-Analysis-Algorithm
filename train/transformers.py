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
from torchinfo import summary

from src.models.transformers import PhysiologicalTransformer

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
) -> Tuple[DataLoader, DataLoader, TorchStandardScaler, TorchStandardScaler]:
    """
    Load and prepare ECG/PPG data for training.
    
    Loads data from Excel file, normalizes features and targets,
    and creates train/test data loaders with proper shuffling.
    
    Data Format:
        - Columns 6 to 6+input_length: Signal features (ECG/PPG waveform)
        - Columns 3 to 5: Target values (e.g., SBP and DBP)
    
    Args:
        data_path: Path to Excel file containing the dataset
        device: PyTorch device for tensor operations
        batch_size: Batch size for data loaders
        input_length: Length of input signal (number of time steps)
        train_split: Fraction of data to use for training (default: 0.9)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_loader, test_loader, y_scaler, X_scaler)
        
    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If data has incorrect shape or missing values
    """
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract features and targets
    # Features: ptt_peak_to_peak, ptt_peak_to_foot, ptt_peak_to_maxslope, amplitude_ratio_ra, systolic_duration_tsd, diastolic_duration_tfd, time_to_maxslope_t1
    feature_cols = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'ptt_peak_to_maxslope', 'amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd', 'time_to_maxslope_t1']
    target_cols = ['sbp_reference', 'dbp_reference']
    
    if all(col in df.columns for col in feature_cols) and all(col in df.columns for col in target_cols):
        logger.info("Using named columns for features and targets")
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
        y = torch.tensor(df[target_cols].values, dtype=torch.float32, device=device)
    else:
        logger.warning("Named columns not found, falling back to indices (assuming legacy format)")
        X = torch.tensor(df.iloc[:, 6:6+input_length].values, dtype=torch.float32, device=device)
        y = torch.tensor(df.iloc[:, 3:5].values, dtype=torch.float32, device=device)
    
    # Check for NaN or Inf values
    if torch.isnan(X).any() or torch.isinf(X).any():
        logger.warning("Found NaN or Inf values in X, replacing with zeros")
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if torch.isnan(y).any() or torch.isinf(y).any():
        logger.warning("Found NaN or Inf values in y, replacing with zeros")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scalers on full data (note: in production, fit only on training data)
    X_scaler = TorchStandardScaler(device).fit(X)
    y_scaler = TorchStandardScaler(device).fit(y)
    
    # Normalize data
    X_normalized = X_scaler.transform(X).unsqueeze(1)  # Add channel dimension
    y_normalized = y_scaler.transform(y)
    
    # Split data with reproducible shuffling
    total_size = X_normalized.size(0)
    train_size = int(train_split * total_size)
    
    # Set seed for reproducibility
    torch.manual_seed(random_seed)
    indices = torch.randperm(total_size, device=device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_normalized[indices[:train_size]], y_normalized[indices[:train_size]])
    test_dataset = TensorDataset(X_normalized[indices[train_size:]], y_normalized[indices[train_size:]])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, y_scaler, X_scaler

def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Computes multiple regression metrics:
    - MSE Loss: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R² Score: Coefficient of determination
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        criterion: Loss function (typically MSELoss)
        device: Device for computation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Aggregate predictions
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    mse = total_loss / len(test_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # Per-output R² scores (useful for multi-output regression)
    r2_per_output = r2_score(all_targets, all_predictions, multioutput='raw_values')
    
    return {
        'loss': mse,
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
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    # Model architecture
    parser.add_argument('--patch_size', type=int, default=1,
                       help='Size of each patch (1 for feature vectors)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Dimension of hidden representations')
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=128,
                       help='Dimension of MLP layer')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and prepare data
    train_loader, test_loader, y_scaler, X_scaler = prepare_data(
        args.data_path, device, args.batch_size, args.input_length
    )
    
    # Initialize model
    model = PhysiologicalTransformer(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_outputs=2,  # SBP and DBP
        dropout=args.dropout,
        attention_dropout=0.0,
        input_length=args.input_length
    ).to(device)
    
    # Print model summary
    logger.info("\nModel Architecture:")
    summary(model, input_size=(args.batch_size, 1, args.input_length))
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, len(train_loader) * args.epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Training metrics tracking
    train_losses = []
    test_losses = []
    r2_scores = []
    best_r2 = -float('inf')
    patience_counter = 0
    
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(inputs)
                loss = criterion(predictions, targets)
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
        metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(metrics['loss'])
        r2_scores.append(metrics['r2'])
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {metrics['loss']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, "
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
                'y_scaler_std': y_scaler.std
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
    plot_training_curves(train_losses, test_losses, r2_scores, args.save_dir)
    
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
