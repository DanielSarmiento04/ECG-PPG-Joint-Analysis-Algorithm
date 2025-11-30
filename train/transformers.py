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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
# from torchinfo import summary

from src.models.transformers import TemporalPhysiologicalTransformer
from src.utils.hardware import get_device
from src.utils.losses import PhysiologicalLoss
from src.utils.visualizations import plot_training_curves, plot_predictions

from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for Temporal Physiological Transformer.
    Returns sliding windows of dynamic features and corresponding static features.
    """
    def __init__(self, sequences, static_cont, static_cat, targets):
        self.sequences = sequences # List of tensors [Seq_Len, Features]
        self.static_cont = static_cont # List of tensors [Num_Cont]
        self.static_cat = static_cat # List of tensors [Num_Cat]
        self.targets = targets # List of tensors [2] (SBP, DBP)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.static_cont[idx], self.static_cat[idx], self.targets[idx]

def create_sequences(df, patient_col, dynamic_cols, static_cont_cols, static_cat_cols, target_cols, seq_len=30):
    """
    Create sliding window sequences grouped by patient.
    """
    sequences = []
    statics_cont = []
    statics_cat = []
    targets = []
    
    # Group by patient to ensure windows don't cross patient boundaries
    # Use tqdm for progress on patients
    patient_groups = list(df.groupby(patient_col))
    
    for pid, group in tqdm(patient_groups, desc="Creating sequences"):
        # Sort by time if a timestamp exists, otherwise assume index is temporal
        # Assuming data is already sorted by time per patient in the CSV
        
        data_dyn = group[dynamic_cols].values
        data_stat_cont = group[static_cont_cols].values
        data_stat_cat = group[static_cat_cols].values
        data_tgt = group[target_cols].values
        
        if len(group) <= seq_len:
            continue
            
        # Create sliding windows
        # We can vectorize this for speed if needed, but loop is clearer
        # Fix: Iterate to allow using the last window
        for i in range(len(group) - seq_len + 1):
            # Input: T steps
            seq = data_dyn[i : i + seq_len]
            # Static: Constant for the window (take last step)
            stat_cont = data_stat_cont[i + seq_len - 1]
            stat_cat = data_stat_cat[i + seq_len - 1]
            # Target: The BP at the END of the window (Estimation, not Forecasting)
            tgt = data_tgt[i + seq_len - 1]
            
            sequences.append(torch.tensor(seq, dtype=torch.float32))
            statics_cont.append(torch.tensor(stat_cont, dtype=torch.float32))
            statics_cat.append(torch.tensor(stat_cat, dtype=torch.long))
            targets.append(torch.tensor(tgt, dtype=torch.float32))
            
    return sequences, statics_cont, statics_cat, targets

def create_optimizer_and_scheduler(
    model: nn.Module, 
    args: argparse.Namespace, 
    num_training_steps: int
) -> Tuple[AdamW, Any]:
    """
    Create optimizer with weight decay and learning rate scheduler.
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
    
    # Use ReduceLROnPlateau as requested
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
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

def prepare_data_temporal(
    data_path: str, 
    device: torch.device, 
    batch_size: int, 
    seq_length: int = 30,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, TorchStandardScaler, TorchStandardScaler, TorchStandardScaler, int, int, list]:
    """
    Load and prepare ECG/PPG data for Temporal Transformer.
    Splits data by PATIENT ID to prevent leakage.
    Returns: Train, Val, Test loaders
    """
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Ensure patient_id exists
    if 'patient_id' not in df.columns:
        logger.warning("patient_id column not found! Using dummy patient IDs (High Risk of Leakage)")
        df['patient_id'] = 0
    
    # --- 1. Dynamic Features (Time-Series) ---
    dynamic_cols = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'ptt_peak_to_maxslope', 
                    'amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd', 
                    'time_to_maxslope_t1', 'hr_bpm', 'cycle_correlation']
    
    # Check which exist
    dynamic_cols = [c for c in dynamic_cols if c in df.columns]
    
    # --- 2. Static Features (Demographics + Categorical) ---
    # Numerical Static
    num_static_cols = ['age', 'bmi']
    # Categorical Static
    cat_static_cols = ['sex', 'position', 'approach', 'aline1', 'preop_ecg', 'dx', 'opname']
    
    target_cols = ['sbp_reference', 'dbp_reference']
    
    # Fill missing
    for c in num_static_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mean())
        else:
            df[c] = 0.0
            
    for c in cat_static_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')
        else:
            df[c] = 'Unknown'
            
    # Encode Categoricals
    final_static_cont_cols = [c for c in num_static_cols if c in df.columns]
    final_static_cat_cols = []
    cat_cardinalities = []
    
    for col in cat_static_cols:
        if col in df.columns:
            le = LabelEncoder()
            col_data = df[col].astype(str).tolist()
            df[col + '_enc'] = le.fit_transform(col_data)
            final_static_cat_cols.append(col + '_enc')
            cat_cardinalities.append(len(le.classes_))
            logger.info(f"Encoded {col}: {len(le.classes_)} classes")

    # --- Patient-Wise Split ---
    patient_ids = df['patient_id'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(patient_ids)
    
    # First split: Train+Val vs Test
    split_idx_test = int(len(patient_ids) * train_split)
    dev_pids = patient_ids[:split_idx_test]
    test_pids = patient_ids[split_idx_test:]
    
    # Second split: Train vs Val (90/10 of Dev)
    split_idx_val = int(len(dev_pids) * 0.9)
    train_pids = dev_pids[:split_idx_val]
    val_pids = dev_pids[split_idx_val:]
    
    logger.info(f"Splitting by Patient: {len(train_pids)} Train, {len(val_pids)} Val, {len(test_pids)} Test")
    
    train_df = df[df['patient_id'].isin(train_pids)].copy()
    val_df = df[df['patient_id'].isin(val_pids)].copy()
    test_df = df[df['patient_id'].isin(test_pids)].copy()
    
    # --- Normalization ---
    # Fit scalers ONLY on Training Data
    
    # Dynamic Features
    X_dyn_train = torch.tensor(train_df[dynamic_cols].values, dtype=torch.float32, device=device)
    dyn_scaler = TorchStandardScaler(device).fit(X_dyn_train)
    
    # Static Continuous Features
    X_stat_cont_train = torch.tensor(train_df[final_static_cont_cols].values, dtype=torch.float32, device=device)
    stat_scaler = TorchStandardScaler(device).fit(X_stat_cont_train)
    
    # Targets
    y_train = torch.tensor(train_df[target_cols].values, dtype=torch.float32, device=device)
    y_scaler = TorchStandardScaler(device).fit(y_train)
    
    # Apply Normalization
    # Transform Train
    train_df[dynamic_cols] = dyn_scaler.transform(X_dyn_train).cpu().numpy()
    train_df[final_static_cont_cols] = stat_scaler.transform(X_stat_cont_train).cpu().numpy()
    train_df[target_cols] = y_scaler.transform(y_train).cpu().numpy()
    
    # Transform Val
    X_dyn_val = torch.tensor(val_df[dynamic_cols].values, dtype=torch.float32, device=device)
    X_stat_cont_val = torch.tensor(val_df[final_static_cont_cols].values, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_df[target_cols].values, dtype=torch.float32, device=device)
    
    val_df[dynamic_cols] = dyn_scaler.transform(X_dyn_val).cpu().numpy()
    val_df[final_static_cont_cols] = stat_scaler.transform(X_stat_cont_val).cpu().numpy()
    val_df[target_cols] = y_scaler.transform(y_val).cpu().numpy()
    
    # Transform Test
    X_dyn_test = torch.tensor(test_df[dynamic_cols].values, dtype=torch.float32, device=device)
    X_stat_cont_test = torch.tensor(test_df[final_static_cont_cols].values, dtype=torch.float32, device=device)
    y_test = torch.tensor(test_df[target_cols].values, dtype=torch.float32, device=device)
    
    test_df[dynamic_cols] = dyn_scaler.transform(X_dyn_test).cpu().numpy()
    test_df[final_static_cont_cols] = stat_scaler.transform(X_stat_cont_test).cpu().numpy()
    test_df[target_cols] = y_scaler.transform(y_test).cpu().numpy()
    
    # --- Create Sequences ---
    logger.info("Creating training sequences...")
    train_seq, train_stat_cont, train_stat_cat, train_tgt = create_sequences(
        train_df, 'patient_id', dynamic_cols, final_static_cont_cols, final_static_cat_cols, target_cols, seq_length
    )
    
    logger.info("Creating validation sequences...")
    val_seq, val_stat_cont, val_stat_cat, val_tgt = create_sequences(
        val_df, 'patient_id', dynamic_cols, final_static_cont_cols, final_static_cat_cols, target_cols, seq_length
    )
    
    logger.info("Creating testing sequences...")
    test_seq, test_stat_cont, test_stat_cat, test_tgt = create_sequences(
        test_df, 'patient_id', dynamic_cols, final_static_cont_cols, final_static_cat_cols, target_cols, seq_length
    )
    
    # Create Datasets
    train_dataset = SequenceDataset(train_seq, train_stat_cont, train_stat_cat, train_tgt)
    val_dataset = SequenceDataset(val_seq, val_stat_cont, val_stat_cat, val_tgt)
    test_dataset = SequenceDataset(test_seq, test_stat_cont, test_stat_cat, test_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, y_scaler, dyn_scaler, stat_scaler, len(dynamic_cols), len(final_static_cont_cols), cat_cardinalities

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
        for inputs, x_static_cont, x_static_cat, targets in test_loader:
            inputs = inputs.to(device)
            x_static_cont = x_static_cont.to(device)
            x_static_cat = x_static_cat.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs, x_static_cont, x_static_cat)
            
            # Pass y_scaler to criterion if it's WeightedMSELoss or PhysiologicalLoss
            if isinstance(criterion, (WeightedMSELoss, PhysiologicalLoss)):
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
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # AAMI Standard Check
    errors = all_predictions - all_targets
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    
    logger.info(f"AAMI Stats - SBP: ME={mean_error[0]:.2f}, STD={std_error[0]:.2f}")
    logger.info(f"AAMI Stats - DBP: ME={mean_error[1]:.2f}, STD={std_error[1]:.2f}")
    
    return {
        'loss': total_loss / len(test_loader),
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """Main training loop with advanced techniques."""
    parser = argparse.ArgumentParser(description='Train Temporal Transformer for BP Estimation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='src/data/bp_dataset_features.csv',
                       help='Path to the dataset file (CSV or Excel)')
    parser.add_argument('--seq_length', type=int, default=30,
                       help='Sequence length (number of beats history)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Dimension of hidden representations')
    parser.add_argument('--depth', type=int, default=4,
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='training.log',
                       help='Path to log file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging to file
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and prepare data (Temporal)
    train_loader, val_loader, test_loader, y_scaler, dyn_scaler, stat_scaler, num_dyn, num_stat_cont, cat_cardinalities = prepare_data_temporal(
        args.data_path, device, args.batch_size, args.seq_length, random_seed=args.seed
    )
    
    # Initialize model
    model = TemporalPhysiologicalTransformer(
        num_dynamic_features=num_dyn,
        num_continuous_static=num_stat_cont,
        categorical_cardinalities=cat_cardinalities,
        d_model=args.hidden_size,
        nhead=args.num_heads,
        num_layers=args.depth,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model initialized with {num_dyn} dynamic, {num_stat_cont} static continuous features, and {len(cat_cardinalities)} categorical features.")
    
    # Loss function and optimizer
    # Use PhysiologicalLoss as requested
    criterion = PhysiologicalLoss(device=device, use_huber=True)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, len(train_loader) * args.epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    r2_scores = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, x_static_cont, x_static_cat, targets in pbar:
            inputs = inputs.to(device)
            x_static_cont = x_static_cont.to(device)
            x_static_cat = x_static_cat.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = model(inputs, x_static_cont, x_static_cat)
                    loss = criterion(predictions, targets, y_scaler)
                
                scaler.scale(loss).backward()
                
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(inputs, x_static_cont, x_static_cat)
                loss = criterion(predictions, targets, y_scaler)
                loss.backward()
                
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            # Scheduler step (OneCycleLR steps per batch, but ReduceLROnPlateau steps per epoch)
            # We switched to ReduceLROnPlateau, so we step after validation
            # scheduler.step() 
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        logger.info("Validating...")
        val_metrics = evaluate_model(model, val_loader, criterion, device, y_scaler)
        
        # Step scheduler based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_metrics['loss'])
        r2_scores.append(val_metrics['r2'])
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val RMSE: {val_metrics['rmse']:.4f}, "
            f"Val MAE: {val_metrics['mae']:.4f}, "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        
        # Save best model based on Validation Loss (or R2)
        # Using Loss for early stopping as requested
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'y_scaler_mean': y_scaler.mean,
                'y_scaler_std': y_scaler.std,
                'dyn_scaler_mean': dyn_scaler.mean,
                'dyn_scaler_std': dyn_scaler.std,
                'stat_scaler_mean': stat_scaler.mean,
                'stat_scaler_std': stat_scaler.std
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            logger.info(f"  ✓ New best Val Loss: {best_val_loss:.4f} - Model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
            
    # Final Evaluation on Test Set
    logger.info("\nFinal Evaluation on Test Set...")
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, criterion, device, y_scaler)
    logger.info(
        f"TEST RESULTS - "
        f"Loss: {test_metrics['loss']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, "
        f"MAE: {test_metrics['mae']:.4f}, "
        f"R²: {test_metrics['r2']:.4f}"
    )
    
    plot_training_curves(train_losses, val_losses, r2_scores, args.save_dir)
    
    # Plot predictions for a subset of test data
    # We need to get predictions first. evaluate_model doesn't return them.
    # Let's just run a quick pass to get some predictions for plotting
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, x_static_cont, x_static_cat, targets in test_loader:
            inputs = inputs.to(device)
            x_static_cont = x_static_cont.to(device)
            x_static_cat = x_static_cat.to(device)
            targets = targets.to(device)
            predictions = model(inputs, x_static_cont, x_static_cat)
            
            if y_scaler is not None:
                targets = y_scaler.inverse_transform(targets)
                predictions = y_scaler.inverse_transform(predictions)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            if len(all_targets) * args.batch_size > 200: # Just get enough for plotting
                break
                
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    plot_predictions(all_targets, all_predictions, args.save_dir)
    
    logger.info(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
