import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(targets, predictions, save_dir, num_samples=100):
    """
    Plot predicted vs actual BP for a subset of samples.
    """
    # Randomly select samples
    indices = np.random.choice(len(targets), min(num_samples, len(targets)), replace=False)
    
    sbp_target = targets[indices, 0]
    sbp_pred = predictions[indices, 0]
    dbp_target = targets[indices, 1]
    dbp_pred = predictions[indices, 1]
    
    plt.figure(figsize=(12, 6))
    
    # SBP
    plt.subplot(1, 2, 1)
    plt.scatter(sbp_target, sbp_pred, alpha=0.5, color='blue')
    plt.plot([min(sbp_target), max(sbp_target)], [min(sbp_target), max(sbp_target)], 'r--')
    plt.xlabel('Actual SBP (mmHg)')
    plt.ylabel('Predicted SBP (mmHg)')
    plt.title('Systolic BP: Actual vs Predicted')
    plt.grid(True)
    
    # DBP
    plt.subplot(1, 2, 2)
    plt.scatter(dbp_target, dbp_pred, alpha=0.5, color='green')
    plt.plot([min(dbp_target), max(dbp_target)], [min(dbp_target), max(dbp_target)], 'r--')
    plt.xlabel('Actual DBP (mmHg)')
    plt.ylabel('Predicted DBP (mmHg)')
    plt.title('Diastolic BP: Actual vs Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'))
    plt.close()

def plot_training_curves(train_losses, val_losses, r2_scores, save_dir):
    """
    Plot and save training curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Loss Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # R² plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, r2_scores, label='Val R² Score', marker='d', markersize=3, 
             linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('R² Score Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
