import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMILoss(nn.Module):
    """
    Loss function designed to optimize for AAMI standards (ME and STD).
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        # pred, target: [Batch, 2] (SBP, DBP)
        error = pred - target
        me = torch.mean(error, dim=0)
        std = torch.std(error, dim=0)
        
        # Minimize ME^2 + STD^2 for both SBP and DBP
        loss_sbp = me[0]**2 + std[0]**2
        loss_dbp = me[1]**2 + std[1]**2
        
        return self.alpha * loss_sbp + self.beta * loss_dbp

class PhysiologicalLoss(nn.Module):
    """
    Multi-component loss function for BP prediction.
    Includes:
    - MSE/Huber for SBP and DBP
    - Physiological constraints (SBP > DBP)
    - Range penalties
    """
    def __init__(self, device, alpha=1.0, beta=1.0, use_huber=True, delta=1.0):
        super().__init__()
        self.device = device
        self.alpha = alpha # Weight for SBP
        self.beta = beta   # Weight for DBP
        self.use_huber = use_huber
        self.delta = delta
        
    def forward(self, pred, target, y_scaler=None):
        # pred, target are scaled values (approx N(0,1))
        
        # 1. Main Regression Loss (Huber or MSE) on SCALED values
        # This ensures gradients are well-behaved (not scaled by std^2 ~ 400)
        if self.use_huber:
            # delta=1.0 in scaled space corresponds to ~20mmHg error
            # This acts mostly as MSE for typical errors, and L1 for outliers > 20mmHg
            loss_sbp = F.huber_loss(pred[:, 0], target[:, 0], delta=self.delta)
            loss_dbp = F.huber_loss(pred[:, 1], target[:, 1], delta=self.delta)
        else:
            loss_sbp = F.mse_loss(pred[:, 0], target[:, 0])
            loss_dbp = F.mse_loss(pred[:, 1], target[:, 1])
            
        # Unscale for physiological constraints
        if y_scaler is not None:
            pred_mmhg = y_scaler.inverse_transform(pred)
        else:
            pred_mmhg = pred
            
        sbp_pred_mmhg = pred_mmhg[:, 0]
        dbp_pred_mmhg = pred_mmhg[:, 1]
        
        # 2. Physiological Constraint: SBP > DBP
        # Penalty if DBP > SBP
        constraint_loss = torch.mean(F.relu(dbp_pred_mmhg - sbp_pred_mmhg))
        
        # 3. Pulse Pressure Consistency
        pp = sbp_pred_mmhg - dbp_pred_mmhg
        pp_low_penalty = torch.mean(F.relu(10.0 - pp))
        pp_high_penalty = torch.mean(F.relu(pp - 150.0))
        
        # 4. Range Penalty
        sbp_range_penalty = torch.mean(F.relu(40.0 - sbp_pred_mmhg) + F.relu(sbp_pred_mmhg - 250.0))
        dbp_range_penalty = torch.mean(F.relu(20.0 - dbp_pred_mmhg) + F.relu(dbp_pred_mmhg - 150.0))
        
        # Scale penalties down to match the magnitude of scaled loss
        # Scaled loss is ~1.0. Unscaled penalties can be ~10-100.
        # We multiply penalties by 0.01 to keep them as "soft" constraints
        penalty_weight = 0.01
        
        total_loss = (self.alpha * loss_sbp + self.beta * loss_dbp + 
                      penalty_weight * (10.0 * constraint_loss + 
                                      1.0 * pp_low_penalty + 1.0 * pp_high_penalty +
                                      1.0 * sbp_range_penalty + 1.0 * dbp_range_penalty))
                      
        return total_loss
