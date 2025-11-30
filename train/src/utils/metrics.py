import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(targets, predictions):
    """
    Calculate comprehensive metrics for BP prediction.
    
    Args:
        targets: [N, 2] array (SBP, DBP)
        predictions: [N, 2] array (SBP, DBP)
        
    Returns:
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # AAMI Standards
    errors = predictions - targets
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'sbp_me': mean_error[0],
        'sbp_std': std_error[0],
        'dbp_me': mean_error[1],
        'dbp_std': std_error[1]
    }
    
    return metrics

def check_aami_compliance(metrics):
    """
    Check if metrics meet AAMI standards (ME <= 5, STD <= 8).
    """
    sbp_pass = (abs(metrics['sbp_me']) <= 5) and (metrics['sbp_std'] <= 8)
    dbp_pass = (abs(metrics['dbp_me']) <= 5) and (metrics['dbp_std'] <= 8)
    
    return sbp_pass and dbp_pass
