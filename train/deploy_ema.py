"""
Deployment Script: Real-time Blood Pressure Estimation (EMA Algorithm)

This script demonstrates the final "production" algorithm derived from the analysis.
Since the complex feature-based models failed to beat the simple calibration baseline,
this script implements the winning strategy: Exponential Moving Average (EMA).

Usage:
    This class can be embedded in a wearable device or monitoring system.
    It requires an initial calibration (first few readings) and then updates
    continuously.

Algorithm:
    BP_pred[t] = alpha * BP_measured[t-1] + (1 - alpha) * BP_pred[t-1]

    * In a real scenario without continuous cuff measurements, 'BP_measured' 
      would be an intermittent calibration measurement (e.g., every 30 mins).
    * Between calibrations, the prediction holds steady or drifts slowly.
"""

import numpy as np
import pandas as pd

class BPEstimator:
    def __init__(self, alpha_sbp=0.1, alpha_dbp=0.1):
        """
        Initialize the estimator.
        
        Args:
            alpha_sbp (float): Adaptation rate for SBP (0.0 to 1.0).
                               Higher = faster adaptation to new readings.
                               Lower = smoother, more stable predictions.
            alpha_dbp (float): Adaptation rate for DBP.
        """
        self.alpha_sbp = alpha_sbp
        self.alpha_dbp = alpha_dbp
        self.sbp_ema = None
        self.dbp_ema = None
        self.is_calibrated = False

    def calibrate(self, sbp_val, dbp_val):
        """
        Initial calibration or re-calibration event.
        Sets the baseline immediately to the provided value.
        """
        self.sbp_ema = sbp_val
        self.dbp_ema = dbp_val
        self.is_calibrated = True
        return self.sbp_ema, self.dbp_ema

    def update(self, sbp_measurement=None, dbp_measurement=None):
        """
        Update the estimator state.
        
        Args:
            sbp_measurement (float, optional): A new ground-truth measurement if available.
            dbp_measurement (float, optional): A new ground-truth measurement if available.
            
        Returns:
            (float, float): Current predicted SBP, DBP
        """
        if not self.is_calibrated:
            raise RuntimeError("Estimator must be calibrated before use.")

        # If we have a new measurement, update the EMA
        if sbp_measurement is not None and self.sbp_ema is not None:
            self.sbp_ema = self.alpha_sbp * sbp_measurement + (1 - self.alpha_sbp) * self.sbp_ema
            
        if dbp_measurement is not None and self.dbp_ema is not None:
            self.dbp_ema = self.alpha_dbp * dbp_measurement + (1 - self.alpha_dbp) * self.dbp_ema
            
        # If no new measurement, we return the current EMA (hold value)
        # In a more complex version, we could use PTT trends here to adjust the EMA
        # but our analysis showed PTT features added no value.
        
        return self.sbp_ema, self.dbp_ema

def simulate_patient_run(patient_id, df):
    """
    Simulates a real-time run for a single patient.
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    
    # Initialize estimator
    estimator = BPEstimator(alpha_sbp=0.1, alpha_dbp=0.1)
    
    predictions = []
    actuals = []
    
    print(f"Simulating Patient {patient_id} ({len(patient_data)} cycles)...")
    
    # Run simulation
    for i, row in enumerate(patient_data.itertuples()):
        sbp_true = row.sbp_reference
        dbp_true = row.dbp_reference
        
        if i == 0:
            # First beat: Calibrate
            pred_sbp, pred_dbp = estimator.calibrate(sbp_true, dbp_true)
        else:
            # Subsequent beats: Predict FIRST, then Update (simulate real-time lag)
            # We predict for time t using info up to t-1
            pred_sbp, pred_dbp = estimator.update(sbp_measurement=None, dbp_measurement=None)
            
            # After prediction, we assume we get the measurement (for the NEXT step)
            # In a real device, this 'update' might only happen every N minutes.
            # Here we update every beat to match the 'EMA Baseline' performance.
            estimator.update(sbp_measurement=sbp_true, dbp_measurement=dbp_true)
            
        predictions.append((pred_sbp, pred_dbp))
        actuals.append((sbp_true, dbp_true))
        
    # Calculate Error
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    sbp_error = preds[:, 0] - acts[:, 0]
    dbp_error = preds[:, 1] - acts[:, 1]
    
    print(f"  SBP ME: {np.mean(sbp_error):.2f}, STD: {np.std(sbp_error):.2f}")
    print(f"  DBP ME: {np.mean(dbp_error):.2f}, STD: {np.std(dbp_error):.2f}")
    
    return preds, acts

if __name__ == "__main__":
    # Load a sample of data
    print("Loading data...")
    df = pd.read_csv('src/data/bp_dataset_strict.csv')
    
    # Pick 3 random patients
    sample_patients = df['patient_id'].unique()[:3]
    
    for pid in sample_patients:
        simulate_patient_run(pid, df)
