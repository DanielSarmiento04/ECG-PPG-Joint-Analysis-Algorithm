import pandas as pd
import numpy as np
import argparse
import os
import sys

def analyze_data(file_path, output_dir, limit=None):
    print(f"Analyzing file: {file_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    # 1. DATA LOADING & STRUCTURE
    log("--- 1. DATA LOADING & STRUCTURE ---")
    try:
        # Read only first few rows to check structure first if needed, but we need full data for analysis
        # Using chunksize if memory is an issue, but let's try loading full first as 3M rows is manageable on many systems
        # If it fails, we might need to optimize.
        if limit is not None:
            df = pd.read_csv(file_path, nrows=limit)
            log(f"Successfully loaded {limit} rows from {file_path}")
        else:
            df = pd.read_csv(file_path)
            log(f"Successfully loaded {file_path}")
    except Exception as e:
        log(f"CRITICAL: Failed to load CSV: {e}")
        return

    log(f"Total rows: {len(df)}")
    log(f"Columns: {list(df.columns)}")
    
    # Check missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        log(f"Missing values found:\n{missing}")
    else:
        log("No missing values found in any column.")

    # Check dtypes
    log("\nData Types:")
    log(str(df.dtypes))

    # 2. TEMPORAL INTEGRITY
    log("\n--- 2. TEMPORAL INTEGRITY ---")
    if 'timestamp' in df.columns and 'patient_id' in df.columns:
        # Check monotonicity per patient
        # Group by patient_id (or session_id if patient_id is not unique per session)
        # Assuming session_id is unique per recording session.
        
        # Sort by session_id and timestamp just in case
        df_sorted = df.sort_values(['session_id', 'timestamp'])
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['session_id', 'timestamp']).sum()
        log(f"Duplicate timestamps within sessions: {duplicates}")
        
        # Check monotonicity
        # We can check if timestamp is always increasing
        # Calculate diff of timestamp within group
        df_sorted['time_diff'] = df_sorted.groupby('session_id')['timestamp'].diff()
        
        non_monotonic = (df_sorted['time_diff'] <= 0).sum()
        log(f"Non-monotonic timestamps count (<= 0 diff): {non_monotonic}")
        
        # Check for gaps (large jumps) - assuming 1Hz or similar, gaps > 2s might be issues?
        # The prompt mentions 30 timesteps ~ 30 seconds, so 1Hz.
        large_gaps = (df_sorted['time_diff'] > 5.0).sum() # Arbitrary 5s threshold
        log(f"Large gaps (> 5.0s) count: {large_gaps}")
        
        # Check cycle_index sequentiality
        if 'cycle_index' in df.columns:
            df_sorted['cycle_diff'] = df_sorted.groupby('session_id')['cycle_index'].diff()
            non_seq_cycles = (df_sorted['cycle_diff'] != 1).sum()
            # First item in group will be NaN, so subtract number of groups
            num_sessions = df['session_id'].nunique()
            non_seq_cycles -= num_sessions 
            log(f"Non-sequential cycle_index count: {non_seq_cycles}")

    else:
        log("Skipping Temporal Integrity: 'timestamp' or 'patient_id'/'session_id' missing.")

    # 3. TARGET VARIABLE SANITY
    log("\n--- 3. TARGET VARIABLE SANITY ---")
    targets = ['sbp_reference', 'dbp_reference'] # Based on head output
    for target in targets:
        if target in df.columns:
            col = df[target]
            log(f"Variable: {target}")
            log(f"  Min: {col.min()}, Max: {col.max()}, Mean: {col.mean():.2f}, Std: {col.std():.2f}")
            
            # Outliers (3 sigma)
            mean, std = col.mean(), col.std()
            outliers = ((col < mean - 3*std) | (col > mean + 3*std)).sum()
            log(f"  Outliers (3 sigma): {outliers} ({outliers/len(df)*100:.2f}%)")
            
            # Impossible values
            if target == 'sbp_reference':
                impossible = ((col < 0) | (col > 300)).sum()
                log(f"  Impossible values (<0 or >300): {impossible}")
            if target == 'dbp_reference':
                impossible = ((col < 0) | (col > 200)).sum()
                log(f"  Impossible values (<0 or >200): {impossible}")

    if 'sbp_reference' in df.columns and 'dbp_reference' in df.columns:
        sbp_lt_dbp = (df['sbp_reference'] <= df['dbp_reference']).sum()
        log(f"SBP <= DBP count: {sbp_lt_dbp}")

    # 4. FEATURE INTEGRITY
    log("\n--- 4. FEATURE INTEGRITY ---")
    features_to_check = {
        'ptt_peak_to_peak': (50, 400),
        'hr_bpm': (40, 180),
        'amplitude_ratio_ra': (0, 10)
    }
    
    for feat, (min_val, max_val) in features_to_check.items():
        if feat in df.columns:
            col = df[feat]
            log(f"Feature: {feat}")
            log(f"  Range: [{col.min()}, {col.max()}]")
            out_of_range = ((col < min_val) | (col > max_val)).sum()
            log(f"  Out of expected range [{min_val}, {max_val}]: {out_of_range} ({out_of_range/len(df)*100:.2f}%)")
        else:
            log(f"Feature {feat} not found.")

    # 5. CATEGORICAL ENCODING
    log("\n--- 5. CATEGORICAL ENCODING ---")
    cat_cols = ['sex', 'position', 'approach', 'aline1', 'preop_ecg']
    for col in cat_cols:
        if col in df.columns:
            log(f"Categorical: {col}")
            log(f"  Type: {df[col].dtype}")
            counts = df[col].value_counts(dropna=False)
            log(f"  Class counts:\n{counts}")
            log(f"  Unique values: {len(counts)}")
        else:
            log(f"Categorical {col} not found.")

    # 6. PATIENT-LEVEL VALIDATION
    log("\n--- 6. PATIENT-LEVEL VALIDATION ---")
    if 'patient_id' in df.columns:
        num_patients = df['patient_id'].nunique()
        log(f"Number of unique patients: {num_patients}")
        
        samples_per_patient = df.groupby('patient_id').size()
        log(f"Samples per patient: Min={samples_per_patient.min()}, Max={samples_per_patient.max()}, Median={samples_per_patient.median()}")
        
        short_patients = (samples_per_patient < 30).sum()
        log(f"Patients with < 30 samples: {short_patients}")
    
    if 'session_id' in df.columns and 'timestamp' in df.columns:
        session_durations = df.groupby('session_id')['timestamp'].apply(lambda x: x.max() - x.min())
        log(f"Session duration (s): Min={session_durations.min():.2f}, Max={session_durations.max():.2f}, Median={session_durations.median():.2f}")

    # Save report
    report_path = os.path.join(output_dir, 'data_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze data quality for BP prediction.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Directory to save report")
    parser.add_argument("--limit", type=int, help="Limit number of rows to read")
    
    args = parser.parse_args()
    
    analyze_data(args.input, args.output, args.limit)
