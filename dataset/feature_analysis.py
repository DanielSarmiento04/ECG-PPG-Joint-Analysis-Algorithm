print("Starting analysis script...")
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse

def analyze_features(csv_path, output_dir='./data/figures'):
    if not os.path.exists(csv_path):
        # Fallback to raw features if cleaned doesn't exist
        csv_path = './data/processed/bp_dataset_features.csv'
        if not os.path.exists(csv_path):
            print(f"No dataset file found.")
            return
        print(f"Using raw features (cleaned not found)")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distributions of Target Variables
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='sbp_reference', kde=True, color='blue')
    plt.title('SBP Distribution')
    plt.xlabel('SBP (mmHg)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='dbp_reference', kde=True, color='red')
    plt.title('DBP Distribution')
    plt.xlabel('DBP (mmHg)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bp_distributions.png'))
    print("Saved BP distributions plot.")
    
    # 2. Feature Distributions
    features = ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'amplitude_ratio_ra', 'systolic_duration_tsd']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        if feature in df.columns:
            plt.subplot(2, 2, i+1)
            sns.histplot(data=df, x=feature, kde=True)
            plt.title(f'{feature} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    print("Saved feature distributions plot.")
    
    # 3. Correlation Matrix
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude some ID columns if they are numeric
    cols_to_corr = [c for c in numeric_cols if 'id' not in c and 'index' not in c]
    
    plt.figure(figsize=(12, 10))
    corr = df[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    print("Saved correlation matrix.")
    
    # 4. PTT vs BP Scatter Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Sample data if too large
    df_sample = df.sample(min(10000, len(df)), random_state=42)
    sns.scatterplot(data=df_sample, x='ptt_peak_to_peak', y='sbp_reference', alpha=0.3)
    plt.title('PTT (Peak-to-Peak) vs SBP')
    plt.xlabel('PTT (ms)')
    plt.ylabel('SBP (mmHg)')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_sample, x='ptt_peak_to_peak', y='dbp_reference', alpha=0.3)
    plt.title('PTT (Peak-to-Peak) vs DBP')
    plt.xlabel('PTT (ms)')
    plt.ylabel('DBP (mmHg)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptt_vs_bp.png'))
    print("Saved PTT vs BP scatter plots.")
    
    # 5. Per-patient PTT-BP correlation analysis
    print("\n" + "="*60)
    print("PER-PATIENT PTT-SBP CORRELATION ANALYSIS")
    print("="*60)
    
    patient_corrs = []
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id]
        if len(patient_data) > 50:  # Need enough samples
            corr = patient_data['ptt_peak_to_peak'].corr(patient_data['sbp_reference'])
            if not np.isnan(corr):
                patient_corrs.append({'patient_id': patient_id, 'correlation': corr, 'n_samples': len(patient_data)})
    
    if patient_corrs:
        corr_df = pd.DataFrame(patient_corrs)
        print(f"\nPatients analyzed: {len(corr_df)}")
        print(f"Mean correlation: {corr_df['correlation'].mean():.4f}")
        print(f"Median correlation: {corr_df['correlation'].median():.4f}")
        print(f"Std correlation: {corr_df['correlation'].std():.4f}")
        print(f"Patients with negative correlation: {(corr_df['correlation'] < 0).sum()} ({(corr_df['correlation'] < 0).mean()*100:.1f}%)")
        print(f"Patients with r < -0.1: {(corr_df['correlation'] < -0.1).sum()}")
        
        # Plot per-patient correlations
        plt.figure(figsize=(10, 5))
        plt.hist(corr_df['correlation'], bins=30, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Zero correlation')
        plt.axvline(x=corr_df['correlation'].mean(), color='green', linestyle='--', label=f'Mean: {corr_df["correlation"].mean():.3f}')
        plt.xlabel('PTT-SBP Correlation (r)')
        plt.ylabel('Number of Patients')
        plt.title('Per-Patient PTT-SBP Correlation Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_patient_correlations.png'))
        print("Saved per-patient correlation histogram.")
    
    # 6. Dataset summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"\nSBP: {df['sbp_reference'].mean():.1f} ± {df['sbp_reference'].std():.1f} mmHg")
    print(f"DBP: {df['dbp_reference'].mean():.1f} ± {df['dbp_reference'].std():.1f} mmHg")
    print(f"HR: {df['hr_bpm'].mean():.1f} ± {df['hr_bpm'].std():.1f} BPM")
    print(f"PTT (peak-to-peak): {df['ptt_peak_to_peak'].mean():.1f} ± {df['ptt_peak_to_peak'].std():.1f} ms")
    
    print("\nAnalysis Complete.")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":

    """Main training loop with advanced techniques."""
    parser = argparse.ArgumentParser(description='Train Temporal Transformer for BP Estimation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='src/data/bp_dataset_features.csv',
                       help='Path to the dataset file (CSV or Excel)')
    
    parser.add_argument('--output_dir', type=str, default='./data/figures',
                          help='Directory to save output figures')

    analyze_features(
        csv_path=parser.parse_args().data_path,
        output_dir=parser.parse_args().output_dir
    )
