print("Starting analysis script...")
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyze_features(csv_path='./data/processed/bp_dataset_features.csv', output_dir='./data/figures'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

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
    sns.scatterplot(data=df, x='ptt_peak_to_peak', y='sbp_reference', alpha=0.1)
    plt.title('PTT (Peak-to-Peak) vs SBP')
    plt.xlabel('PTT (ms)')
    plt.ylabel('SBP (mmHg)')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='ptt_peak_to_peak', y='dbp_reference', alpha=0.1)
    plt.title('PTT (Peak-to-Peak) vs DBP')
    plt.xlabel('PTT (ms)')
    plt.ylabel('DBP (mmHg)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptt_vs_bp.png'))
    print("Saved PTT vs BP scatter plots.")
    
    print("\nAnalysis Complete.")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    analyze_features()
