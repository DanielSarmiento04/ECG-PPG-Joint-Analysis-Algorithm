import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
import argparse

def analyze_categorical(csv_path, output_dir='./data/figures'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=100000)
    os.makedirs(output_dir, exist_ok=True)

    # New categorical columns to analyze
    cat_cols = ['position', 'approach', 'aline1', 'preop_ecg', 'dx', 'opname']
    
    print("\n--- Categorical Variable Analysis ---")
    
    for col in cat_cols:
        if col not in df.columns:
            print(f"Skipping {col} (not found)")
            continue
            
        print(f"\nAnalysis for: {col}")
        print("-" * 30)
        
        # 1. Value Counts (Top 10 for high cardinality)
        counts = df[col].value_counts()
        print(f"Value Counts (Top 10):\n{counts.head(10)}\n")
        print(f"Total unique values: {len(counts)}")

        # Filter for ANOVA: only keep categories with > 50 samples to be robust
        valid_categories = counts[counts > 50].index
        df_filtered = df[df[col].isin(valid_categories)]
        
        # 2. BP Stats by Category (Top 5 for brevity in print)
        stats_df = df_filtered.groupby(col)[['sbp_reference', 'dbp_reference']].agg(['mean', 'std', 'count'])
        print(f"BP Statistics by {col} (Top 5):\n{stats_df.head(5)}\n")
        
        # 3. ANOVA Test (if more than 1 category)
        if len(valid_categories) > 1:
            groups_sbp = [group['sbp_reference'].values for name, group in df_filtered.groupby(col)]
            f_val, p_val = stats.f_oneway(*groups_sbp)
            print(f"One-way ANOVA for SBP: F={f_val:.2f}, p={p_val:.4e}")
            if p_val < 0.05:
                print("  -> Significant difference in SBP between groups! (Good predictor)")
            else:
                print("  -> No significant difference. (Likely noise)")
        
        # 4. Visualization (Only for low cardinality or Top N)
        if len(valid_categories) <= 10:
            plot_data = df_filtered
            rotation = 45
        else:
            # Take top 10 for plotting
            top_10 = counts.head(10).index
            plot_data = df[df[col].isin(top_10)]
            rotation = 90
            
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=plot_data, x=col, y='sbp_reference')
        plt.title(f'SBP by {col} (Top {len(plot_data[col].unique())})')
        plt.xticks(rotation=rotation)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=plot_data, x=col, y='dbp_reference')
        plt.title(f'DBP by {col} (Top {len(plot_data[col].unique())})')
        plt.xticks(rotation=rotation)
        
        plt.tight_layout()
        safe_col_name = col.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'bp_by_{safe_col_name}.png'))
        print(f"Saved boxplot to bp_by_{safe_col_name}.png")

if __name__ == "__main__":

    """Main training loop with advanced techniques."""
    parser = argparse.ArgumentParser(description='Train Temporal Transformer for BP Estimation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='src/data/bp_dataset_features.csv',
                       help='Path to the dataset file (CSV or Excel)')
    
    parser.add_argument('--output_dir', type=str, default='./data/figures',
                          help='Directory to save output figures')
    
    analyze_categorical(
        csv_path=parser.parse_args().data_path,
        output_dir=parser.parse_args().output_dir
    )
