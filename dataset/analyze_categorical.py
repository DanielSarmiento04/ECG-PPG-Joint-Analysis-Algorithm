import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats

def analyze_categorical(csv_path='./data/processed/bp_dataset_features.csv', output_dir='./data/figures'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # New categorical columns to analyze
    cat_cols = ['position', 'approach', 'aline1', 'preop_ecg']
    
    print("\n--- Categorical Variable Analysis ---")
    
    for col in cat_cols:
        if col not in df.columns:
            print(f"Skipping {col} (not found)")
            continue
            
        print(f"\nAnalysis for: {col}")
        print("-" * 30)
        
        # 1. Value Counts
        counts = df[col].value_counts()
        print(f"Value Counts:\n{counts}\n")
        
        # 2. BP Stats by Category
        stats_df = df.groupby(col)[['sbp_reference', 'dbp_reference']].agg(['mean', 'std', 'count'])
        print(f"BP Statistics by {col}:\n{stats_df}\n")
        
        # 3. ANOVA Test (if more than 1 category)
        if len(counts) > 1:
            groups_sbp = [group['sbp_reference'].values for name, group in df.groupby(col)]
            f_val, p_val = stats.f_oneway(*groups_sbp)
            print(f"One-way ANOVA for SBP: F={f_val:.2f}, p={p_val:.4e}")
            if p_val < 0.05:
                print("  -> Significant difference in SBP between groups!")
            else:
                print("  -> No significant difference.")
        
        # 4. Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x=col, y='sbp_reference')
        plt.title(f'SBP by {col}')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x=col, y='dbp_reference')
        plt.title(f'DBP by {col}')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        safe_col_name = col.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'bp_by_{safe_col_name}.png'))
        print(f"Saved boxplot to bp_by_{safe_col_name}.png")

if __name__ == "__main__":
    analyze_categorical()
