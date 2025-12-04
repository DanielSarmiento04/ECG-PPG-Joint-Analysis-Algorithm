#!/usr/bin/env python3
"""
Detailed boundary analysis for PTT and Amplitude Ratio.
Identifies anomalies and clustering at boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def analyze_ptt_boundaries(df, output_dir='./data/figures'):
    """Detailed PTT boundary analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PTT BOUNDARY ANALYSIS")
    print("=" * 70)
    
    # Check which PTT columns exist
    ptt_cols = ['ptt_peak_to_foot', 'pat_ecg_ppg', 'ptt_peak_to_peak', 'pat_to_peak']
    available_ptt = [c for c in ptt_cols if c in df.columns]
    
    print(f"\nAvailable PTT columns: {available_ptt}")
    
    for ptt_col in available_ptt:
        ptt = df[ptt_col].dropna()
        
        print(f"\n--- {ptt_col} ---")
        print(f"Total samples: {len(ptt):,}")
        print(f"Mean: {ptt.mean():.2f} ms")
        print(f"Std: {ptt.std():.2f} ms")
        print(f"Min: {ptt.min():.2f} ms")
        print(f"Max: {ptt.max():.2f} ms")
        
        # Check specific boundaries
        boundaries = [40, 50, 80, 100, 120, 200, 248, 250, 300]
        print(f"\nBoundary clustering (±1ms):")
        for b in boundaries:
            count = ((ptt >= b-1) & (ptt <= b+1)).sum()
            pct = count / len(ptt) * 100
            if pct > 0.5:  # Only show if > 0.5%
                print(f"  {b}ms: {count:,} ({pct:.2f}%)")
        
        # Check ranges
        print(f"\nRange distribution:")
        ranges = [(0, 40), (40, 50), (50, 80), (80, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 500)]
        for low, high in ranges:
            count = ((ptt >= low) & (ptt < high)).sum()
            pct = count / len(ptt) * 100
            if pct > 0.1:
                print(f"  {low}-{high}ms: {count:,} ({pct:.1f}%)")
    
    # Create detailed histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, ptt_col in enumerate(available_ptt[:4]):
        ax = axes[idx // 2, idx % 2]
        ptt = df[ptt_col].dropna()
        
        # Histogram with fine bins
        bins = np.arange(0, min(400, ptt.max() + 10), 5)
        ax.hist(ptt, bins=bins, edgecolor='black', alpha=0.7)
        
        # Mark boundaries
        for b in [40, 80, 100, 200, 250]:
            count = ((ptt >= b-1) & (ptt <= b+1)).sum()
            pct = count / len(ptt) * 100
            if pct > 1:
                ax.axvline(x=b, color='red', linestyle='--', alpha=0.7)
                ax.text(b, ax.get_ylim()[1]*0.9, f'{b}ms\n{pct:.1f}%', 
                       ha='center', fontsize=8, color='red')
        
        ax.set_xlabel(f'{ptt_col} (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'{ptt_col} Distribution\n(n={len(ptt):,}, mean={ptt.mean():.1f}ms)')
        
        # Add stats box
        stats_text = f'Min: {ptt.min():.0f}\nMax: {ptt.max():.0f}\nStd: {ptt.std():.1f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptt_boundary_detailed.png'), dpi=150)
    plt.close()
    print(f"\nSaved: ptt_boundary_detailed.png")


def analyze_amplitude_ratio_boundaries(df, output_dir='./data/figures'):
    """Detailed amplitude ratio boundary analysis."""
    
    print("\n" + "=" * 70)
    print("AMPLITUDE RATIO BOUNDARY ANALYSIS")
    print("=" * 70)
    
    amp_col = 'amplitude_ratio_ra'
    if amp_col not in df.columns:
        print(f"Column {amp_col} not found")
        return
    
    amp = df[amp_col].dropna()
    
    print(f"\nTotal samples: {len(amp):,}")
    print(f"Mean: {amp.mean():.4f}")
    print(f"Std: {amp.std():.4f}")
    print(f"Min: {amp.min():.4f}")
    print(f"Max: {amp.max():.4f}")
    
    # Check specific boundaries
    print(f"\nBoundary clustering:")
    boundaries = [0, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 0.95, 0.98, 0.99, 1.0]
    for b in boundaries:
        if b == 0:
            count = (amp == 0).sum()
        elif b == 1:
            count = (amp == 1).sum()
        else:
            count = ((amp >= b-0.01) & (amp <= b+0.01)).sum()
        pct = count / len(amp) * 100
        print(f"  {b:.2f}: {count:,} ({pct:.2f}%)")
    
    # Check ranges
    print(f"\nRange distribution:")
    ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), 
              (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in ranges:
        count = ((amp >= low) & (amp < high)).sum()
        pct = count / len(amp) * 100
        print(f"  {low:.1f}-{high:.1f}: {count:,} ({pct:.1f}%)")
    
    # Identify exact values that appear too often (potential artifacts)
    print(f"\nMost common exact values:")
    value_counts = amp.round(3).value_counts().head(20)
    for val, count in value_counts.items():
        pct = count / len(amp) * 100
        if pct > 0.5:
            print(f"  {val:.3f}: {count:,} ({pct:.2f}%)")
    
    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full range
    ax1 = axes[0]
    ax1.hist(amp, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.05, color='red', linestyle='--', label='0.05 (low boundary)')
    ax1.axvline(x=0.95, color='orange', linestyle='--', label='0.95 (high boundary)')
    ax1.set_xlabel('Amplitude Ratio')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Amplitude Ratio Distribution\n(n={len(amp):,})')
    ax1.legend()
    
    # Zoomed to boundaries
    ax2 = axes[1]
    ax2.hist(amp[amp < 0.2], bins=40, edgecolor='black', alpha=0.7, label='Low end')
    ax2.set_xlabel('Amplitude Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Zoomed: Low End (< 0.2)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amplitude_ratio_boundary_detailed.png'), dpi=150)
    plt.close()
    print(f"\nSaved: amplitude_ratio_boundary_detailed.png")


def analyze_ptt_vs_sbp_anomalies(df, output_dir='./data/figures'):
    """Analyze anomalies in PTT vs SBP relationship."""
    
    print("\n" + "=" * 70)
    print("PTT vs SBP ANOMALY ANALYSIS")
    print("=" * 70)
    
    ptt_col = 'ptt_peak_to_foot' if 'ptt_peak_to_foot' in df.columns else 'pat_ecg_ppg'
    
    if ptt_col not in df.columns or 'sbp_reference' not in df.columns:
        print("Required columns not found")
        return
    
    # Clean data
    mask = df[ptt_col].notna() & df['sbp_reference'].notna()
    ptt = df.loc[mask, ptt_col]
    sbp = df.loc[mask, 'sbp_reference']
    
    print(f"\nSamples with both PTT and SBP: {len(ptt):,}")
    
    # Check correlation at different PTT ranges
    print(f"\nCorrelation by PTT range:")
    ptt_ranges = [(40, 60), (60, 80), (80, 100), (100, 120), (120, 150), (150, 200), (200, 300)]
    
    for low, high in ptt_ranges:
        mask_range = (ptt >= low) & (ptt < high)
        if mask_range.sum() > 50:
            corr = ptt[mask_range].corr(sbp[mask_range])
            print(f"  PTT {low}-{high}ms: n={mask_range.sum():,}, r={corr:.4f}")
    
    # Per-patient analysis
    if 'patient_id' in df.columns:
        print(f"\nPer-patient PTT-SBP correlation:")
        patient_stats = []
        for pid in df['patient_id'].unique():
            p_mask = df['patient_id'] == pid
            p_ptt = df.loc[p_mask & mask, ptt_col]
            p_sbp = df.loc[p_mask & mask, 'sbp_reference']
            
            if len(p_ptt) >= 100:
                corr = p_ptt.corr(p_sbp)
                ptt_mean = p_ptt.mean()
                ptt_at_boundary = ((p_ptt >= 79) & (p_ptt <= 81)).sum() / len(p_ptt) * 100
                
                patient_stats.append({
                    'patient_id': pid,
                    'n_samples': len(p_ptt),
                    'ptt_sbp_r': corr,
                    'ptt_mean': ptt_mean,
                    'ptt_at_80ms': ptt_at_boundary
                })
        
        if patient_stats:
            stats_df = pd.DataFrame(patient_stats).sort_values('ptt_sbp_r')
            
            print(f"\n  Patients with WRONG correlation (r > 0):")
            wrong = stats_df[stats_df['ptt_sbp_r'] > 0]
            for _, row in wrong.iterrows():
                print(f"    {row['patient_id']}: r={row['ptt_sbp_r']:.3f}, "
                      f"PTT_mean={row['ptt_mean']:.0f}ms, at_80ms={row['ptt_at_80ms']:.1f}%")
            
            print(f"\n  Patients with CORRECT correlation (r < -0.1):")
            correct = stats_df[stats_df['ptt_sbp_r'] < -0.1]
            for _, row in correct.head(10).iterrows():
                print(f"    {row['patient_id']}: r={row['ptt_sbp_r']:.3f}, "
                      f"PTT_mean={row['ptt_mean']:.0f}ms, at_80ms={row['ptt_at_80ms']:.1f}%")
    
    # Create scatter plot colored by patient
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample for visualization
    sample_idx = np.random.choice(len(ptt), min(5000, len(ptt)), replace=False)
    ptt_sample = ptt.iloc[sample_idx]
    sbp_sample = sbp.iloc[sample_idx]
    
    ax1 = axes[0]
    ax1.scatter(ptt_sample, sbp_sample, alpha=0.3, s=5)
    ax1.set_xlabel(f'{ptt_col} (ms)')
    ax1.set_ylabel('SBP (mmHg)')
    ax1.set_title(f'PTT vs SBP\nr={ptt.corr(sbp):.4f}')
    
    # Add trend line
    z = np.polyfit(ptt_sample, sbp_sample, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ptt_sample.min(), ptt_sample.max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend: slope={z[0]:.2f}')
    ax1.legend()
    
    # Highlight boundary points
    ax2 = axes[1]
    colors = ['red' if (p >= 79 and p <= 81) else 'blue' for p in ptt_sample]
    ax2.scatter(ptt_sample, sbp_sample, c=colors, alpha=0.3, s=5)
    ax2.set_xlabel(f'{ptt_col} (ms)')
    ax2.set_ylabel('SBP (mmHg)')
    ax2.set_title('PTT vs SBP (Red = at 80ms boundary)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptt_sbp_anomaly_analysis.png'), dpi=150)
    plt.close()
    print(f"\nSaved: ptt_sbp_anomaly_analysis.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Boundary Analysis')
    parser.add_argument('--data_path', type=str, default='./data/processed/bp_dataset_features.csv')
    parser.add_argument('--output_dir', type=str, default='./data/figures')
    args = parser.parse_args()
    
    print(f"Loading: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df):,} samples from {df['patient_id'].nunique()} patients\n")
    
    analyze_ptt_boundaries(df, args.output_dir)
    analyze_amplitude_ratio_boundaries(df, args.output_dir)
    analyze_ptt_vs_sbp_anomalies(df, args.output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
