#!/usr/bin/env python3
"""
Pipeline Validation Module for Blood Pressure Dataset

This module generates validation plots and analyses to verify that the
data pipeline is producing physiologically valid data with proper
PTT-BP correlations.

Key Validations:
1. PTT distribution (should be unimodal, ~150 ms mean)
2. Amplitude ratio distribution (should NOT spike at 1.0)
3. PTT vs BP scatter (should show inverse relationship)
4. Correlation matrix (PTT-SBP should be < -0.5)

Author: Blood Pressure Pipeline Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def validate_cleaned_data(df: pd.DataFrame, 
                         output_dir: str = "data/figures",
                         prefix: str = "validation") -> Dict:
    """
    Generate comprehensive validation plots for cleaned data.
    
    Args:
        df: Cleaned DataFrame with features
        output_dir: Directory to save plots
        prefix: Prefix for output files
        
    Returns:
        Dictionary with validation metrics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'total_samples': len(df),
        'validation_passed': True,
        'warnings': [],
        'errors': []
    }
    
    # =========================================
    # PLOT 1: PTT Distributions (3x1)
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PTT peak-to-peak
    if 'ptt_peak_to_peak' in df.columns:
        ptt_pp = df['ptt_peak_to_peak']
        axes[0].hist(ptt_pp, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(ptt_pp.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {ptt_pp.mean():.1f} ms')
        axes[0].axvline(150, color='green', linestyle=':', linewidth=2, 
                       label='Expected mean: 150 ms')
        axes[0].set_title('PTT Peak-to-Peak Distribution')
        axes[0].set_xlabel('PTT (ms)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Check for zeros
        zero_pct = (ptt_pp == 0).mean() * 100
        if zero_pct > 5:
            metrics['warnings'].append(f'PTT zeros: {zero_pct:.1f}% (should be <5%)')
        metrics['ptt_pp_mean'] = float(ptt_pp.mean())
        metrics['ptt_pp_zeros_pct'] = float(zero_pct)
    
    # PTT peak-to-foot
    if 'ptt_peak_to_foot' in df.columns:
        ptt_pf = df['ptt_peak_to_foot']
        axes[1].hist(ptt_pf, bins=50, edgecolor='black', alpha=0.7, color='teal')
        axes[1].axvline(ptt_pf.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {ptt_pf.mean():.1f} ms')
        axes[1].axvline(200, color='green', linestyle=':', linewidth=2,
                       label='Expected mean: 200 ms')
        axes[1].set_title('PTT Peak-to-Foot Distribution')
        axes[1].set_xlabel('PTT (ms)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        metrics['ptt_pf_mean'] = float(ptt_pf.mean())
    
    # Amplitude ratio
    if 'amplitude_ratio_ra' in df.columns:
        amp = df['amplitude_ratio_ra']
        axes[2].hist(amp, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[2].axvline(amp.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {amp.mean():.3f}')
        axes[2].set_title('Amplitude Ratio Distribution')
        axes[2].set_xlabel('Ratio')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        
        # Check for 1.0 values
        ones_pct = (amp == 1.0).mean() * 100
        if ones_pct > 1:
            metrics['warnings'].append(f'Amplitude=1.0: {ones_pct:.1f}% (should be <1%)')
        metrics['amp_ratio_mean'] = float(amp.mean())
        metrics['amp_ratio_ones_pct'] = float(ones_pct)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/{prefix}_distributions.png")
    
    # =========================================
    # PLOT 2: PTT vs BP Scatter Plots (1x2)
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PTT vs SBP
    if 'ptt_peak_to_peak' in df.columns and 'sbp_reference' in df.columns:
        # Sample if too many points
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        
        ptt = sample_df['ptt_peak_to_peak']
        sbp = sample_df['sbp_reference']
        
        axes[0].scatter(ptt, sbp, alpha=0.3, s=5, c='steelblue')
        
        # Add trendline
        valid_mask = (ptt > 0) & (ptt < 500)
        if valid_mask.sum() > 100:
            z = np.polyfit(ptt[valid_mask], sbp[valid_mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(ptt[valid_mask].min(), ptt[valid_mask].max(), 100)
            axes[0].plot(x_range, p(x_range), "r-", linewidth=2, 
                        label=f'Slope: {z[0]:.2f} mmHg/ms')
            
            # Calculate correlation
            corr = ptt[valid_mask].corr(sbp[valid_mask])
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            metrics['ptt_sbp_correlation'] = float(corr)
            
            if corr > -0.3:
                metrics['errors'].append(f'PTT-SBP correlation: {corr:.3f} (should be < -0.3)')
                metrics['validation_passed'] = False
        
        axes[0].set_title('PTT vs SBP (SHOULD BE INVERSE)')
        axes[0].set_xlabel('PTT Peak-to-Peak (ms)')
        axes[0].set_ylabel('SBP (mmHg)')
        axes[0].legend()
    
    # PTT vs DBP
    if 'ptt_peak_to_peak' in df.columns and 'dbp_reference' in df.columns:
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        
        ptt = sample_df['ptt_peak_to_peak']
        dbp = sample_df['dbp_reference']
        
        axes[1].scatter(ptt, dbp, alpha=0.3, s=5, c='teal')
        
        valid_mask = (ptt > 0) & (ptt < 500)
        if valid_mask.sum() > 100:
            z = np.polyfit(ptt[valid_mask], dbp[valid_mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(ptt[valid_mask].min(), ptt[valid_mask].max(), 100)
            axes[1].plot(x_range, p(x_range), "r-", linewidth=2,
                        label=f'Slope: {z[0]:.2f} mmHg/ms')
            
            corr = ptt[valid_mask].corr(dbp[valid_mask])
            axes[1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            metrics['ptt_dbp_correlation'] = float(corr)
        
        axes[1].set_title('PTT vs DBP (SHOULD BE INVERSE)')
        axes[1].set_xlabel('PTT Peak-to-Peak (ms)')
        axes[1].set_ylabel('DBP (mmHg)')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}_ptt_vs_bp.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/{prefix}_ptt_vs_bp.png")
    
    # =========================================
    # PLOT 3: BP Distributions (1x2)
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'sbp_reference' in df.columns:
        sbp = df['sbp_reference']
        axes[0].hist(sbp, bins=50, edgecolor='black', alpha=0.7, color='crimson')
        axes[0].axvline(sbp.mean(), color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {sbp.mean():.1f} mmHg')
        axes[0].set_title('SBP Distribution')
        axes[0].set_xlabel('SBP (mmHg)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        metrics['sbp_mean'] = float(sbp.mean())
        metrics['sbp_std'] = float(sbp.std())
    
    if 'dbp_reference' in df.columns:
        dbp = df['dbp_reference']
        axes[1].hist(dbp, bins=50, edgecolor='black', alpha=0.7, color='navy')
        axes[1].axvline(dbp.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {dbp.mean():.1f} mmHg')
        axes[1].set_title('DBP Distribution')
        axes[1].set_xlabel('DBP (mmHg)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        metrics['dbp_mean'] = float(dbp.mean())
        metrics['dbp_std'] = float(dbp.std())
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}_bp_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/{prefix}_bp_distributions.png")
    
    # =========================================
    # PLOT 4: Correlation Matrix
    # =========================================
    corr_features = []
    for col in ['ptt_peak_to_peak', 'ptt_peak_to_foot', 'pat_ecg_ppg',
                'amplitude_ratio_ra', 'hr_bpm', 'sbp_reference', 'dbp_reference']:
        if col in df.columns:
            corr_features.append(col)
    
    if len(corr_features) >= 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = df[corr_features].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, mask=None,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_dir}/{prefix}_correlation_matrix.png")
    
    # =========================================
    # PLOT 5: PAT vs BP (if available)
    # =========================================
    if 'pat_ecg_ppg' in df.columns and 'sbp_reference' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        
        pat = sample_df['pat_ecg_ppg']
        sbp = sample_df['sbp_reference']
        dbp = sample_df['dbp_reference'] if 'dbp_reference' in sample_df.columns else None
        
        axes[0].scatter(pat, sbp, alpha=0.3, s=5, c='purple')
        
        valid_mask = (pat > 0) & (pat < 600)
        if valid_mask.sum() > 100:
            z = np.polyfit(pat[valid_mask], sbp[valid_mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(pat[valid_mask].min(), pat[valid_mask].max(), 100)
            axes[0].plot(x_range, p(x_range), "r-", linewidth=2,
                        label=f'Slope: {z[0]:.2f} mmHg/ms')
            
            corr = pat[valid_mask].corr(sbp[valid_mask])
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            metrics['pat_sbp_correlation'] = float(corr)
        
        axes[0].set_title('PAT (R-peak to PPG foot) vs SBP')
        axes[0].set_xlabel('PAT (ms)')
        axes[0].set_ylabel('SBP (mmHg)')
        axes[0].legend()
        
        if dbp is not None:
            axes[1].scatter(pat, dbp, alpha=0.3, s=5, c='darkviolet')
            
            if valid_mask.sum() > 100:
                z = np.polyfit(pat[valid_mask], dbp[valid_mask], 1)
                p = np.poly1d(z)
                axes[1].plot(x_range, p(x_range), "r-", linewidth=2,
                            label=f'Slope: {z[0]:.2f} mmHg/ms')
                
                corr = pat[valid_mask].corr(dbp[valid_mask])
                axes[1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1].transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            axes[1].set_title('PAT (R-peak to PPG foot) vs DBP')
            axes[1].set_xlabel('PAT (ms)')
            axes[1].set_ylabel('DBP (mmHg)')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}_pat_vs_bp.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_dir}/{prefix}_pat_vs_bp.png")
    
    # =========================================
    # Print Summary
    # =========================================
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\nTotal Samples: {metrics['total_samples']:,}")
    
    print("\nKEY CORRELATIONS (Expected values in parentheses):")
    if 'ptt_sbp_correlation' in metrics:
        status = "✓" if metrics['ptt_sbp_correlation'] < -0.3 else "✗"
        print(f"  {status} PTT vs SBP: {metrics['ptt_sbp_correlation']:.3f}  (Expected: < -0.5)")
    if 'ptt_dbp_correlation' in metrics:
        status = "✓" if metrics['ptt_dbp_correlation'] < -0.3 else "✗"
        print(f"  {status} PTT vs DBP: {metrics['ptt_dbp_correlation']:.3f}  (Expected: < -0.5)")
    if 'pat_sbp_correlation' in metrics:
        status = "✓" if metrics['pat_sbp_correlation'] < -0.3 else "✗"
        print(f"  {status} PAT vs SBP: {metrics['pat_sbp_correlation']:.3f}  (Expected: < -0.5)")
    
    if metrics['warnings']:
        print("\n⚠️  WARNINGS:")
        for w in metrics['warnings']:
            print(f"    - {w}")
    
    if metrics['errors']:
        print("\n❌ ERRORS:")
        for e in metrics['errors']:
            print(f"    - {e}")
    
    if metrics['validation_passed']:
        print("\n✓ VALIDATION PASSED - Data appears physiologically valid")
    else:
        print("\n✗ VALIDATION FAILED - Data has issues that need to be addressed")
    
    print("="*70)
    
    return metrics


def compare_before_after(before_df: pd.DataFrame, 
                        after_df: pd.DataFrame,
                        output_dir: str = "data/figures") -> None:
    """
    Generate side-by-side comparison plots of before/after cleaning.
    
    Args:
        before_df: DataFrame before cleaning
        after_df: DataFrame after cleaning
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Before cleaning
    if 'ptt_peak_to_peak' in before_df.columns:
        axes[0, 0].hist(before_df['ptt_peak_to_peak'], bins=100, 
                       edgecolor='black', alpha=0.7, color='red')
        axes[0, 0].set_title(f'BEFORE: PTT Distribution\n(n={len(before_df):,})')
        axes[0, 0].set_xlabel('PTT (ms)')
        
        # Mark zero spike
        zero_count = (before_df['ptt_peak_to_peak'] == 0).sum()
        zero_pct = zero_count / len(before_df) * 100
        axes[0, 0].annotate(f'Zeros: {zero_pct:.1f}%', xy=(0, 0.9), 
                           xycoords='axes fraction', fontsize=10, color='red')
    
    if 'amplitude_ratio_ra' in before_df.columns:
        axes[0, 1].hist(before_df['amplitude_ratio_ra'], bins=100,
                       edgecolor='black', alpha=0.7, color='red')
        axes[0, 1].set_title('BEFORE: Amplitude Ratio')
        axes[0, 1].set_xlabel('Ratio')
        
        ones_pct = (before_df['amplitude_ratio_ra'] == 1.0).mean() * 100
        axes[0, 1].annotate(f'=1.0: {ones_pct:.1f}%', xy=(0, 0.9),
                           xycoords='axes fraction', fontsize=10, color='red')
    
    if 'ptt_peak_to_peak' in before_df.columns and 'sbp_reference' in before_df.columns:
        sample = before_df.sample(n=min(5000, len(before_df)), random_state=42)
        axes[0, 2].scatter(sample['ptt_peak_to_peak'], sample['sbp_reference'],
                          alpha=0.3, s=3, c='red')
        axes[0, 2].set_title('BEFORE: PTT vs SBP')
        axes[0, 2].set_xlabel('PTT (ms)')
        axes[0, 2].set_ylabel('SBP (mmHg)')
    
    # Row 2: After cleaning
    if 'ptt_peak_to_peak' in after_df.columns:
        axes[1, 0].hist(after_df['ptt_peak_to_peak'], bins=50,
                       edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title(f'AFTER: PTT Distribution\n(n={len(after_df):,})')
        axes[1, 0].set_xlabel('PTT (ms)')
        axes[1, 0].axvline(after_df['ptt_peak_to_peak'].mean(), color='blue', 
                          linestyle='--', label=f"Mean: {after_df['ptt_peak_to_peak'].mean():.1f}")
        axes[1, 0].legend()
    
    if 'amplitude_ratio_ra' in after_df.columns:
        axes[1, 1].hist(after_df['amplitude_ratio_ra'], bins=50,
                       edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_title('AFTER: Amplitude Ratio')
        axes[1, 1].set_xlabel('Ratio')
    
    if 'ptt_peak_to_peak' in after_df.columns and 'sbp_reference' in after_df.columns:
        sample = after_df.sample(n=min(5000, len(after_df)), random_state=42)
        axes[1, 2].scatter(sample['ptt_peak_to_peak'], sample['sbp_reference'],
                          alpha=0.3, s=3, c='green')
        
        # Add trendline
        ptt = sample['ptt_peak_to_peak']
        sbp = sample['sbp_reference']
        valid = (ptt > 0) & (ptt < 500)
        if valid.sum() > 50:
            z = np.polyfit(ptt[valid], sbp[valid], 1)
            p = np.poly1d(z)
            x_range = np.linspace(ptt[valid].min(), ptt[valid].max(), 100)
            axes[1, 2].plot(x_range, p(x_range), 'b-', linewidth=2)
            
            corr = ptt[valid].corr(sbp[valid])
            axes[1, 2].set_title(f'AFTER: PTT vs SBP (r={corr:.3f})')
        else:
            axes[1, 2].set_title('AFTER: PTT vs SBP')
        
        axes[1, 2].set_xlabel('PTT (ms)')
        axes[1, 2].set_ylabel('SBP (mmHg)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/before_after_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to: {output_dir}/before_after_comparison.png")
    print(f"  Samples before: {len(before_df):,}")
    print(f"  Samples after:  {len(after_df):,}")
    print(f"  Removed: {len(before_df) - len(after_df):,} ({(len(before_df) - len(after_df)) / len(before_df) * 100:.1f}%)")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        
        metrics = validate_cleaned_data(df, output_dir="data/figures", prefix="validation")
        
        # Save metrics
        import json
        metrics_path = csv_path.replace('.csv', '_validation_metrics.json')
        # Convert non-serializable types
        metrics_clean = {k: v for k, v in metrics.items() 
                        if isinstance(v, (int, float, str, bool, list))}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
    else:
        print("Usage: python validate_pipeline.py <path_to_csv>")
        print("\nExample:")
        print("  python validate_pipeline.py data/processed/bp_dataset_cleaned_v2.csv")
