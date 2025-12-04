print("Starting analysis script...")
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from scipy.signal import find_peaks, butter, filtfilt, correlate
from scipy.ndimage import gaussian_filter1d


def plot_ecg_ppg_alignment(case_file, output_dir, fs=500):
    """
    Plot ECG-PPG alignment to verify signal synchronization.
    Shows 10 seconds of raw data and zoomed view of 2 cardiac cycles.
    """
    try:
        case = np.load(case_file, allow_pickle=True)
        ppg = case['wave_SNUADC_PLETH']
        ecg = case['wave_SNUADC_ECG_II']
    except Exception as e:
        print(f"Could not load {case_file}: {e}")
        return
    
    # Handle NaN
    ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg))
    ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg))
    
    # Use 30 seconds of data
    duration = min(30 * fs, len(ecg))
    ecg = ecg[:duration]
    ppg = ppg[:duration]
    
    # Find R-peaks
    nyq = fs / 2
    b, a = butter(2, [5/nyq, 15/nyq], btype='band')
    ecg_filt = filtfilt(b, a, ecg)
    ecg_sq = ecg_filt ** 2
    threshold = np.percentile(ecg_sq, 95) * 0.3
    r_peaks, _ = find_peaks(ecg_sq, distance=int(0.5*fs), height=threshold)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: 10 seconds overview
    t = np.arange(10*fs) / fs
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ecg_10s = ecg[:10*fs]
    ppg_10s = ppg[:10*fs]
    
    # Normalize for visualization
    ecg_norm = (ecg_10s - np.mean(ecg_10s)) / np.std(ecg_10s)
    ppg_norm = (ppg_10s - np.mean(ppg_10s)) / np.std(ppg_10s)
    
    ax1.plot(t, ecg_norm, 'b-', alpha=0.7, label='ECG')
    ax1_twin.plot(t, ppg_norm, 'r-', alpha=0.7, label='PPG')
    
    # Mark R-peaks
    rpk_10s = r_peaks[r_peaks < 10*fs]
    ax1.scatter(rpk_10s/fs, ecg_norm[rpk_10s], c='blue', s=50, zorder=5, marker='v')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('ECG (normalized)', color='blue')
    ax1_twin.set_ylabel('PPG (normalized)', color='red')
    ax1.set_title('ECG-PPG Alignment: 10 Second Overview')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Zoomed view (2 cardiac cycles)
    if len(r_peaks) >= 3:
        start_idx = r_peaks[0]
        end_idx = r_peaks[2] + int(0.2*fs)
        
        t_zoom = np.arange(end_idx - start_idx) / fs * 1000  # ms
        ecg_zoom = ecg[start_idx:end_idx]
        ppg_zoom = ppg[start_idx:end_idx]
        
        ecg_zoom_norm = (ecg_zoom - np.mean(ecg_zoom)) / np.std(ecg_zoom)
        ppg_zoom_norm = (ppg_zoom - np.mean(ppg_zoom)) / np.std(ppg_zoom)
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(t_zoom, ecg_zoom_norm, 'b-', linewidth=2, label='ECG')
        ax2_twin.plot(t_zoom, ppg_zoom_norm, 'r-', linewidth=2, label='PPG')
        
        # Mark R-peaks in zoomed view
        for rpk in r_peaks[:3]:
            if start_idx <= rpk < end_idx:
                ax2.axvline(x=(rpk-start_idx)/fs*1000, color='blue', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('ECG (normalized)', color='blue')
        ax2_twin.set_ylabel('PPG (normalized)', color='red')
        ax2.set_title('Zoomed View: 2 Cardiac Cycles (R-peak aligned)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
    
    # Plot 3: Cross-correlation to find optimal lag
    ax3 = axes[2]
    
    # Use 10 seconds for cross-correlation
    ecg_corr = (ecg[:10*fs] - np.mean(ecg[:10*fs])) / np.std(ecg[:10*fs])
    ppg_corr = (ppg[:10*fs] - np.mean(ppg[:10*fs])) / np.std(ppg[:10*fs])
    
    xcorr = correlate(ppg_corr, ecg_corr, mode='full')
    lags = np.arange(-len(ecg_corr)+1, len(ecg_corr)) / fs * 1000  # ms
    
    # Focus on physiological range (-500 to 500ms)
    mask = (lags >= -500) & (lags <= 500)
    
    ax3.plot(lags[mask], xcorr[mask])
    
    # Find best lag
    best_idx = np.argmax(xcorr[mask])
    best_lag = lags[mask][best_idx]
    ax3.axvline(x=best_lag, color='red', linestyle='--', label=f'Best lag: {best_lag:.0f}ms')
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Lag (ms) - PPG relative to ECG')
    ax3.set_ylabel('Cross-correlation')
    ax3.set_title(f'ECG-PPG Cross-correlation (Best lag: {best_lag:.0f}ms)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ecg_ppg_alignment.png'), dpi=150)
    plt.close()
    print(f"Saved ECG-PPG alignment plot.")
    
    return best_lag


def plot_ppg_beat_morphology(case_file, output_dir, fs=500):
    """
    Plot PPG beat morphology showing foot detection points.
    Useful for validating PTT extraction.
    """
    try:
        from fixed_feature_extraction import FixedFeatureExtractor, detect_global_ppg_polarity
    except ImportError:
        print("Could not import fixed_feature_extraction, skipping beat morphology plot")
        return
    
    try:
        case = np.load(case_file, allow_pickle=True)
        ppg = case['wave_SNUADC_PLETH']
        ecg = case['wave_SNUADC_ECG_II']
    except Exception as e:
        print(f"Could not load {case_file}: {e}")
        return
    
    # Handle NaN
    ppg = np.nan_to_num(ppg, nan=np.nanmean(ppg))
    ecg = np.nan_to_num(ecg, nan=np.nanmean(ecg))
    
    # Find R-peaks
    nyq = fs / 2
    b, a = butter(2, [5/nyq, 15/nyq], btype='band')
    ecg_filt = filtfilt(b, a, ecg)
    ecg_sq = ecg_filt ** 2
    threshold = np.percentile(ecg_sq, 95) * 0.3
    r_peaks, _ = find_peaks(ecg_sq, distance=int(0.5*fs), height=threshold)
    
    if len(r_peaks) < 10:
        print("Not enough R-peaks for beat morphology plot")
        return
    
    # Initialize extractor
    extractor = FixedFeatureExtractor(fs=fs)
    extractor.set_global_polarity(ppg, r_peaks)
    
    # Extract beats and detect foot points
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot first 6 beats
    for i in range(min(6, len(r_peaks)-1)):
        ax = axes[i//3, i%3]
        
        rpk = r_peaks[i]
        next_rpk = r_peaks[i+1]
        ppg_segment = ppg[rpk:next_rpk]
        
        if len(ppg_segment) < 100:
            continue
        
        # Get foot detection
        foot_idx, foot_time = extractor.find_ppg_foot(ppg_segment)
        peak_idx, peak_val = extractor.find_systolic_peak(ppg_segment)
        
        t = np.arange(len(ppg_segment)) / fs * 1000  # ms
        
        # Normalize for visualization
        ppg_norm = (ppg_segment - np.min(ppg_segment)) / (np.max(ppg_segment) - np.min(ppg_segment))
        
        ax.plot(t, ppg_norm, 'b-', linewidth=1.5, label='PPG')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='R-peak')
        ax.axvline(x=foot_time, color='red', linestyle='-', linewidth=2, label=f'Foot: {foot_time:.0f}ms')
        ax.axvline(x=peak_idx/fs*1000, color='orange', linestyle='-', linewidth=2, label=f'Peak: {peak_idx/fs*1000:.0f}ms')
        
        ax.scatter([foot_time], [ppg_norm[foot_idx]], c='red', s=100, zorder=5)
        ax.scatter([peak_idx/fs*1000], [ppg_norm[peak_idx]], c='orange', s=100, zorder=5)
        
        ax.set_xlabel('Time from R-peak (ms)')
        ax.set_ylabel('PPG (normalized)')
        ax.set_title(f'Beat {i+1}: PTT = {foot_time:.0f}ms')
        ax.legend(fontsize=8)
        ax.set_xlim(-10, 600)
    
    polarity = "INVERTED" if extractor._global_ppg_inverted else "NORMAL"
    plt.suptitle(f'PPG Beat Morphology with Foot Detection (Polarity: {polarity})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppg_beat_morphology.png'), dpi=150)
    plt.close()
    print("Saved PPG beat morphology plot.")


def plot_ptt_distribution_analysis(df, output_dir):
    """
    Detailed PTT distribution analysis showing boundary clustering issues.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: PTT histogram with boundary markers
    ax1 = axes[0, 0]
    ptt_col = 'ptt_peak_to_foot' if 'ptt_peak_to_foot' in df.columns else 'pat_ecg_ppg'
    
    if ptt_col in df.columns:
        ptt_data = df[ptt_col].dropna()
        ax1.hist(ptt_data, bins=50, edgecolor='black', alpha=0.7)
        
        # Mark common boundaries
        for boundary in [40, 80, 100, 200, 250]:
            count_at_boundary = ((ptt_data >= boundary-1) & (ptt_data <= boundary+1)).sum()
            pct = count_at_boundary / len(ptt_data) * 100
            if pct > 1:
                ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
                ax1.text(boundary, ax1.get_ylim()[1]*0.9, f'{boundary}ms\n({pct:.1f}%)', 
                        ha='center', fontsize=9, color='red')
        
        ax1.set_xlabel(f'{ptt_col} (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'PTT Distribution (n={len(ptt_data):,})')
        
        # Add statistics
        stats_text = f'Mean: {ptt_data.mean():.1f}ms\nStd: {ptt_data.std():.1f}ms\nMin: {ptt_data.min():.1f}ms\nMax: {ptt_data.max():.1f}ms'
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Per-patient PTT variability
    ax2 = axes[0, 1]
    if ptt_col in df.columns and 'patient_id' in df.columns:
        patient_stats = df.groupby('patient_id')[ptt_col].agg(['mean', 'std', 'count']).reset_index()
        patient_stats = patient_stats[patient_stats['count'] >= 100]  # At least 100 samples
        
        ax2.errorbar(range(len(patient_stats)), patient_stats['mean'], 
                    yerr=patient_stats['std'], fmt='o', capsize=3, alpha=0.7)
        ax2.axhline(y=150, color='green', linestyle='--', label='Typical PTT (150ms)')
        ax2.set_xlabel('Patient Index')
        ax2.set_ylabel(f'{ptt_col} (ms)')
        ax2.set_title(f'Per-Patient PTT (mean ± std), n={len(patient_stats)} patients')
        ax2.legend()
    
    # Plot 3: PTT vs SBP colored by patient
    ax3 = axes[1, 0]
    if ptt_col in df.columns and 'sbp_reference' in df.columns:
        # Sample for visualization
        df_sample = df.dropna(subset=[ptt_col, 'sbp_reference']).sample(min(5000, len(df)), random_state=42)
        
        scatter = ax3.scatter(df_sample[ptt_col], df_sample['sbp_reference'], 
                             c=df_sample['patient_id'].astype('category').cat.codes,
                             alpha=0.3, s=10, cmap='tab20')
        ax3.set_xlabel(f'{ptt_col} (ms)')
        ax3.set_ylabel('SBP (mmHg)')
        ax3.set_title('PTT vs SBP (colored by patient)')
        
        # Add correlation
        corr = df_sample[ptt_col].corr(df_sample['sbp_reference'])
        ax3.text(0.02, 0.98, f'r = {corr:.3f}', transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: PTT boundary percentage by patient
    ax4 = axes[1, 1]
    if ptt_col in df.columns and 'patient_id' in df.columns:
        boundary_pct = []
        for pid in df['patient_id'].unique():
            p_data = df[df['patient_id'] == pid][ptt_col].dropna()
            if len(p_data) >= 50:
                # Check for 80ms boundary (or 40ms for old data)
                at_80 = ((p_data >= 79) & (p_data <= 81)).sum() / len(p_data) * 100
                at_40 = ((p_data >= 39) & (p_data <= 41)).sum() / len(p_data) * 100
                boundary_pct.append({'patient_id': pid, 'at_80ms': at_80, 'at_40ms': at_40, 'n': len(p_data)})
        
        if boundary_pct:
            bp_df = pd.DataFrame(boundary_pct)
            bp_df = bp_df.sort_values('at_80ms', ascending=False)
            
            x = range(len(bp_df))
            ax4.bar(x, bp_df['at_80ms'], alpha=0.7, label='At 80ms boundary')
            ax4.bar(x, bp_df['at_40ms'], alpha=0.7, label='At 40ms boundary')
            ax4.axhline(y=10, color='red', linestyle='--', label='10% threshold')
            ax4.set_xlabel('Patient (sorted by boundary %)')
            ax4.set_ylabel('% samples at boundary')
            ax4.set_title('Boundary Clustering by Patient')
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ptt_distribution_analysis.png'), dpi=150)
    plt.close()
    print("Saved PTT distribution analysis plot.")


def plot_patient_physiology_validation(df, output_dir):
    """
    Plot patient-level physiological validation metrics.
    Shows which patients have correct PTT-BP relationships.
    """
    if 'patient_id' not in df.columns:
        print("No patient_id column, skipping physiology validation plot")
        return
    
    ptt_col = 'ptt_peak_to_foot' if 'ptt_peak_to_foot' in df.columns else 'pat_ecg_ppg'
    
    results = []
    for pid in df['patient_id'].unique():
        p_df = df[df['patient_id'] == pid]
        if len(p_df) < 100:
            continue
        
        ptt_sbp_r = p_df[ptt_col].corr(p_df['sbp_reference']) if ptt_col in p_df.columns else np.nan
        ptt_hr_r = p_df[ptt_col].corr(p_df['hr_bpm']) if ptt_col in p_df.columns and 'hr_bpm' in p_df.columns else np.nan
        ptt_std = p_df[ptt_col].std() if ptt_col in p_df.columns else np.nan
        ppg_std = p_df['stat_std'].mean() if 'stat_std' in p_df.columns else np.nan
        
        results.append({
            'patient_id': pid,
            'n_samples': len(p_df),
            'ptt_sbp_r': ptt_sbp_r,
            'ptt_hr_r': ptt_hr_r,
            'ptt_std': ptt_std,
            'ppg_std': ppg_std,
            'sbp_mean': p_df['sbp_reference'].mean(),
            'sbp_std': p_df['sbp_reference'].std()
        })
    
    if not results:
        print("No patients with enough data for validation plot")
        return
    
    res_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: PTT-SBP correlation vs PTT-HR correlation
    ax1 = axes[0, 0]
    colors = ['green' if (r['ptt_sbp_r'] < 0 and r['ptt_hr_r'] < 0) else 'red' for _, r in res_df.iterrows()]
    ax1.scatter(res_df['ptt_sbp_r'], res_df['ptt_hr_r'], c=colors, alpha=0.6, s=50)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PTT-SBP Correlation')
    ax1.set_ylabel('PTT-HR Correlation')
    ax1.set_title('Patient Physiology Validation\n(Green = correct signs, Red = wrong)')
    
    # Count good vs bad patients
    good = sum(1 for c in colors if c == 'green')
    bad = sum(1 for c in colors if c == 'red')
    ax1.text(0.02, 0.98, f'Good: {good}\nBad: {bad}', transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: PTT std vs PPG std (signal quality indicator)
    ax2 = axes[0, 1]
    if 'ppg_std' in res_df.columns and not res_df['ppg_std'].isna().all():
        ax2.scatter(res_df['ppg_std'], res_df['ptt_std'], c=colors, alpha=0.6, s=50)
        ax2.set_xlabel('PPG Signal Std (quality indicator)')
        ax2.set_ylabel('PTT Std (ms)')
        ax2.set_title('Signal Quality vs PTT Variability')
    else:
        ax2.scatter(res_df['ptt_std'], res_df['ptt_sbp_r'], c=colors, alpha=0.6, s=50)
        ax2.set_xlabel('PTT Std (ms)')
        ax2.set_ylabel('PTT-SBP Correlation')
        ax2.set_title('PTT Variability vs Correlation')
    
    # Plot 3: SBP range vs PTT-SBP correlation
    ax3 = axes[1, 0]
    ax3.scatter(res_df['sbp_std'], res_df['ptt_sbp_r'], c=colors, alpha=0.6, s=50)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('SBP Std (BP variability)')
    ax3.set_ylabel('PTT-SBP Correlation')
    ax3.set_title('BP Variability vs PTT-SBP Correlation')
    
    # Plot 4: Sample count vs correlation
    ax4 = axes[1, 1]
    ax4.scatter(res_df['n_samples'], res_df['ptt_sbp_r'], c=colors, alpha=0.6, s=50)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('PTT-SBP Correlation')
    ax4.set_title('Sample Count vs Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_physiology_validation.png'), dpi=150)
    plt.close()
    print("Saved patient physiology validation plot.")


def plot_feature_importance_for_bp(df, output_dir):
    """
    Plot feature correlations with BP to identify most predictive features.
    """
    # Define feature groups
    ptt_features = ['pat_ecg_ppg', 'pat_to_peak', 'pat_to_maxslope', 
                   'ptt_peak_to_foot', 'ptt_peak_to_peak', 'ptt_peak_to_maxslope']
    morphology_features = ['amplitude_ratio_ra', 'systolic_duration_tsd', 'diastolic_duration_tfd',
                          'reflection_index', 'systolic_area_ratio', 'pulse_width_50', 'crest_time']
    derivative_features = ['max_upslope', 'max_downslope', 'slope_ratio', 
                          'max_acceleration', 'min_acceleration']
    stat_features = ['stat_mean', 'stat_std', 'stat_skew', 'stat_kurtosis',
                    'stat_power_cardiac', 'stat_power_low', 'stat_power_ratio']
    
    all_features = ptt_features + morphology_features + derivative_features + stat_features
    available_features = [f for f in all_features if f in df.columns]
    
    if len(available_features) == 0:
        print("No features available for importance plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Calculate correlations with SBP and DBP
    sbp_corrs = []
    dbp_corrs = []
    
    for feat in available_features:
        if feat in df.columns and 'sbp_reference' in df.columns:
            sbp_corr = df[feat].corr(df['sbp_reference'])
            dbp_corr = df[feat].corr(df['dbp_reference'])
            sbp_corrs.append(sbp_corr)
            dbp_corrs.append(dbp_corr)
    
    # Sort by absolute SBP correlation
    sorted_idx = np.argsort(np.abs(sbp_corrs))[::-1]
    
    # Plot SBP correlations
    ax1 = axes[0]
    y_pos = range(len(available_features))
    colors = ['green' if c < 0 else 'red' for c in [sbp_corrs[i] for i in sorted_idx]]
    ax1.barh(y_pos, [sbp_corrs[i] for i in sorted_idx], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([available_features[i] for i in sorted_idx])
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Correlation with SBP')
    ax1.set_title('Feature-SBP Correlations\n(Green=negative, Red=positive)')
    
    # Plot DBP correlations
    ax2 = axes[1]
    colors = ['green' if c < 0 else 'red' for c in [dbp_corrs[i] for i in sorted_idx]]
    ax2.barh(y_pos, [dbp_corrs[i] for i in sorted_idx], color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([available_features[i] for i in sorted_idx])
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Correlation with DBP')
    ax2.set_title('Feature-DBP Correlations\n(Green=negative, Red=positive)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_bp_correlations.png'), dpi=150)
    plt.close()
    print("Saved feature-BP correlations plot.")


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
    if 'ptt_peak_to_peak' in df.columns:
        print(f"PTT (peak-to-peak): {df['ptt_peak_to_peak'].mean():.1f} ± {df['ptt_peak_to_peak'].std():.1f} ms")
    if 'ptt_peak_to_foot' in df.columns:
        print(f"PTT (peak-to-foot): {df['ptt_peak_to_foot'].mean():.1f} ± {df['ptt_peak_to_foot'].std():.1f} ms")
    
    # ============================================================
    # NEW DIAGNOSTIC PLOTS
    # ============================================================
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    # 7. PTT Distribution Analysis (boundary clustering)
    print("\n--- PTT Distribution Analysis ---")
    plot_ptt_distribution_analysis(df, output_dir)
    
    # 8. Patient Physiology Validation
    print("\n--- Patient Physiology Validation ---")
    plot_patient_physiology_validation(df, output_dir)
    
    # 9. Feature-BP Correlations
    print("\n--- Feature Importance for BP ---")
    plot_feature_importance_for_bp(df, output_dir)
    
    # 10. ECG-PPG Alignment (using first available raw file)
    print("\n--- ECG-PPG Alignment Analysis ---")
    raw_dir = './data/raw'
    if os.path.exists(raw_dir):
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.npz')]
        if raw_files:
            first_case = os.path.join(raw_dir, raw_files[0])
            plot_ecg_ppg_alignment(first_case, output_dir)
            
            # 11. PPG Beat Morphology
            print("\n--- PPG Beat Morphology ---")
            plot_ppg_beat_morphology(first_case, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":

    """Main training loop with advanced techniques."""
    parser = argparse.ArgumentParser(description='Feature Analysis for BP Estimation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/processed/bp_dataset_features.csv',
                       help='Path to the dataset file (CSV or Excel)')
    
    parser.add_argument('--output_dir', type=str, default='./data/figures',
                          help='Directory to save output figures')

    analyze_features(
        csv_path=parser.parse_args().data_path,
        output_dir=parser.parse_args().output_dir
    )
