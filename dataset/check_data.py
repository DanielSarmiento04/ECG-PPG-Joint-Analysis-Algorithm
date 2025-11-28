import numpy as np
import sys

# Load case 50
print("Loading case 50...")
data = np.load('./data/raw/case_00050.npz', allow_pickle=True)

print(f"\nAvailable keys: {data.files}\n")

# Check each waveform
for key in ['wave_SNUADC_ECG_II', 'wave_SNUADC_PLETH', 'wave_SNUADC_ART']:
    if key in data.files:
        signal = data[key]
        signal_name = key.replace('wave_SNUADC_', '')
        
        print(f"{signal_name}:")
        print(f"  Shape: {signal.shape}")
        print(f"  Data type: {signal.dtype}")
        print(f"  Range: [{signal.min():.3f}, {signal.max():.3f}]")
        print(f"  Mean: {signal.mean():.3f}")
        print(f"  Std: {signal.std():.3f}")
        print(f"  Has NaN: {np.isnan(signal).any()}")
        print(f"  % NaN: {np.isnan(signal).sum() / len(signal) * 100:.2f}%")
        print(f"  First 5 values: {signal[:5]}")
        print()

# Check numeric data
print("\nNumeric data:")
for key in data.files:
    if key.startswith('num_'):
        signal = data[key]
        print(f"  {key}: shape={signal.shape}, mean={np.nanmean(signal):.1f}")
