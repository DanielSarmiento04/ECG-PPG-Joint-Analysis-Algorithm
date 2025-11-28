import numpy as np

# Load saved data
data = np.load('./data/raw/case_00050.npz', allow_pickle=True)

print("Checking saved waveform shapes and content...\n")

# Check waveforms
for key in ['wave_SNUADC_ECG_II', 'wave_SNUADC_PLETH', 'wave_SNUADC_ART']:
    if key in data.files:
        arr = data[key]
        print(f"{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dimensions: {arr.ndim}")
        
        # Find first non-NaN
        non_nan = np.where(~np.isnan(arr))[0]
        if len(non_nan) > 0:
            first_idx = non_nan[0]
            last_idx = non_nan[-1]
            print(f"  First valid data at index: {first_idx} ({first_idx/500:.1f}s)")
            print(f"  Last valid data at index: {last_idx} ({last_idx/500:.1f}s)")
            print(f"  Valid data samples: {len(non_nan)}")
            print(f"  First 10 valid values: {arr[first_idx:first_idx+10]}")
            print(f"  Valid range: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
        else:
            print(f"  ❌ ALL NaN!")
        print()

print("\nChecking if this is a 2D array issue...")
print(f"ECG array ndim: {data['wave_SNUADC_ECG_II'].ndim}")
print(f"ECG array shape: {data['wave_SNUADC_ECG_II'].shape}")
