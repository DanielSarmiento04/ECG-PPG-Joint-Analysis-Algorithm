"""Quick test of VitalDB API fixes"""
import vitaldb
import pandas as pd

print("Testing VitalDB 1.5.8 API...")

# Test 1: Get cases list
print("\n1. Testing cases download...")
try:
    url = 'https://api.vitaldb.net/cases'
    df_cases = pd.read_csv(url)
    print(f"✓ Successfully loaded {len(df_cases)} cases")
    print(f"  Columns: {df_cases.columns.tolist()[:5]}...")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Get tracks for a case
print("\n2. Testing vital_trks()...")
try:
    caseid = 1
    trks_df = vitaldb.vital_trks(caseid)
    if trks_df is not None and len(trks_df) > 0:
        print(f"✓ Case {caseid} has {len(trks_df)} tracks")
        print(f"  Sample tracks: {trks_df['tname'].tolist()[:5]}")
    else:
        print(f"  Case {caseid} has no tracks")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Load a single track
print("\n3. Testing load_case()...")
try:
    caseid = 1
    track = 'SNUADC/ECG_II'
    data = vitaldb.load_case(caseid, track)
    if data is not None:
        print(f"✓ Loaded {track}: {len(data)} samples")
    else:
        print(f"  Track {track} not found in case {caseid}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All API tests completed!")
