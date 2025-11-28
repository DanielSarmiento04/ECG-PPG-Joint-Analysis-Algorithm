from bp_pipeline import BPEstimationPipeline
import os
import pandas as pd

def test_single_file():
    pipeline = BPEstimationPipeline('pipeline_config.yaml')
    raw_data_dir = './data/raw'
    
    # Find first file
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.npz')]
    if not files:
        print("No .npz files found.")
        return

    test_file = os.path.join(raw_data_dir, files[0])
    print(f"Testing on {test_file}")
    
    df = pipeline.process_recording(test_file)
    
    print("Resulting DataFrame:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    if not df.empty:
        print("\nColumns:")
        print(df.columns.tolist())
        
        print("\nStatistics:")
        print(df.describe())

if __name__ == "__main__":
    test_single_file()
