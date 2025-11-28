from bp_pipeline import BPEstimationPipeline
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    config_path = 'pipeline_config.yaml'
    raw_data_dir = './data/raw'
    
    if not os.path.exists(raw_data_dir):
        logger.error(f"Data directory {raw_data_dir} not found!")
        return

    logger.info("Initializing BP Estimation Pipeline...")
    pipeline = BPEstimationPipeline(config_path)
    
    logger.info("Starting batch processing of all files...")
    # Set max_files=None to process all files
    df = pipeline.run_batch(raw_data_dir, max_files=None)
    
    if not df.empty:
        logger.info("Batch processing complete.")
        logger.info(f"Total records extracted: {len(df)}")
        logger.info(f"Unique patients: {df['patient_id'].nunique()}")
        
        print("\nDataset Summary:")
        print(df.describe())
    else:
        logger.warning("No features extracted. Check logs for errors.")

if __name__ == "__main__":
    main()
