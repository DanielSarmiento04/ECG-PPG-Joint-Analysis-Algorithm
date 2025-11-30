from bp_pipeline import BPEstimationPipeline
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    config_path = 'pipeline_config.yaml'
    raw_data_dir = './data/raw'
    
    
    parser = argparse.ArgumentParser(description='Run full batch processing of BP estimation pipeline.')

    # data paremeters
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process. Default is None (process all files).")
    
    args = parser.parse_args()
    
    if not os.path.exists(raw_data_dir):
        logger.error(f"Data directory {raw_data_dir} not found!")
        return

    logger.info("Initializing BP Estimation Pipeline...")
    pipeline = BPEstimationPipeline(config_path)
    
    logger.info("Starting batch processing of all files...")
    # Set max_files to the value provided by the user
    df = pipeline.run_batch(raw_data_dir, max_files=args.max_files)
    
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
