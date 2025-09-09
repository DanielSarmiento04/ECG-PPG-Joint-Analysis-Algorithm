"""
Logging configuration for the blood pressure prediction pipeline.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, 
                 format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    """
    
    # Default format string
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('xgboost').setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    if log_file:
        logger.info(f"Log file: {log_file}")
