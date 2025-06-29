import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from pipeline.data_processor import DataProcessor
from pipeline.model_trainer import ModelTrainer
from pipeline.plot_generator import PlotGenerator
from pipeline.file_watcher import FileWatcher
from pipeline.utils import ConfigLoader, FileHandler, TimeUtils

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging with file and console handlers.
    
    Args:
        config: Application configuration dictionary
        
    Raises:
        PermissionError: If log file cannot be written
    """
    try:
        log_file = Path(config["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing log file if it exceeds max size
        if log_file.exists() and log_file.stat().st_size > config.get("max_log_size_mb", 10) * 1024 * 1024:
            log_file.unlink()
            
        logging.basicConfig(
            level=config.get("log_level", logging.INFO),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        
        # Capture uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception
        
    except Exception as e:
        print(f"Failed to configure logging: {str(e)}", file=sys.stderr)
        raise

def initialize_components(config: Dict[str, Any]) -> tuple:
    """Initialize and validate all pipeline components.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        Tuple of (processor, trainer, plotter) components
        
    Raises:
        RuntimeError: If components fail to initialize
    """
    try:
        processor = DataProcessor(config)
        trainer = ModelTrainer(config)
        plotter = PlotGenerator(config)
        
        # Validate directories
        required_dirs = ["input_directory", "output_directory", "img_directory"]
        for dir_key in required_dirs:
            Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
            
        return processor, trainer, plotter
        
    except Exception as e:
        logging.error("Component initialization failed", exc_info=True)
        raise RuntimeError(f"Failed to initialize components: {str(e)}")

def train_or_load_model(config: Dict[str, Any], processor: DataProcessor, 
                      trainer: ModelTrainer) -> Any:
    """Train new model or load existing one.
    
    Args:
        config: Application configuration
        processor: Data processor instance
        trainer: Model trainer instance
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If model training fails
    """
    model_path = Path(config["model_path"])
    
    if not model_path.exists() or config.get("force_retrain", False):
        try:
            logging.info("Starting model training...")
            train_file = config.get("train_data_path", "data/train_data.csv")
            
            if not Path(train_file).exists():
                raise FileNotFoundError(f"Training data not found at {train_file}")
                
            train_data = processor.load_data(train_file)
            train_data = processor.transform_data(train_data)
            
            # Log data statistics before training
            logging.info(f"Training data shape: {train_data.shape}")
            logging.info(f"Columns: {list(train_data.columns)}")
            
            model = trainer.train_model(train_data)
            logging.info(f"Model saved to {model_path}")
            return model
            
        except Exception as e:
            logging.error("Model training failed", exc_info=True)
            raise ValueError(f"Model training failed: {str(e)}")
    else:
        try:
            logging.info(f"Loading existing model from {model_path}")
            return trainer.load_model()
        except Exception as e:
            logging.error("Model loading failed", exc_info=True)
            raise ValueError(f"Model loading failed: {str(e)}")

def main() -> int:
    """Main application entry point.
    
    Returns:
        int: Exit code (0 for success)
    """
    try:
        # Load and validate configuration
        config = ConfigLoader.load_config()
        
        # Initialize logging
        setup_logging(config)
        logging.info("Starting application")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Initialize components
        processor, trainer, plotter = initialize_components(config)
        
        # Train or load model
        model = train_or_load_model(config, processor, trainer)
        
        # Start file watcher
        watcher = FileWatcher(config, processor, model, plotter)
        
        try:
            logging.info("Starting file watcher")
            watcher.start()
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down")
        finally:
            logging.info("Application stopped")
            
        return 0
        
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    start_time = datetime.now()
    exit_code = main()
    duration = datetime.now() - start_time
    print(f"Application completed in {duration.total_seconds():.2f} seconds")
    sys.exit(exit_code)