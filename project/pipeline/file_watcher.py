import time
import logging
from pathlib import Path
from typing import Dict, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

class PipelineHandler(FileSystemEventHandler):
    def __init__(self, config: Dict, processor, model, plotter):
        self.config = config
        self.processor = processor
        self.model = model
        self.plotter = plotter
        self.logger = logging.getLogger(__name__)
        self._processing_lock = False  # Prevents concurrent processing
        self._processed_files = set()  # Tracks processed files to avoid duplicates

    def on_created(self, event):
        """Handle new file creation events"""
        if (not event.is_directory and 
            event.src_path.endswith('.csv') and 
            not self._processing_lock and
            event.src_path not in self._processed_files):
            
            try:
                self._processing_lock = True
                self.process_file(event.src_path)
                self._processed_files.add(event.src_path)
            except Exception as e:
                self.logger.error(f"Failed to process {event.src_path}: {str(e)}")
            finally:
                self._processing_lock = False

    def process_file(self, file_path: str) -> Optional[Path]:
        """Process a single data file through the pipeline"""
        try:
            file_path = Path(file_path)
            self.logger.info(f"Processing new file: {file_path.name}")
            
            # Validate file
            if not self._validate_file(file_path):
                return None

            # 1. Load and transform data
            start_time = time.time()
            df = self.processor.load_data(str(file_path))
            df = self.processor.transform_data(df)
            load_time = time.time() - start_time
            
            # 2. Make predictions
            start_time = time.time()
            features = df.select_dtypes(include=['float64'])
            df['anomaly'] = self.model.predict(features)
            df['anomaly_score'] = self.model.decision_function(features)  # Add confidence scores
            predict_time = time.time() - start_time
            
            # 3. Save results
            output_file = self._generate_output_path(file_path)
            self.processor.save_processed_data(df, str(output_file))
            
            # 4. Generate visualizations
            plot_paths = []
            for sensor in self.config["sensors_to_plot"]:
                if sensor in df.columns:
                    fig = self.plotter.plot_sensor_anomalies(df, sensor)
                    plot_name = self._generate_plot_name(file_path, sensor)
                    plot_path = self.plotter.save_plot(fig, plot_name)
                    plot_paths.append(plot_path)
            
            # 5. Clean up
            file_path.unlink()
            
            # Log performance metrics
            self._log_processing_metrics(
                file_path.name,
                len(df),
                load_time,
                predict_time,
                output_file,
                plot_paths
            )
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
            raise

    def _validate_file(self, file_path: Path) -> bool:
        """Validate input file before processing"""
        if not file_path.exists():
            self.logger.warning(f"File disappeared before processing: {file_path}")
            return False
            
        if file_path.stat().st_size == 0:
            self.logger.error(f"Empty file detected: {file_path}")
            return False
            
        return True

    def _generate_output_path(self, input_path: Path) -> Path:
        """Generate output path with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_path.stem}_processed_{timestamp}.csv"

    def _generate_plot_name(self, input_path: Path, sensor_name: str) -> str:
        """Generate standardized plot filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{input_path.stem}_{sensor_name}_{timestamp}.png"

    def _log_processing_metrics(self, filename: str, num_records: int,
                              load_time: float, predict_time: float,
                              output_path: Path, plot_paths: list):
        """Log detailed processing metrics"""
        metrics = {
            "input_file": filename,
            "records_processed": num_records,
            "load_time_sec": round(load_time, 3),
            "predict_time_sec": round(predict_time, 3),
            "output_file": output_path.name,
            "plots_generated": [p.name for p in plot_paths],
            "total_time_sec": round(load_time + predict_time, 3)
        }
        self.logger.info("Processing metrics: %s", metrics)

class FileWatcher:
    def __init__(self, config: Dict, processor, model, plotter):
        self.config = config
        self.observer = Observer()
        self.handler = PipelineHandler(config, processor, model, plotter)
        self.logger = logging.getLogger(__name__)
        self._running = False

    def start(self) -> None:
        """Start watching the input directory"""
        input_dir = Path(self.config["input_directory"])
        input_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.observer.schedule(
                self.handler,
                path=str(input_dir),
                recursive=self.config.get("recursive_watch", False)
            )
            self.observer.start()
            self._running = True
            self.logger.info(f"Started watching directory: {input_dir}")
            
            while self._running:
                time.sleep(self.config["check_interval"])
                
        except Exception as e:
            self.logger.error(f"File watcher failed: {str(e)}", exc_info=True)
            raise
            
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop watching the input directory"""
        if self._running:
            self.observer.stop()
            self.observer.join()
            self._running = False
            self.logger.info("File watcher stopped")

    def __enter__(self):
        """Support context manager protocol"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup on exit"""
        self.stop()