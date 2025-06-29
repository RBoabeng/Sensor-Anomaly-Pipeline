import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import yaml  # Added for YAML config support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    missing_values: Dict[str, int]
    dtypes: Dict[str, str]
    numeric_stats: Optional[Dict[str, Dict[str, float]]]
    shape: Tuple[int, int]
    duplicate_rows: int
    memory_usage: str

class ConfigLoader:
    """Enhanced configuration loader with validation and multi-format support"""
    
    @staticmethod
    def load_config(config_path: str = "config/application.json") -> Dict[str, Any]:
        """
        Load configuration file with validation.
        Supports both JSON and YAML formats.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: For invalid configuration
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path) as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Validate required directories
            required_dirs = ["input_directory", "output_directory", "img_directory"]
            for dir_key in required_dirs:
                if dir_key not in config:
                    raise ValueError(f"Missing required config key: {dir_key}")
                Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
                
            # Set defaults for optional parameters
            config.setdefault("check_interval", 30)
            config.setdefault("max_file_age_days", 7)
            
            return config
            
        except Exception as e:
            logger.error(f"Config loading failed: {str(e)}", exc_info=True)
            raise

class DataValidator:
    """Enhanced data validation with comprehensive checks"""
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate DataFrame contains all required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if all columns are present
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
        return not missing
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataQualityReport: Structured quality report
        """
        numeric_cols = df.select_dtypes(include=np.number).columns
        return DataQualityReport(
            missing_values=df.isnull().sum().to_dict(),
            dtypes=df.dtypes.astype(str).to_dict(),
            numeric_stats=df[numeric_cols].describe().to_dict() if not numeric_cols.empty else None,
            shape=df.shape,
            duplicate_rows=df.duplicated().sum(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )
    
    @staticmethod
    def validate_timestamps(df: pd.DataFrame, time_col: str = "timestamp") -> bool:
        """
        Validate timestamp column is properly formatted and monotonic.
        
        Args:
            df: DataFrame containing timestamps
            time_col: Name of timestamp column
            
        Returns:
            bool: True if timestamps are valid
        """
        if time_col not in df.columns:
            return False
            
        # Check if already datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                return False
                
        # Check for monotonic increasing
        return df[time_col].is_monotonic_increasing

class FileHandler:
    """Enhanced file operations with safety checks"""
    
    @staticmethod
    def get_latest_file(directory: str, pattern: str = "*.csv", 
                       max_age_days: Optional[int] = None) -> Optional[Path]:
        """
        Get most recent file matching pattern with optional age filter.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            max_age_days: Maximum file age in days
            
        Returns:
            Path to latest file or None
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return None
                
            files = list(dir_path.glob(pattern))
            if not files:
                return None
                
            latest = max(files, key=lambda x: x.stat().st_mtime)
            
            if max_age_days:
                file_age = datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)
                if file_age > timedelta(days=max_age_days):
                    logger.info(f"Ignoring old file: {latest} (age: {file_age})")
                    return None
                    
            return latest
            
        except Exception as e:
            logger.warning(f"Error finding latest file: {str(e)}")
            return None
    
    @staticmethod
    def clean_directory(directory: str, pattern: str = "*", 
                       keep_recent: int = 3, dry_run: bool = False) -> List[Path]:
        """
        Clean directory while keeping N most recent files.
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            keep_recent: Number of recent files to keep
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of deleted files
        """
        deleted = []
        try:
            dir_path = Path(directory)
            files = sorted(dir_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
            
            for f in files[keep_recent:]:
                if dry_run:
                    logger.info(f"[Dry run] Would delete: {f}")
                else:
                    try:
                        f.unlink()
                        deleted.append(f)
                        logger.info(f"Deleted old file: {f}")
                    except Exception as e:
                        logger.error(f"Failed to delete {f}: {str(e)}")
                        
            return deleted
            
        except Exception as e:
            logger.error(f"Directory cleanup failed: {str(e)}")
            return []

class TimeUtils:
    """Enhanced time utilities with timezone support"""
    
    @staticmethod
    def timestamp_to_str(timestamp: datetime, 
                        fmt: str = "%Y-%m-%d %H:%M:%S",
                        timezone: Optional[str] = None) -> str:
        """
        Convert timestamp to string with optional timezone.
        
        Args:
            timestamp: datetime object
            fmt: Format string
            timezone: Optional timezone string (e.g., 'UTC')
            
        Returns:
            Formatted datetime string
        """
        if timezone:
            import pytz
            tz = pytz.timezone(timezone)
            timestamp = timestamp.astimezone(tz)
        return timestamp.strftime(fmt)
    
    @staticmethod
    def generate_run_id(prefix: str = "", suffix: str = "") -> str:
        """
        Generate unique run identifier with optional prefix/suffix.
        
        Args:
            prefix: Optional prefix
            suffix: Optional suffix
            
        Returns:
            Unique run ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}{timestamp}{suffix}".strip("_")

class AnomalyPostprocessor:
    """Enhanced anomaly post-processing"""
    
    @staticmethod
    def convert_anomaly_labels(predictions: np.ndarray, 
                              inplace: bool = False) -> np.ndarray:
        """
        Convert sklearn anomaly labels (-1/1) to binary (0/1).
        
        Args:
            predictions: Input array of predictions
            inplace: If True, modify array in-place
            
        Returns:
            Converted array
        """
        if not inplace:
            predictions = predictions.copy()
        return np.where(predictions == -1, 1, 0)
    
    @staticmethod
    def filter_confidence_scores(scores: np.ndarray, 
                                threshold: float = 0.9,
                                return_scores: bool = False) -> np.ndarray:
        """
        Filter anomaly scores based on confidence threshold.
        
        Args:
            scores: Confidence scores array
            threshold: Confidence threshold
            return_scores: If True, return scores instead of binary labels
            
        Returns:
            Filtered labels or scores
        """
        if return_scores:
            return scores
        return (scores > threshold).astype(int)
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate anomaly detection metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "anomaly_ratio": y_pred.mean()
        }

class SecurityUtils:
    """Security-related utilities"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal.
        
        Args:
            filename: Input filename
            
        Returns:
            Sanitized filename
        """
        # Remove directory paths
        filename = Path(filename).name
        # Remove special characters
        return re.sub(r'[^\w\-_.]', '_', filename)
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_dirs: List[str]) -> bool:
        """
        Validate file path is within allowed directories.
        
        Args:
            file_path: Path to validate
            allowed_dirs: List of allowed directories
            
        Returns:
            bool: True if path is valid
        """
        try:
            path = Path(file_path).resolve()
            return any(path.is_relative_to(Path(d).resolve()) for d in allowed_dirs)
        except Exception:
            return False