import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.required_columns = config.get("processing", {}).get("required_columns", [])
        self.columns_to_drop = config.get("processing", {}).get("columns_to_drop", [])
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        if not self.required_columns:
            self.logger.warning("No required columns specified in config")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data file.
        
        Args:
            file_path: Path to input CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
                
            df = pd.read_csv(file_path)
            
            # Validate columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            self.logger.info(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all data transformations.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Make copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # 1. Drop specified columns
            df = self._drop_columns(df)
            
            # 2. Handle timestamp conversion
            df = self._process_timestamp(df)
            
            # 3. Handle missing values
            df = self._handle_missing_values(df)
            
            # 4. Additional cleaning
            df = self._clean_data(df)
            
            self.logger.info("Data transformations completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            raise

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns specified in config"""
        cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        return df.drop(columns=cols_to_drop, errors='ignore')

    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert and set timestamp column"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing/infinite values"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with column means
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Drop remaining rows with missing values
        return df.dropna()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional data cleaning steps"""
        # Remove duplicate indices
        if df.index.duplicated().any():
            self.logger.warning("Found duplicate timestamps - keeping first occurrence")
            df = df[~df.index.duplicated(keep='first')]
            
        # Sort by timestamp
        return df.sort_index()

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to specified location.
        
        Args:
            df: Processed DataFrame
            output_path: Destination path for saving
            
        Raises:
            PermissionError: If unable to write to destination
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp in filename if it doesn't exist
            if output_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_path.with_stem(f"{output_path.stem}_{timestamp}")
                
            df.to_csv(output_path, index=True)
            self.logger.info(f"Successfully saved processed data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {str(e)}")
            raise