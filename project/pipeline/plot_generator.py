import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path
from datetime import datetime

class PlotGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.plot_config = config.get("plot_settings", {})
        self.logger = logging.getLogger(__name__)
        
        # Set default plot settings
        self.defaults = {
            "figsize": (12, 6),
            "dpi": 100,
            "normal_color": "#1f77b4",  # Matplotlib default blue
            "anomaly_color": "#d62728",  # Matplotlib default red
            "line_alpha": 0.7,
            "marker_size": 20,
            "title_fontsize": 14,
            "grid_alpha": 0.3
        }

    def plot_sensor_anomalies(self, data: pd.DataFrame, sensor_name: str, 
                             time_range: Optional[tuple] = None) -> plt.Figure:
        """Generate comprehensive anomaly visualization for a specific sensor.
        
        Args:
            data: DataFrame containing sensor data and anomaly labels
            sensor_name: Name of the sensor column to plot
            time_range: Optional tuple of (start, end) timestamps to zoom in
            
        Returns:
            matplotlib.Figure object
            
        Raises:
            ValueError: If sensor_name not found in data
            KeyError: If required columns are missing
        """
        try:
            # Validate input data
            self._validate_plot_input(data, sensor_name)
            
            # Create figure with configured settings
            fig, ax = plt.subplots(
                figsize=self.plot_config.get("figsize", self.defaults["figsize"]),
                dpi=self.plot_config.get("dpi", self.defaults["dpi"])
            )
            
            # Apply time range filter if specified
            plot_data = self._filter_time_range(data, time_range)
            
            # Plot the complete time series as background
            self._plot_time_series(ax, plot_data, sensor_name)
            
            # Highlight anomalies
            self._plot_anomalies(ax, plot_data, sensor_name)
            
            # Add reference line for mean value
            self._add_reference_line(ax, plot_data, sensor_name)
            
            # Configure plot appearance
            self._configure_plot_appearance(ax, sensor_name)
            
            # Add statistical information
            self._add_stats_annotation(ax, plot_data, sensor_name)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot for {sensor_name}: {str(e)}")
            raise

    def _validate_plot_input(self, data: pd.DataFrame, sensor_name: str):
        """Validate input data contains required columns."""
        if sensor_name not in data.columns:
            raise ValueError(f"Sensor {sensor_name} not found in data")
        if 'anomaly' not in data.columns:
            raise KeyError("Data must contain 'anomaly' column")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

    def _filter_time_range(self, data: pd.DataFrame, time_range: Optional[tuple]) -> pd.DataFrame:
        """Filter data to specified time range."""
        if time_range is not None:
            start, end = time_range
            return data.loc[start:end]
        return data

    def _plot_time_series(self, ax: plt.Axes, data: pd.DataFrame, sensor_name: str):
        """Plot the complete time series as background."""
        ax.plot(
            data.index,
            data[sensor_name],
            color=self.plot_config.get("line_color", self.defaults["normal_color"]),
            alpha=self.plot_config.get("line_alpha", self.defaults["line_alpha"]),
            label='Sensor Values',
            linewidth=1
        )

    def _plot_anomalies(self, ax: plt.Axes, data: pd.DataFrame, sensor_name: str):
        """Highlight anomaly points."""
        anomalies = data[data['anomaly'] == 1]
        if not anomalies.empty:
            ax.scatter(
                anomalies.index,
                anomalies[sensor_name],
                color=self.plot_config.get("anomaly_color", self.defaults["anomaly_color"]),
                s=self.plot_config.get("marker_size", self.defaults["marker_size"]),
                label='Anomaly',
                edgecolors='black',
                linewidths=0.5
            )

    def _add_reference_line(self, ax: plt.Axes, data: pd.DataFrame, sensor_name: str):
        """Add reference line for mean value."""
        mean_val = data[sensor_name].mean()
        ax.axhline(
            mean_val,
            color='green',
            linestyle='--',
            alpha=0.5,
            label=f'Mean ({mean_val:.2f})'
        )

    def _configure_plot_appearance(self, ax: plt.Axes, sensor_name: str):
        """Configure plot titles, labels, and grid."""
        ax.set_title(
            f"Anomaly Detection - {sensor_name}",
            fontsize=self.plot_config.get("title_fontsize", self.defaults["title_fontsize"])
        )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Sensor Value")
        ax.grid(alpha=self.plot_config.get("grid_alpha", self.defaults["grid_alpha"]))
        ax.legend()

    def _add_stats_annotation(self, ax: plt.Axes, data: pd.DataFrame, sensor_name: str):
        """Add statistical information to plot."""
        stats_text = (
            f"Total points: {len(data)}\n"
            f"Anomalies: {data['anomaly'].sum()}\n"
            f"Anomaly ratio: {data['anomaly'].mean():.2%}"
        )
        ax.annotate(
            stats_text,
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            ha='left',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def save_plot(self, fig: plt.Figure, filename: str) -> Path:
        """Save plot to configured image directory with validation.
        
        Args:
            fig: matplotlib Figure object
            filename: Name for the output file (without extension)
            
        Returns:
            Path to saved image file
            
        Raises:
            PermissionError: If unable to write to directory
            ValueError: If filename is invalid
        """
        try:
            # Validate filename
            if not filename:
                raise ValueError("Filename cannot be empty")
                
            # Ensure proper file extension
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                filename = f"{filename}.png"
                
            # Create output path
            output_dir = Path(self.config["directories"]["images"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            
            # Save figure
            kwargs = {
                "bbox_inches":"tight",
                "dpi":self.plot_config.get("dpi",self.defaults["dpi"])
            }

            # Add quality only if saving as JPEG/JPG
            if output_path.suffix.lower() in [".jpg",".jpeg"]:
                kwargs["quality"] = 95

            fig.savefig(output_path,**kwargs)
            plt.close(fig)
            
            self.logger.info(f"Successfully saved plot to {output_path}")
            return output_path
            
        except Exception as e:
            plt.close(fig)
            self.logger.error(f"Failed to save plot: {str(e)}")
            raise