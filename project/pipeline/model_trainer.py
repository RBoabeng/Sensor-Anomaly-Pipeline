from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any
import logging
from datetime import datetime

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare training data"""
        try:
            processed = raw_data.drop(columns=self.config.get('columns_to_drop', []))
            processed = processed.replace([np.inf, -np.inf], np.nan)
            processed = processed.fillna(processed.mean())
            return processed.select_dtypes(include=['float64'])
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    from pathlib import Path

    def train_model(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            params = self.config.get("model", {}).get("parameters", {})
            model = IsolationForest(
                n_estimators=params.get('n_estimators', 100),
                contamination=params.get('contamination', 0.05),
                random_state=params.get('random_state', 42),
                max_features=params.get('max_features', 1.0)
            )
            processed_data = self.preprocess_data(train_data)
            model.fit(processed_data)

            predictions = model.predict(processed_data)
            processed_data['anomaly'] = (predictions == -1).astype(int)

            self._save_model(model, processed_data.columns.tolist())

            output_dir = Path(self.config["directories"]["output"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "train_with_anomalies.csv"
            processed_data.to_csv(output_path, index=True)
            self.logger.info(f"Processed training data with anomalies saved to: {output_path}")

            return {"model": model, "data": processed_data}

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise



    def _save_model(self, model, feature_names: list) -> Dict[str, Any]:
        """Persist trained model with metadata"""
        model_info = {
            'model': model,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'feature_names': feature_names,
            'version': '1.0'
        }
        joblib.dump(model_info, self.config["file_paths"]["model_file"])
        self.logger.info(f"Model saved to {self.config['file_paths']['model_file']}")
        return model_info

    def load_model(self) -> Dict[str, Any]:
        """Load trained model with verification"""
        try:
            model_info = joblib.load(self.config["file_paths"]["model_file"])
            if not isinstance(model_info, dict) or 'model' not in model_info:
                raise ValueError("Invalid model file format")
            return model_info
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise