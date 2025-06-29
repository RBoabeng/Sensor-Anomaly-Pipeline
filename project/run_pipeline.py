# run_pipeline.py
import logging
import json
from pathlib import Path
from pipeline.data_processor import DataProcessor
from pipeline.model_trainer import ModelTrainer
from pipeline.plot_generator import PlotGenerator

def setup_logging(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)  # ensure log dir exists
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    with open('config/application.json') as f:
        config = json.load(f)
        
    setup_logging(config["file_paths"]["log_file"])
    logger = logging.getLogger(__name__)

    try:
        logger.info("Found new data file")
        processor = DataProcessor(config)
        trainer = ModelTrainer(config)
        plotter = PlotGenerator(config)

        logger.info("Loading the file")
        train_data = processor.load_data(config["file_paths"]["train_data"])

        logger.info("Received transformed data")
        train_data = processor.transform_data(train_data)

        if train_data.empty:
            logger.error("Training data is empty! Please check your input data and preprocessing.")
            return  # Or continue listening if part of a loop

        logger.info("Received predictions")
        result = trainer.train_model(train_data)
        processed_data_with_anomalies = result.get("data")  # Make sure your train_model returns this

        logger.info("Saving predictions")
        # If you save predictions somewhere, add that logic here and log it

        sensors = config.get("visualization", {}).get("sensors_to_plot", [])
        for sensor in sensors:
            try:
                fig = plotter.plot_sensor_anomalies(processed_data_with_anomalies, sensor)
                img_path = plotter.save_plot(fig, f"{sensor}_anomaly_plot.png")
                logger.info(f"Saving image2 {img_path.name}")
            except Exception as e:
                logger.error(f"Failed to plot {sensor}: {e}")

        logger.info("Resuming listening")
    except Exception as e:
        logger.error(f"Unhandled error in pipeline: {e}")

if __name__ == "__main__":
    main()
