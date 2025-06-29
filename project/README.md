
# Assignment 5 – Machine Learning Pipeline for Anomaly Detection

The project simulates a real-world system for monitoring sensors, identifying anomalies, and visualizing outcomes. It demonstrates Python-based data engineering, anomaly detection, and visualization workflows.

---

## Project Structure

```

Assignment5/
├── data/
│   ├── raw/          # Original dataset (e.g., sensor.csv)
│   ├── processed/    # Split datasets (july\_data.csv & august\_data.csv)
│   ├── input/        # Directory watched for new files (simulates incoming data)
│   ├── output/       # Output predictions with anomalies
│   └── img/          # Generated plots for selected sensors
├── models/           # Persisted machine learning model
├── logs/             # Log files capturing the pipeline process
├── config/
│   └── application.json  # Configuration file with paths, sensors, and interval
├── scripts/          # Python modules and class implementations
│   ├── split\_data.py     # Splits original dataset
│   ├── run\_pipeline.py   # Main entry point that runs the entire pipeline
└── README.md

````

---

## Objective

To develop a **modular and production-ready pipeline** that:

- Trains a model using sensor data from April to June
- Listens for new data files (July and August) in a watch folder
- Applies transformations and anomaly detection
- Generates plots for specified sensors
- Logs all steps to a file
- Is fully configurable via a JSON file
- Follows **SOLID** software design principles

---

## Key Concepts Applied

- Object-Oriented Programming (OOP)
- List Comprehensions and Generators
- Static Code Analysis
- Parallelization (optional)
- Model Persistence (using `joblib`)
- Real-time File Monitoring
- JSON-based Configuration
- Refactoring functions to avoid global variables

---

## Getting Started

### 1. Activate your virtual environment

```bash
cd Assignment5
venv\Scripts\activate  # For Windows
````

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Split the dataset

```bash
python scripts/split_data.py --input data/raw/sensor.csv --output data/processed
```

This produces:

* `july_data.csv`
* `august_data.csv`

### 4. Run the complete pipeline

```bash
python scripts/run_pipeline.py
```

This command:

* Trains the model
* Watches the input folder (`data/input/`)
* Processes and predicts new files (e.g., `july_data.csv`, `august_data.csv`)
* Generates plots in `data/img/`
* Logs everything in `logs/`

---

## Configuration

Edit settings in `config/application.json`:

```json
{
  "input_dir": "data/input",
  "output_dir": "data/output",
  "img_dir": "data/img",
  "sensors_to_plot": ["sensor04", "sensor51"],
  "watch_interval": 10
}
```

---

## Outputs

For each `.csv` dropped into the `data/input/` directory:

* **Anomaly predictions** are saved in `data/output/`
* **Plots** are saved in `data/img/` (named with sensor and timestamp)
* **Logs** are written to `logs/app.log`

### Example log entries:

```
2024-06-11 10:09:32  Found new data file
2024-06-11 10:09:32  Loaded the file 
2024-06-11 10:09:34  Received transformed data
2024-06-11 10:09:38  Received predictions
2024-06-11 10:09:40  Saving image 2018-07-sensor04.png
2024-06-11 10:09:40  Resuming listening
```

---

## Testing the Pipeline

1. Copy either `july_data.csv` or `august_data.csv` into the `data/input/` directory.
2. Ensure `run_pipeline.py` is running.
3. The script will:

   * Detect the file
   * Process and predict anomalies
   * Save a results file
   * Save sensor plots
   * Log the entire process

---

## Design Principles

All components are built to follow the **SOLID principles**:

* **Single Responsibility** – Classes/modules do one job
* **Open/Closed** – Easily extend without modifying core logic
* **Liskov Substitution** – Subclassing will not break functionality
* **Interface Segregation** – Small, focused interfaces (or functions)
* **Dependency Inversion** – High-level modules don’t depend on low-level implementations

---

## Author

**Name:** Richard Boabeng
**LinkedIn:** [DSLS – Programming IV](https://www.linkedin.com/in/richard-boabeng-386992125/)


---

## References

* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [Kaggle Dataset – Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

