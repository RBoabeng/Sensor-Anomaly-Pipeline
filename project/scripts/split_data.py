import pandas as pd
from pathlib import Path

def split_pump_data(input_file: str = "data/raw/sensor.csv", output_dir: str = "data/raw"):
    """Split the raw dataset into training (Apr-Jun), July, and August files."""
    # Convert to Path objects for robust path handling
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Verify input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    
    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the raw data
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    
    # Filter by date ranges
    train_data = df[df['timestamp'].between('2018-04-01', '2018-06-30')]
    july_data = df[df['timestamp'].between('2018-07-01', '2018-07-31')]
    august_data = df[df['timestamp'].between('2018-08-01', '2018-08-31')]
    
    # Save to separate files
    train_data.to_csv(output_path / "training_data.csv", index=False)
    july_data.to_csv(output_path / "july_data.csv", index=False)
    august_data.to_csv(output_path / "august_data.csv", index=False)
    
    print(f"Successfully split data:")
    print(f"  - Training set (Apr-Jun): {output_path/'training_data.csv'} ({len(train_data)} rows)")
    print(f"  - July test set: {output_path/'july_data.csv'} ({len(july_data)} rows)")
    print(f"  - August test set: {output_path/'august_data.csv'} ({len(august_data)} rows)")

if __name__ == "__main__":
    # Use default paths but allow command-line overrides
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="data/raw/sensor.csv", help="Input CSV path")
    parser.add_argument('--output', default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    split_pump_data(input_file=args.input, output_dir=args.output)