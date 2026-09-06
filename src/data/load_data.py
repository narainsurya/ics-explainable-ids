import pandas as pd
import os

# Path to the raw CSV (relative to the project root)
RAW_DATA_PATH = os.path.join("data", "raw", "gas_pipeline.csv")

# ---------------------------------------------------------------------------
# DAY 1 TODO: 
# Replace the KEYS (strings on the left) with the exact column names from your CSV.
# DO NOT change the VALUES (strings on the right) - the team relies on these.
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "timestamp": "time",
    "pressure": "pressure",
    "flow_rate": "flow_rate",
    "temperature": "temperature",
    "valve_status": "command",  # Mapping valve status to the expected 'command' column
    "target": "label",          # Usually 0 for normal, 1 for attack
    "event_type": "attack_type" # The specific name of the attack or event
}

def load_dataset(filepath=RAW_DATA_PATH):
    """Loads the raw dataset and renames columns to the expected schema."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please create the data/raw/ directory and place the CSV there.")
    
    df = pd.read_csv(filepath)
    
    # Check if all keys in COLUMN_MAP actually exist in the raw dataset
    missing_cols = [col for col in COLUMN_MAP.keys() if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following required columns were not found in the raw CSV: {missing_cols}")
    
    # Keep only the mapped columns and rename them
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    return df

def summarize(df):
    """Prints sanity checks: row counts, labels, and sensor ranges."""
    print("="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total Rows: {len(df):,}\n")
    
    print("--- Label Counts (Normal vs. Attack) ---")
    print(df['label'].value_counts(dropna=False).to_string(), "\n")
    
    print("--- Attack Type Counts ---")
    print(df['attack_type'].value_counts(dropna=False).to_string(), "\n")
    
    print("--- Sensor Ranges ---")
    sensors = ['pressure', 'flow_rate', 'temperature']
    for sensor in sensors:
        if sensor in df.columns:
            # Ensure the column is numeric so min/max calculation works
            df[sensor] = pd.to_numeric(df[sensor], errors='coerce')
            s_min = df[sensor].min()
            s_max = df[sensor].max()
            print(f"{sensor.capitalize()}: Min = {s_min:.2f} | Max = {s_max:.2f}")
        else:
            print(f"{sensor.capitalize()}: NOT FOUND IN DATA")
    print("="*50)

if __name__ == "__main__":
    print("Loading raw data and mapping columns...")
    try:
        df_mapped = load_dataset()
        summarize(df_mapped)
    except Exception as e:
        print(f"ERROR: {e}")