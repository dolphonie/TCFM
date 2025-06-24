import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse

def convert_timestamp(ts):
    return datetime.fromisoformat(ts).timestamp()

def csv_to_npz(csv_file, output_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to Unix timestamp
    df['timestamp'] = df['timestamp'].apply(convert_timestamp)
    
    # Save each feature as a separate NPZ file
    for column in df.columns:
        np.savez(os.path.join(output_dir, f"{column}.npz"), df[column].values)
    
    print(f"Converted {csv_file} to NPZ files in {output_dir}")

def process_directory(input_dir, output_base_dir):
    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each CSV file in the input directory
    counter = 0
    for filename in os.listdir(input_dir):
        if '.csv' in filename:
            # Get the file number (assuming filenames are like '0.csv', '1.csv', etc.)
            file_number = os.path.splitext(filename)[0]
            
            # Create output directory for this trajectory
            output_dir = os.path.join(output_base_dir, f"trajectory_{file_number}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the CSV file
            csv_file_path = os.path.join(input_dir, filename)
            csv_to_npz(csv_file_path, output_dir)
            
        counter+=1
        if counter > 100:
            break

def main():
    parser = argparse.ArgumentParser(description="Convert multiple CSV files to NPZ files")
    parser.add_argument("-input_dir", help="Path to the input directory containing CSV files")
    parser.add_argument("-output_dir", help="Path to the base output directory for NPZ files")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()