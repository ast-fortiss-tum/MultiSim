import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from opensbt.utils.files import find_files_with_parent

# metadata
import csv
from datetime import datetime

#######################
def generate_overview_runs(folder_path,
                           combo,
                           sim_path,
                           sim,
                           metrics_names =  ['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid'],
                           n_runs = 5,
                           save_folder = None,
                           seeds = None
                           ):
    csv_files = []

    if save_folder is None:
        save_folder = os.path.join(folder_path,f"overview_{sim}_{combo}/")
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    output_file_name_prefix = f"overview_{sim}_{combo}_"

    validation_folder =f"validation_combined_count-3_{combo}"
    validation_file = f"validation_combined"

    print(seeds)
    
    if seeds is None:
        selector =  range(0,n_runs)
    else:
        selector =  seeds

    for i in selector:
        path = os.path.join(folder_path,f"run_{i}/{sim_path}")
        print(path)
        files = find_files_with_parent(path + os.sep, 
                                    parent_folder=validation_folder, 
                                    prefix=validation_file)
        print(files)    
        assert len(files) > 0

        print("found validation file", files[0])
        csv_files.append(files[0])
        # print(csv_files)
    print(csv_files)
    # # List of files in the folder
    # files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    thresholds = [0.6, 0.8, 1] # we have only 5 reruns

    for threshold in thresholds:
        # List to store the metrics for each run
        metrics = []

        # Read each file and extract the data for threshold 0.5 (first row)
        for i, file in enumerate(sorted(csv_files)):
            # file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file)
            
            row = data[data['threshold'] == threshold]
            if not row.empty:
                metrics.append(row[metrics_names].values[0])

        # Convert to a NumPy array for easier processing
        metrics = np.array(metrics)

        # Calculate averages and standard deviations for each metric
        avg = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)

        # Create the overview table
        overview = pd.DataFrame(metrics, columns=metrics_names)
        overview.index += 1  # Run numbers

        # Add the averages and standard deviations as new rows
        overview.loc['avg'] = avg
        overview.loc['std'] = std

        # Write the overview to a file
        overview.to_csv(f"{save_folder}{output_file_name_prefix}{threshold}.csv", index_label="run", float_format="%.6f")

        print("Overview file generated successfully.")

    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data = [
        ["timestamp", str(timestamp)],
        ["input_files", csv_files]
    ]

    # Write data to CSV file
    with open(save_folder + os.sep + "metadata.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate overview of simulation runs and summarize metrics.")
    
    # Add arguments
    parser.add_argument('--folder_path', type=str, help="Path to the folder containing results.")
    parser.add_argument('--combo', type=str, help="Combination identifier (e.g., 'bd_u').")
    parser.add_argument('--sim_path', type=str, help="Simulation path (e.g., 'sim_1').")
    parser.add_argument('--sim', type=str, help="Simulation identifier (e.g., 'udacity').")
    
    # Optional arguments
    parser.add_argument('--metrics_names', nargs='+', default=['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid'],
                        help="List of metric names to process (default: standard metrics).")
    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs to analyze (default: 5).")
    parser.add_argument('--save_folder', type=str, help="Save folder to write results.", default = None)
    parser.add_argument('--seeds', nargs='+', default=None, help="Specify seed numbers for the runs if stored in seed folder.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_overview_runs(args.folder_path, 
                           args.combo, 
                           args.sim_path, 
                           args.sim, 
                           args.metrics_names, 
                           args.n_runs,
                           args.save_folder,
                           args.seeds)
