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
def generate_pareto_plot(folder_paths,
                           combos,
                           sim_paths,
                           sims,
                           metrics_names =  ['valid_rate', 'n_valid'],
                           n_runs = 5,
                           save_folder = None,
                           seeds = None,
                           is_dss = None
                           ):
    data_plot = {}

    for f,folder_path in enumerate(folder_paths):
        csv_files = []
        
        if save_folder is None:
            save_folder = os.path.join(folder_path,f"overview_pareto/")
        
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        output_file_name_prefix = f"rate_nvalid_pareto_plot_"

        is_dss = [int(v) for v in is_dss]
        
        if not is_dss[f]:
            validation_folder =f"validation_combined_count-3_{combos[f]}"
            validation_folder_re =f"validation_count-3_combined_{combos[f]}"
        else:
            validation_folder = f"validation_all_qu_dsim_combined_count-3.0_{sims[f]}_{combos[f]}"
            validation_folder_re =f"validation_count-3_combined_{combos[f]}"

        validation_file = f"validation_combined"

        print(seeds)
        
        if seeds is None:
            selector =  range(0,n_runs)
        else:
            selector =  seeds

        for i in selector:
            print(sim_paths)
            path = os.path.join(folder_path,f"run_{i}/{sim_paths[f]}")
            print(path)
            files = find_files_with_parent(path + os.sep, 
                                        parent_folder=validation_folder, 
                                        prefix=validation_file)
            print(files)  

            if len(files) == 0:
                # try other name
                files = find_files_with_parent(path + os.sep, 
                                        parent_folder=validation_folder_re, 
                                        prefix=validation_file)
                print(files)  

            assert len(files) > 0

            print("found validation file", files[0])
            csv_files.append(files[0])
            # print(csv_files)
        print(csv_files)
        # # List of files in the folder
        # files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        thresholds = [1] # we have only 5 reruns

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

            print(avg)
            print(std)
        
            data_plot[sims[f]] = avg

        # Write the overview to a file
        create_pareto_plot(data_plot,
                            save_folder=save_folder,
                            filename=f"{output_file_name_prefix}{threshold}.png")
        print("Stacked plot file generated successfully.")

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

def create_pareto_plot(data, save_folder, filename):
    """

    input: 
    
    data = {
        "DB": [0.977907, 23.9],
        "BU": [0.490712, 12.3],
        "UD": [0.516077, 21.1],

        "B": [0.486177, 18.7],
        "D": [0.661983, 26.8],
        "U": [0.220887, 14.793392],

        "DSS-DB": [0.861305, 16.5],
        "DSS-BU": [0.517252, 14.7],
        "DSS-UD": [0.513082, 15.4]
    }
    """
    import matplotlib.pyplot as plt

   # Prepare data for the 2D plot
    categories = list(data.keys())
    valid_rates = [-values[0] for values in data.values()]
    n_valid_failures = [-values[1] for values in data.values()]

    # Create 2D scatter plot
    plt.figure(figsize=(10, 6))
    # Plot first three points
    plt.scatter(valid_rates[:3], n_valid_failures[:3], color='blue', label='First Three', alpha=0.7)

    # Plot second three points
    plt.scatter(valid_rates[3:6], n_valid_failures[3:6], color='green', label='Second Three', alpha=0.7)

    # Plot last three points
    plt.scatter(valid_rates[6:], n_valid_failures[6:], color='red', label='Last Three', alpha=0.7)

    # Annotate each point with its category
    for i, category in enumerate(categories):
        plt.text(valid_rates[i], n_valid_failures[i], category, fontsize=10, ha='right')

    # Adding titles and labels
    plt.title("2D Plot: Valid Rate vs. Number of Valid Failures")
    plt.xlabel("Valid Rate (neg)")
    plt.ylabel("Number of Valid Failures (neg)")
    plt.grid(alpha=0.5, linestyle="--")

    # Show the plot
    plt.tight_layout()
    plt.savefig(save_folder + os.sep + filename )

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate overview of simulation runs and summarize metrics.")
    
    # Add arguments
    parser.add_argument('--folder_paths',  nargs="+", help="Path to the folder containing results.")
    parser.add_argument('--combos', nargs="+", help="Combination identifier (e.g., 'bd_u').")
    parser.add_argument('--sim_paths', nargs="+", help="Simulation path (e.g., 'sim_1').")
    parser.add_argument('--sims', nargs="+", help="Simulation identifier (e.g., 'udacity').")
    
    # Optional arguments
    parser.add_argument('--metrics_names', nargs='+', default=['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid'],
                        help="List of metric names to process (default: standard metrics).")
    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs to analyze (default: 5).")
    parser.add_argument('--save_folder', type=str, help="Save folder to write results.", default = None)
    parser.add_argument('--seeds', nargs='+', default=None, help="Specify seed numbers for the runs if stored in seed folder.")
    parser.add_argument('--is_dss', nargs='+', default=None, help="Specify seed numbers for the runs if stored in seed folder.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_pareto_plot(args.folder_paths, 
                           args.combos, 
                           args.sim_paths, 
                           args.sims, 
                           args.metrics_names, 
                           args.n_runs,
                           args.save_folder,
                           args.seeds,
                           args.is_dss)
