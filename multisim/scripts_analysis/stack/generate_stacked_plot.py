import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from opensbt.utils.files import find_files_with_parent
import matplotlib.pyplot as plt

# metadata
import csv
from datetime import datetime

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 16,      # Axis label font size
    'xtick.labelsize': 14,     # X-axis tick label font size
    'ytick.labelsize': 14,     # Y-axis tick label font size
    'legend.fontsize': 14,     # Legend label font size
    'legend.title_fontsize': 16  # Legend title font size
})

#######################
def generate_stack_valid(folder_paths,
                           combos,
                           sim_paths,
                           sims,
                           metrics_names =  ['valid_rate', 'n_valid'],
                           n_runs = 5,
                           save_folder = None,
                           seeds = None,
                           is_dss = None,
                           plot_names = None
                           ):
    data_plot = {}

    for f,folder_path in enumerate(folder_paths):
        csv_files = []
        
        if save_folder is None:
            save_folder = os.path.join(folder_path,f"overview_stacked/")
        
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        output_file_name_prefix = f"stack_valid_"
        is_dss = [int(v) for v in is_dss]
        
        if not is_dss[f]:
            validation_folder =f"validation_combined_count-3_{combos[f]}"
            validation_folder_re =f"validation_count-3_combined_{combos[f]}"
        else:
            validation_folder = f"validation_all_qu_dsim_combined_count-3.0_{sims[f]}_{combos[f]}"

        validation_file = f"validation_combined"
        
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
        
            data_plot[sims[f]] = [avg[1], 1 * avg[1]/avg[0]]

    # Write the overview to a file
    create_stacked_plot(data_plot,
                        save_folder=save_folder,
                        filename=f"{output_file_name_prefix}{threshold}.png",#
                        plot_names=plot_names)
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

def create_stacked_plot(data, save_folder, filename, plot_names):
    """
    input: data = {
        "DB": [23.9, 24],
        "BU": [12.3, 25],
        "UD": [21.1, 41],

        "B": [18.7, 39],
        "D": [26.8, 41],
        "U": [23.2, 53.9],

        "DSS-DB": [16.5, 19],
        "DSS-BU": [14.7, 28],
        "DSS-UD": [15.4, 30]
    }
    """
    import matplotlib.pyplot as plt

    # Prepare data
    if plot_names is not None:
        categories = plot_names
    else:
        categories = list(data.keys())

    valid_tests = [values[0] for values in data.values()]
    total_tests = [values[1] for values in data.values()]

    # Calculate non-valid tests and ratios
    non_valid_tests = [total - valid for total, valid in zip(total_tests, valid_tests)]
    ratios = [valid / total for valid, total in zip(valid_tests, total_tests)]

    # Create stacked bar plot
    plt.figure(figsize=(10, 6))

    bars_valid = plt.bar(categories, valid_tests, label='Valid Failures', color='orange')
    bars_non_valid = plt.bar(categories, non_valid_tests, bottom=valid_tests, label='Non-Valid Failures', color='skyblue')

    # Annotate ratios on the bars
    for bar, ratio in zip(bars_valid, ratios):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{ratio:.2f}",
            ha='center',
            va='center',
            fontsize=14,
            color='black'
        )

    # Adding titles and labels
    #plt.title("Stacked Bar Plot with Ratios of Valid Failures")
    plt.ylabel("Number of Failures")
    plt.xlabel("Method")

    plt.xticks(rotation=45)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(save_folder + os.sep + filename )
    plt.savefig(save_folder + os.sep + filename + ".pdf", format = "pdf" )

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
    parser.add_argument('--plot_names', nargs='+', default=None, help="Specify names to be plotted.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_stack_valid(args.folder_paths, 
                           args.combos, 
                           args.sim_paths, 
                           args.sims, 
                           args.metrics_names, 
                           args.n_runs,
                           args.save_folder,
                           args.seeds,
                           args.is_dss,
                           args.plot_names)
