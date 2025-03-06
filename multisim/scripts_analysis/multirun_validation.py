import argparse
import glob
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from opensbt.utils.files import find_file, find_files
from opensbt.utils.encoder_utils import NumpyEncoder
from config import VALIDATION_THRESHOLD_RANGE

def load_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def multirun_validation(input_folder, 
                                  output_folder,
                                  n_runs = None,
                                  simulators = ['msim', 'sim_1', 'sim_2'],
                                  metrics =  ['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid'],
                                  do_plot = True,
                                  input_file_name = "validation_compared.csv",
                                  prefix_output = ""
                        ):
    
    Path(output_folder).mkdir(exist_ok=True,parents=True)

    if n_runs is not None:
        csv_files = find_files(input_folder + os.sep, input_file_name)[:n_runs]
    else:
        csv_files = find_files(input_folder + os.sep, input_file_name)
        
    data = pd.concat((pd.read_csv(file) for file in csv_files))

    grouped = data.groupby('threshold')

    for metric in metrics:
        results = pd.DataFrame(index=grouped.groups.keys(), columns=[f'{sim}_mean' for sim in simulators] + 
                                                                [f'{sim}_std' for sim in simulators])
        for threshold, group in grouped:
            for sim in simulators:
                # Extract the relevant columns
                values = group[f'{sim}_{metric}']
                # Calculate mean and standard deviation
                mean_val = np.mean(values)
                std_val = np.std(values)

                # Store the results
                results.loc[threshold, f'{sim}_mean'] = mean_val
                results.loc[threshold, f'{sim}_std'] = std_val
        
        # Reset the index to include the threshold as a column
        results.reset_index(inplace=True)
        results.rename(columns={'index': 'threshold'}, inplace=True)
        
        # Step 4: Save the results to CSV files
        results.to_csv(output_folder + os.sep + f'{prefix_output}{metric}_mean_std.csv', index=False)
                    
        plot_folder = output_folder

        if do_plot:

            x_displacement = np.linspace(-0.005, 0.005, len(simulators))  # Create small displacements

            #  Step 5: Plotting
            plt.figure(figsize=(10, 6))
            for i, sim in enumerate(simulators):
                plt.errorbar(results['threshold'] + x_displacement[i], results[f'{sim}_mean'], yerr=results[f'{sim}_std'], 
                            label=f'{sim}', fmt='-o', capsize=5)
            
            plt.xlabel('Threshold')
            plt.ylabel(f'{metric} (Mean ± Std)')
            plt.title(f'{metric.capitalize()} vs. Threshold')
            plt.legend()
            plt.grid(True)
            
            # Save the plot as an image file
            plt.savefig(plot_folder + f'{prefix_output}{metric}_plot.png')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multirun validation with specified parameters.")

    # Required arguments
    parser.add_argument("--input_folder", type=str, help="Path to the input folder", required=True)
    parser.add_argument("--output_folder", type=str, help="Path to the output folder", required=True)

    # Optional arguments
    parser.add_argument("--n_runs", type=int, help="Number of runs for the validation", default=None)
    parser.add_argument("--simulators", nargs='+', type=str, help="List of simulators to use", 
                        default=['msim', 'sim_1', 'sim_2'])
    parser.add_argument("--metrics", nargs='+', type=str, help="List of metrics to evaluate", 
                        default=['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid'])
    parser.add_argument("--do_plot", action="store_true", help="Flag to generate plots", default=False)
    
    parser.add_argument("--input_file_name", default="validation_compared.csv", type=str, help="Input file name")
    parser.add_argument("--prefix_output", default="", type=str, help="Path to the output folder")

    args = parser.parse_args()

    multirun_validation(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        n_runs=args.n_runs,
        simulators=args.simulators,
        metrics=args.metrics,
        do_plot=args.do_plot,
        input_file_name=args.input_file_name,
        prefix_output=args.prefix_output
    )
    
    # # output_folder = "/home/lev/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_5-runs_22-08-2024_13-15-06"
    # # output_folder = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_2-runs_27-08-2024_10-28-43_5-gen_5-pop"
    # # output_folder = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_2-runs_30-08-2024_00-48-40_3-gen_3-pop"
    # output_folder = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_1-runs_02-09-2024_20-23-24_10-gen_10-pop"
    
    # n_runs = 2

    # multirun_validation(input_folder=output_folder,
    #                     output_folder = output_folder,
    #                     n_runs = n_runs)
