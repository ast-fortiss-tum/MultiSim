# fmap_failures
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opensbt.utils.files import find_files_with_parent


fmap_all_fail = "/home/sorokin/Projects/testing/Multi-Simulation/results/analysis/approximation_fail/fmap_all_valid_fail.json"


def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

n_cell_fail_real = load_json(fmap_all_fail)["n_cell_fail"]

###################

# Folder containing the run files
folder_paths_msim = [
    # "../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/",
    # "../results/analysis/analysis_5-runs_17-09-2024_DB",
    # "../results/analysis/analysis-5-runs_19-09-2024_BU"
    ]

folder_paths_sim_1 = [
   "../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/",
    #"../results/analysis/analysis_5-runs_17-09-2024_DB/"
   ]

# folder_path = "../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/"
validation_folder = "validation_count-3_udb"
sims = ["msim", "sim_1"]

n_runs = 5

#######################
thresholds = [0.6, 0.8, 1.0]

for threshold in thresholds:
    validation_file = f"{threshold}_fmap"

    for i, folder_paths in enumerate([folder_paths_msim,folder_paths_sim_1]):
        sim = sims[i]
        for folder_path in folder_paths:
            save_folder = os.path.join(
                folder_path,
                "failure_coverage/"
            )

            output_file_name_prefix = f"overview_fail_cov_{sim}_"


            Path(save_folder).mkdir(parents=True, exist_ok=True)
            json_files = []

            for i in range(0,n_runs):
                path = os.path.join(folder_path,f"run_{i}/{sim}")
                print(path)
                files = find_files_with_parent(path + os.sep, 
                                            parent_folder=validation_folder, 
                                            prefix=validation_file)
                
                assert len(files) > 0

                print("found validation file", files[0])
                json_files.append(files[0])

            # write down in comparison file
            # List to store the metrics for each run
            metrics = []

            # Read each file and extract the data for threshold 0.5 (first row)
            for i, file in enumerate(sorted(json_files)):
                data = load_json(file)
                n_cell_fail_found = data["n_cell_fail"]
                print("read n fail", n_cell_fail_found)
                metrics.append(n_cell_fail_found/n_cell_fail_real)

            # Convert to a NumPy array for easier processing
            metrics = np.array(metrics)

            print(metrics)
            metrics_names = [
                "fail_cov"
            ]
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

            print("Overview file generated successfully:",f"{save_folder}{output_file_name_prefix}{threshold}.csv")
