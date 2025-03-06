# merge feature maps from all validated failures

import os
from pathlib import Path
import numpy as np
import pandas as pd

from features.feature_map import FeatureMap
from opensbt.utils.files import find_files_with_parent

# Folder containing the run files
folder_paths_msim = [
    "../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/",
    "../results/analysis/analysis_5-runs_17-09-2024_DB",
    "../results/analysis/analysis-5-runs_19-09-2024_BU"
    ]

folder_paths_sim_1 = [
   #"../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/",
    "../results/analysis/analysis_5-runs_17-09-2024_DB/"
   ]


validation_folder = "validation_count-3_udb"
validation_file = "1.0_fmap"
sim = "msim"
sim_1 = "sim_1"

output_file_name_prefix = f"overview_{sim}_"

n_runs = 5

#######################

json_files = []

save_folder = os.path.join(
    "../results/analysis/",
    "approximation_fail/"
)
Path(save_folder).mkdir(parents=True, exist_ok=True)

for folder_path in folder_paths_msim:
    for i in range(0,n_runs):
        path = os.path.join(folder_path,f"run_{i}/{sim}")
        print(path)
        files = find_files_with_parent(path + os.sep, 
                                    parent_folder=validation_folder, 
                                    prefix=validation_file)
        
        assert len(files) > 0

        print("found validation file", files[0])
        json_files.append(files[0])

for folder_path in folder_paths_sim_1:
    for i in range(0,n_runs):
        path = os.path.join(folder_path,f"run_{i}/{sim_1}")
        print(path)

        files = find_files_with_parent(path + os.sep, 
                                    parent_folder=validation_folder, 
                                    prefix=validation_file)
        
        assert len(files) > 0

        print("found validation file", files[0])
        json_files.append(files[0])

#########
# generate feature map
fmap2 = None
fmap1 = None
# create pop from test inputs
for json_file in json_files:
    fmap1 = FeatureMap.from_json(json_file) 
    if fmap2 is not None:
        fmap2 = FeatureMap.migrate(fmap1, fmap2)
    else:
        fmap2 = fmap1
    
fmap2.plot_map(save_folder, "fmap_all_valid_fail")
fmap2.export_to_json(f"{save_folder}/fmap_all_valid_fail.json")

print("Fmap stored in:", f"{save_folder}/fmap_all_valid_fail.json")