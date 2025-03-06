import argparse
import shutil
import os
from pathlib import Path
from opensbt.utils.files import concatenate_csv, find_file, find_file_contains, find_parent
from features.feature_map import FeatureMap
import gc

def write_combined_results(folder_valid_0, folder_valid_1, write_folder, seed):
    ''' Files required in the folders:
            backup/problem
            all_critical_testcases.csv
            fmap*seed*.json
    '''
    path_problem = find_file(folder_valid_0 + os.sep, "problem")

    Path(write_folder).mkdir(parents=True,exist_ok=True)
    Path(write_folder + os.sep + "backup").mkdir(parents=True,exist_ok=True)
    
    shutil.copy(path_problem, write_folder + os.sep + "backup" + os.sep + "problem")
    #######################

    # path_testcases = save_folder + os.sep + "all_critical_testcases.csv"

    path_t1 = find_file(folder_valid_0 + os.sep, "all_critical_testcases")
    path_t2 = find_file(folder_valid_1 + os.sep, "all_critical_testcases")
    
    concatenate_csv(path_t1, path_t2, write_folder + "all_critical_testcases")
    #############

    path_fmap_1 =  find_file_contains(folder_valid_0 + os.sep, ["fmap","seed"])
    path_fmap_2 =  find_file_contains(folder_valid_1 + os.sep, ["fmap","seed"])
    
    fmap_1: FeatureMap = FeatureMap.from_json(path_fmap_1)
    fmap_2: FeatureMap = FeatureMap.from_json(path_fmap_2)
    
    ##################

    fmap = FeatureMap.migrate(fm1=fmap_1, fm2=fmap_2)
    fmap.plot_map(filepath=write_folder + os.sep)
    fmap.export_to_json(write_folder + os.sep + f"seed-{seed}_" + "fmap.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine validation results into a single output.")

    parser.add_argument("--folder_valid_0", type=str, required=True, help="Path to the folder containing validation results 0")
    parser.add_argument("--folder_valid_1", type=str, required=True, help="Path to the folder containing validation results 1")
    parser.add_argument("--write_folder", type=str, required=True, help="Path to the folder where combined results will be written")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for the analysis")

    args = parser.parse_args()

    write_combined_results(
        folder_valid_0=args.folder_valid_0,
        folder_valid_1=args.folder_valid_1,
        write_folder=args.write_folder,
        seed=args.seed
    )
