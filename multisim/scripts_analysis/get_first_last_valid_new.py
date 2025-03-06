import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pymoo
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.utils.files import find_file, find_files_with_parent
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

MAX_EVALUATIONS_FIRST_LAST_VALID = 500

def read_pf_single(filename):
    print("filename", filename)
    individuals = []
    table = pd.read_csv(filename)
    n_var = -1
    k = 0
    # identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        k = k + 1
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        ind = IndividualSimulated()
        ind.set("X", X)
        ind.set("F", F)
        individuals.append(ind)
    print(f"Reading of {len(individuals)} tests completed")
    return PopulationExtended(individuals=individuals)

def get_first_last_valid(path_all_tests,
                         valid_tests,
                         save_folder,
                         output_filename_prefix
                         ):
    
    pop = read_pf_single(path_all_tests)

    # read in all valid
    with open(valid_tests, 'r') as file:
        data = json.load(file)

    # save_folder = os.path.dirname(valid_tests)
    angles_valid = data["angles_valid"]

    # check for each when it was first encountered in pop_mix
    evaluation_num_valid = []

    for x_valid in angles_valid:
        for i, x in enumerate(pop.get("X")):
            if np.array_equal(x, x_valid):
                evaluation_num_valid.append(i)
                break
    if len(evaluation_num_valid) == 0:
        road_index_first = None
        road_index_last = None

         # # write done
        # with open(save_folder + os.sep + f'{output_filename_prefix}.csv', 'w', encoding='UTF8', newline='') as f:
        #     write_to = csv.writer(f)
        #     header = ['item', 'evaluation', 'test']
        #     write_to.writerow(header)
        #     write_to.writerow([f'first_valid', 30, None])
        #     write_to.writerow([f'last_valid', 30, None])
        # we set the maximum number of evals of 600 if non valid was found
        return  [MAX_EVALUATIONS_FIRST_LAST_VALID, MAX_EVALUATIONS_FIRST_LAST_VALID, None, None]
    else:

        # assert len(evaluation_num_valid) > 0
                
        # print("evaluations_valid", evaluation_num_valid)
        road_index_first = np.argmin(evaluation_num_valid)          
        road_index_last = np.argmax(evaluation_num_valid)
        
        # print("road_index_first", road_index_first)
        # print("road_index_last", road_index_last)  
        # print("len valid", len(angles_valid))

        # print("first valid road:", angles_valid[road_index_first])
        # print("found at:", min(evaluation_num_valid))
        
        # print("last valid road:", angles_valid[road_index_last])
        # print("found at:", max(evaluation_num_valid))
        
        # # write done
        # with open(save_folder + os.sep + f'{output_filename_prefix}.csv', 'w', encoding='UTF8', newline='') as f:
        #     write_to = csv.writer(f)
        #     header = ['item', 'evaluation', 'test']
        #     write_to.writerow(header)
        #     write_to.writerow([f'first_valid', min(evaluation_num_valid), angles_valid[road_index_first]])
        #     write_to.writerow([f'last_valid', max(evaluation_num_valid), angles_valid[road_index_last]])

    return  [min(evaluation_num_valid), max(evaluation_num_valid),  angles_valid[road_index_first], angles_valid[road_index_last]]
###########################

if __name__ == "__main__":
    paths = [
            "../msim-results/analysis/analysis_5-runs_26-11-2024_BU/",
            "../msim-results/analysis/analysis_5-runs_17-09-2024_DB/",
            "../msim-results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop/"
        ]

    validation_folders = [
        "validation_combined_count-3_d",
        "validation_combined_count-3_u",
        "validation_combined_count-3_b"
    ]

    sim_map = ["msim",
            "msim",
            "msim"
    ]

    sim_names = [
        "ub",
        "bd",
        "ud"
    ]

    save_folder_all = r"..\msim-results\analysis\analysis_5-runs_msim_29-09-24\\"

    from datetime import datetime

    date = datetime.now().strftime(f"%d-%m-%Y_%H-%M-%S")

    for j, folder_path in enumerate(paths):
        
        # Folder containing the run files
        # folder_path = "../results/analysis/analysis-5-runs_19-09-2024_BU/"
        validation_folder = validation_folders[j]
        
        threshold=1.0
        validation_file = f"validation_{threshold}"
        testcases_file = "all_testcases"

        sim = sim_map[j]
        # sim_name = "u"
        
        output_file_name_prefix = f"overview_first_last_valid_time_{threshold}"

        save_folder = os.path.join(save_folder_all,f"efficiency_{sim_names[j]}_{date}/")
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        n_runs = 5

        #######################

        csv_files = []
        results_all = []

        paths_valid_tests = []
        
        for i in range(0,n_runs):

            # HACK for now
            # if j == 0 and i == 2:
            #     continue
            path = os.path.join(folder_path,f"run_{i}/{sim}")

            path_all_tests = find_files_with_parent(path + os.sep, 
                                        prefix=testcases_file,
                                        parent_folder="myfolder")[0]
            
            print("path_all_tests", path_all_tests)

            valid_tests = find_files_with_parent(path + os.sep, 
                                        prefix=validation_file,
                                        parent_folder=validation_folder)[0]
            print("valid_tests", valid_tests)

            results_run = get_first_last_valid(path_all_tests,
                                valid_tests,
                                save_folder,
                                output_file_name_prefix
                                )
            
            # multiply by 2 to get simulation number
            results_run[0] = 2*results_run[0]
            results_run[1] = 2*results_run[1]
            results_all.append(results_run)
            
            paths_valid_tests.append(valid_tests)

        with open(save_folder + os.sep + f'{output_file_name_prefix}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            header = ['run', 'first_valid','last_valid', 'test_first', 'test_last']
            write_to.writerow(header)
            first = []
            last = []

            for i, result in enumerate(results_all):
                write_to.writerow([f'{i}', result[0], result[1], result[2], result[3]])
                first.append(result[0])
                last.append(result[1])
            
            BATCH_SIZE = 20

            std_first = np.asarray(first).std()
            mean_first = np.asarray(first).mean()

            iter_std_first =  round(np.asarray(first).std() / BATCH_SIZE, 1)
            iter_mean_first = round(np.asarray(first).mean() / BATCH_SIZE, 1)
            
            std_last = np.asarray(last).std()
            mean_last = np.asarray(last).mean()
            
            iter_std_last =  round(np.asarray(last).std() / BATCH_SIZE, 1)
            iter_mean_last = round(np.asarray(last).mean() / BATCH_SIZE, 1)
            
            write_to.writerow(["avg", mean_first, mean_last, "-1", "-1"])
            write_to.writerow(["std", std_first, std_last, "-1", "-1"])
            
            write_to.writerow(["avg_iter", iter_mean_first, iter_mean_last, "-1", "-1"])
            write_to.writerow(["std_iter", iter_std_first, iter_std_last, "-1", "-1"])

        print("evaluation stored in:", save_folder + os.sep + f'{output_file_name_prefix}.csv')
        
        # Generate the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        data = [
            ["timestamp", str(timestamp)],
            ["files_valid_tests:", str(paths_valid_tests)]
        ]

        # Write data to CSV file
        with open(save_folder + os.sep + "metadata.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            

    # if __name__ == "__main__":
        
    #     parser = argparse.ArgumentParser(description='Process test and valid test case files.')
        
    #     # Add arguments for path_all_tests and valid_tests
    #     parser.add_argument('--path_all_tests', type=str, required=True, 
    #                         help='Path to the CSV file containing all test cases')
    #     parser.add_argument('--path_valid_tests', type=str, required=True, 
    #                         help='Path to the JSON file containing valid test cases')
        
    #     # Parse arguments from the command line
    #     args = parser.parse_args()
        
    #     # Call the function with parsed arguments
    #     get_first_last_valid(args.path_all_tests, args.path_valid_tests)
        