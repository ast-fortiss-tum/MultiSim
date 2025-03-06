import argparse
import csv
import json
import os
from pathlib import Path
from config import MAX_EVALS_FIRST_VALID_MAP
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
                         max_evals
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
        return  [max_evals, max_evals, None, None]
    else:
        # assert len(evaluation_num_valid) > 0
        # print("evaluations_valid", evaluation_num_valid)
        road_index_first = np.argmin(evaluation_num_valid)          
        road_index_last = np.argmax(evaluation_num_valid)
        
    return  [min(evaluation_num_valid), max(evaluation_num_valid),  angles_valid[road_index_first], angles_valid[road_index_last]]

def get_first_last_valid_all_dss(paths,
                                 paths_second_dss,
                             validation_folders,
                             sim_map,
                             sim_names,
                             save_folder_all,
                             seeds = None,
                             n_runs = None):
                             
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
        
        output_file_name_prefix = f"overview_first_last_valid_{threshold}"
        output_file_name_prefix_relative = f"overview_first_last_valid_rel_{threshold}"

        save_folder = os.path.join(save_folder_all,f"efficiency_dss_{sim_names[j]}/")
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        n_runs = 5

        #######################

        csv_files = []
        results_all = []

        paths_valid_tests = []
          
        if seeds is None:
            selector =  range(0,n_runs)
        else:
            selector =  seeds

        for i in selector:
            # AA'BB' (A' is validation of A, B' is validation of B)
            # calculate if the error is found in the first A or second dss B
            # if it is in the first, add the search effor for A with its evaluation number
            # if it is in the second, add twice the search effort in A + the one from B + the evaluation number

            # get first the evaliation numbers for A and 

            # calculate whether test is in A or B

            # apply formel

            path = os.path.join(folder_path,f"run_{i}/{sim}")

            print(path)
            print(testcases_file)
            path_all_tests_A = find_files_with_parent(path + os.sep, 
                                        prefix=testcases_file,
                                        parent_folder="myfolder")[0]
            
            pop = read_pf_single(path_all_tests_A)

            n_evals_A = int(len(pop) / 4) # quarter search budget

            print("path_all_tests", path_all_tests_A)

            # second dss
            path = os.path.join(folder_path,f"run_{i}/{sim}")

            path_second_dss = os.path.join(paths_second_dss[j],f"run_{i}/{sim}")

            path_all_tests_B = find_files_with_parent(path_second_dss + os.sep, 
                                        prefix=testcases_file,
                                        parent_folder="myfolder")[0]
            
            pop = read_pf_single(path_all_tests_B)

            n_evals_B = int(len(pop) / 4) # quarter search budget

            print("path_all_tests", path_all_tests_B)

            ##########
            valid_tests = find_files_with_parent(path + os.sep, 
                                        prefix=validation_file,
                                        parent_folder=validation_folder)[0]
            print("valid_tests", valid_tests)
            
            # sim_name = sim_names[j]
            
            # max_evals = MAX_EVALS_FIRST_VALID_MAP["dss-" + sim_name]  \
            #                 if "dss-" + sim_name in MAX_EVALS_FIRST_VALID_MAP else \
            #                         MAX_EVALS_FIRST_VALID_MAP["dss-" + sim_name[::-1]]
            
            results_run_A = get_first_last_valid(path_all_tests_A,
                                valid_tests,
                                MAX_EVALUATIONS_FIRST_LAST_VALID)
            
            results_run_B = get_first_last_valid(path_all_tests_B,
                                valid_tests,
                                MAX_EVALUATIONS_FIRST_LAST_VALID)
            
            results_run = []

            # considering evaluations

            if results_run_A[0] == MAX_EVALUATIONS_FIRST_LAST_VALID:
                # first valid is in the second search
                results_run_first_valid = 2*n_evals_A + n_evals_B + results_run_B[0]
                results_run_first_test = results_run_B[2]
            else:
                # first valid is in the second search
                results_run_first_valid = n_evals_A + results_run_A[0]
                results_run_first_test = results_run_A[2]

            if results_run_A[1] == MAX_EVALUATIONS_FIRST_LAST_VALID:
                # first valid is in the second search
                results_run_last_valid = 2*n_evals_A + n_evals_B + results_run_B[1]
                results_run_last_test = results_run_B[3]
            else:
                # first valid is in the second search
                results_run_last_valid = n_evals_A + results_run_A[1]
                results_run_last_test = results_run_A[3]

            # considering iterations

            # if results_run_A[0] == MAX_EVALUATIONS_FIRST_LAST_VALID:
            #     # first valid is in the second search
            #     results_run_first_valid = n_evals_A + results_run_B[0]
            #     results_run_first_test = results_run_B[2]
            # else:
            #     # first valid is in the second search
            #     results_run_first_valid = results_run_A[0]
            #     results_run_first_test = results_run_A[2]

            # if results_run_A[1] == MAX_EVALUATIONS_FIRST_LAST_VALID:
            #     # first valid is in the second search
            #     results_run_last_valid = n_evals_A + results_run_B[1]
            #     results_run_last_test = results_run_B[3]
            # else:
            #     # first valid is in the second search
            #     results_run_last_valid = results_run_A[0]
            #     results_run_last_test = results_run_A[3]

            results_run = [
                results_run_first_valid,
                results_run_last_valid,
                results_run_first_test,
                results_run_last_test
            ]
            print("results_run", results_run)
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
        
        # budget relative
        max_evals_total = 2*n_evals_A + 2*n_evals_B

        for i, result in enumerate(results_all):
            result[0] = result[0] / max_evals_total
            result[1] = result[1] / max_evals_total
            first.append(result[0])
            last.append(result[1])

        with open(save_folder + os.sep + f'{output_file_name_prefix_relative}.csv', 'w', encoding='UTF8', newline='') as f:
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
           
            std_last = np.asarray(last).std()
            mean_last = np.asarray(last).mean()
            
            write_to.writerow(["avg", mean_first, mean_last, "-1", "-1"])
            write_to.writerow(["std", std_first, std_last, "-1", "-1"])

        print("evaluation stored in:", save_folder + os.sep + f'{output_file_name_prefix_relative}.csv')

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
            


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process test and valid test case files.')
    
    # Correct the argument definitions
    parser.add_argument('--paths', nargs='+', required=True, 
                        help='List of paths to test case directories')
    parser.add_argument('--paths_second_dss', nargs='+', required=True, 
                        help='List of paths to test case directories for second dss')
    parser.add_argument('--validation_folders', nargs='+', required=True, 
                        help='List of validation folders')
    parser.add_argument('--sim_map', nargs='+', required=True, 
                        help='List of simulation mappings')
    parser.add_argument('--sim_names', nargs='+', required=True, 
                        help='List of simulation names')
    parser.add_argument('--save_folder_all', type=str, required=True, 
                        help='Folder to store results.')
    parser.add_argument('--seeds', nargs='+', default=None, 
                        help='Seed folders.')
    parser.add_argument('--n_runs', type=int, required=True, 
                        help='Number of runs')
    # Parse arguments from the command line
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    get_first_last_valid_all_dss(args.paths, 
                                 args.paths_second_dss,
                             args.validation_folders,
                             args.sim_map,
                             args.sim_names,
                             args.save_folder_all,
                             args.seeds,
                             args.n_runs)
    