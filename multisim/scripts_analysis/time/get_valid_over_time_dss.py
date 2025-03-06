import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from opensbt.utils.duplicates import duplicate_free
from opensbt.utils.time_utils import convert_pymoo_time_to_seconds
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

import copy
from datetime import datetime

from config import MAX_EVALS_FIRST_VALID_MAP

from scripts_analysis.time.calc_combined import calc_average_over_runs, \
                                                write_metric_data_to_csv, \
                                                retrieve_metric_data_from_csv, \
                                                make_comparison_plot

# N_FUNC_EVALS_LIM = 800
N_FITTING_POINTS = 10

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
                         max_evals = 500
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

###########################

# paths = [
#         "../msim-results/analysis/analysis_5-runs_12-12-2024_BU",
#         "../msim-results/analysis/analysis_5-runs_11-11-2024_DB/",
#         "../msim-results/analysis/analysis_5-runs_29-11-2024_UD/"

#     ]

# validation_folders = [
#     "validation_count-3_combined_d",
#     "validation_combined_count-3_u",
#     "validation_combined_count-3_b"
# ]

# sim_map = [
#         "msim",
#         "msim",
#         "msim"
# ]

# sim_names = [
#     "ub",
#     "bd",
#     "ud"
# ]

# save_folder_all = r"..\msim-results\analysis\analysis_5-runs_msim_more_10-12-24\\"

def get_valid_over_time_dss(paths,
                            paths_second_dss,
                            validation_folders,
                            sim_map,
                            sim_names,
                            save_folder_all,
                            seeds = None,
                            n_runs = None,
                            use_percentage = True):
    
    date = datetime.now().strftime(f"%d-%m-%Y_%H-%M-%S")

    n_max_evals = -1
    
    results_all_sims = {}
    
    save_folder_efficiency = f"{save_folder_all}/efficiency_det_dss/{date}/"

    for j, folder_path in enumerate(paths):
        
        # Folder containing the run files
        # folder_path = "../results/analysis/analysis-5-runs_19-09-2024_BU/"
        validation_folder = validation_folders[j]
        
        threshold=1.0
        validation_file = f"validation_{threshold}"
        testcases_file = "all_testcases"

        sim = sim_map[j]
    
        print(sim_names[j])
        save_folder = os.path.join(f"{save_folder_efficiency}/{sim_names[j]}/")
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        #######################

        results_all = []

        paths_valid_tests = []
      
        if seeds is None:
            selector =  range(0,n_runs)
        else:
            selector =  seeds

        for i in selector:
            # HACK for now
            # if j == 0 and i == 2:
            #     continue
            path = os.path.join(folder_path,f"run_{i}/{sim}")
            print(path)
            path_all_tests_A = find_files_with_parent(path + os.sep, 
                                        prefix=testcases_file,
                                        parent_folder="myfolder")[0]
            
            print("path_all_tests", path_all_tests_A)
            
            path_second_dss = os.path.join(paths_second_dss[j],f"run_{i}/{sim}")

            path_all_tests_B = find_files_with_parent(path_second_dss + os.sep, 
                                        prefix=testcases_file,
                                        parent_folder="myfolder")[0]
            
      
            valid_tests = find_files_with_parent(path + os.sep, 
                                        prefix=validation_file,
                                        parent_folder=validation_folder)[0]
            print("valid_tests", valid_tests)

            def get_valid_in_buckets_dss(path_all_tests_A, 
                                         path_all_tests_B,
                                     valid_tests, 
                                     bucket_size):
                
                P_A = read_pf_single(path_all_tests_A).get("X")
                P_B = read_pf_single(path_all_tests_B).get("X")
                
                n_evals_A = len(P_A)
                n_evals_B = len(P_B)
                
                P = PopulationExtended.merge(P_A, P_B)

                # read in all valid
                with open(valid_tests, 'r') as file:
                    V = json.load(file)["angles_valid"]
                
                # sometimes there is 0.00000000000001 added
                #V =  [np.asarray([round(v,2) for v in test]) for test in V]

                print("len valid initial:", len(V))
                print("len PA", len(P_A))
                print("len PB", len(P_B))

                #print("V", V)
                # individuals = []
                # for v in V:
                #     ind = IndividualSimulated()
                #     ind.set("X", np.asarray(v))
                #     ind.set("CB", True)
                # individuals.append(ind)
                # V_clean = duplicate_free(PopulationExtended(individuals=individuals)).get("X")
                
                # print("len valid after dup_free:", len(V_clean))

                n_valid_per_bucket_A = []
                t_valid_per_bucket_A = []
                
                buckets_A = [P_A[i:i + bucket_size] for i in range(0, len(P_A), bucket_size)]
                V_temp = copy.deepcopy(V)
                for bucket in buckets_A:
                    n_valid = 0
                    t_valid = []
                    for x in bucket:
                        for x_valid in V_temp:
                            if np.array_equal(x, np.asarray([round(v,2) for v in x_valid])):
                                n_valid += 1
                                t_valid_per_bucket_A.append(x)
                                # remove to save later costs
                                #np.delete(V_temp, index, axis=0)
                                V_temp.remove(x_valid)
                                
                    n_valid_per_bucket_A.append(n_valid)
                    t_valid_per_bucket_A.append(t_valid)
                    
                bucket_sizes_A = [len(bucket) for bucket in buckets_A]
                
                ##################
                n_valid_per_bucket_B = []
                t_valid_per_bucket_B = []
                
                buckets_B = [P_B[i:i + bucket_size] for i in range(0, len(P_B), bucket_size)]
                V_temp = copy.deepcopy(V)
                for bucket in buckets_B:
                    n_valid = 0
                    t_valid = []
                    for x in bucket:
                        for x_valid in V_temp:
                            if np.array_equal(x, np.asarray([round(v,2) for v in x_valid])):
                                n_valid += 1
                                t_valid_per_bucket_B.append(x)
                                # remove to save later costs
                                #np.delete(V_temp, index, axis=0)
                                V_temp.remove(x_valid)
                                
                    n_valid_per_bucket_B.append(n_valid)
                    t_valid_per_bucket_B.append(t_valid)
                    
                bucket_sizes_B = [len(bucket) for bucket in buckets_B]
                
                ############## buckets combined 
                
                bucket_sizes = bucket_sizes_A + bucket_sizes_B
                n_valid_per_bucket = n_valid_per_bucket_A + n_valid_per_bucket_B
                t_valid_per_bucket = t_valid_per_bucket_A + t_valid_per_bucket_B
                buckets = buckets_A + buckets_B
                
                print(n_valid_per_bucket)
                ##############################
                # assert len(V) == np.sum(n_valid_per_bucket), f"V is {len(V)}, n_valid_per_bucket is {np.sum(n_valid_per_bucket)}"  
                # assert np.ceil(len(P) / bucket_size) == len(buckets)
                
                return zip(bucket_sizes, n_valid_per_bucket), t_valid_per_bucket, n_evals_A, n_evals_B
            
            valid_over_buckets, t_valid_per_bucket, n_evals_A, n_evals_B = get_valid_in_buckets_dss(path_all_tests_A, 
                                                             path_all_tests_B,
                                                            valid_tests, 
                                                            bucket_size = 20)
            print("valid over buckets:", valid_over_buckets)    
            valid_agg_buckets = []
            evals = 0
            valid_all = 0
            
            for key,value in valid_over_buckets:
                evals += key
                valid_all += value
                valid_agg_buckets.append([evals, valid_all])
            
            new_valid_agg_buckets = []
            # adapt time of encountering the valid test after sequentialization
            for evals,value in valid_agg_buckets:
                if evals <= n_evals_A:
                    evals_new = n_evals_A + evals
                else:
                    evals_new = n_evals_A + n_evals_B + evals
                
                new_valid_agg_buckets.append([evals_new, value])
                
            valid_agg_buckets = new_valid_agg_buckets 
            print(valid_agg_buckets)

            if use_percentage:
                total_tests = 2*(n_evals_A + n_evals_B)
                print(valid_agg_buckets)
                percentage_valid_agg_buckets = [[round(subarray[0]/total_tests,2), subarray[1]] for subarray in valid_agg_buckets]
                valid_agg_buckets = percentage_valid_agg_buckets

            # we need to compute the max number of evaluations for comparison and creation of the plot
            if use_percentage:
                n_max_evals = 1
            else:
                if n_max_evals < valid_agg_buckets[-1][0]:
                    n_max_evals = valid_agg_buckets[-1][0]

            n_evals = [sub_array[0] for sub_array in valid_agg_buckets]
            hist_valid_tests =  [sub_array[1] for sub_array in valid_agg_buckets]

            # write down
            def write_valid_over_time_run(n_evals, hist_valid_tests, metric_name, save_folder, do_plot = False):
                history_folder = save_folder 
                Path(history_folder).mkdir(parents=True, exist_ok=True)
                with open(history_folder+ os.sep + metric_name + '.csv', 'w', encoding='UTF8', newline='') as f:
                    write_to = csv.writer(f)
                    header = ['n_evals', metric_name]
                    write_to.writerow(header)
                    for i,_ in enumerate(n_evals):
                        write_to.writerow([n_evals[i], hist_valid_tests[i]])
                    f.close()
                    
                if do_plot:
                    import matplotlib.pyplot as plt
                    f = plt.figure()
                    plt.plot(n_evals, hist_valid_tests, color='black', lw=0.7)
                    plt.scatter(n_evals, hist_valid_tests, facecolor="none", edgecolor='black', marker="o")
                    plt.title(f"{metric_name} analysis over time")
                    plt.xlabel("Simulations")
                    plt.ylabel(f"{metric_name}")
                    plt.savefig(history_folder + os.sep + metric_name + ".png")
                    plt.close()
                    plt.clf()
                    plt.close(f)
        
            write_valid_over_time_run(n_evals, 
                                      hist_valid_tests,
                                      f"n_valid_run_{i}", 
                                      save_folder = save_folder + os.sep + "history")

            #sim_name = sim_names[j]

            # max_evals = MAX_EVALS_FIRST_VALID_MAP[sim_name]  \
            #                 if sim_name in MAX_EVALS_FIRST_VALID_MAP else \
            #                         MAX_EVALS_FIRST_VALID_MAP[sim_name[::-1]]
            
            # results_run = get_first_last_valid(path_all_tests,
            #                     valid_tests,
            #                     max_evals
            #                     )
            
            # raw number is only roads generated

            results_all.append(valid_agg_buckets)
        
        # store results for this sim combination
        results_all_sims.update(
            {
                sim_names[j].upper()  :  results_all
            }
        )
        
        # # Open a CSV file to write the data
        # with open(save_folder + os.sep + f'{"n_valid_results_"}{sim_names[j]}.csv', 'w', encoding='UTF8', newline='') as file:
        #     writer = csv.writer(file)

        #     # Write the header (optional, you can define your own header)
        #     writer.writerow(['x_plot', 'y_mean', 'y_error'])

        #     # Iterate over each entry in the plot_array
        #     for data in avg_valid_runs:
        #         x_values = data[0]  # x_plot
        #         y_values = data[1]  # y_mean
        #         y_errors = data[2]  # y_error
                
        #         # Write each row of data to the CSV file
        #         for x, y, error in zip(x_values, y_values, y_errors):
        #             writer.writerow([x, y, error])
                
        # print("evaluation stored in:", save_folder + os.sep + f'{"n_valid_results_"}{sim_names[j]}.csv')
        
    avg_valid_runs_all_sims = calc_average_over_runs(metric_name="n_valid",
                        metric_values_array_algo = results_all_sims,
                        n_func_evals_lim = n_max_evals,
                        n_fitting_points = N_FITTING_POINTS,
                        save_folder = save_folder_efficiency + os.sep +  "n_valid" + os.sep,
                        metric_name_label = "n_valid",
                        is_percentage=True)
    
    paths_metric_data = write_metric_data_to_csv(save_folder_efficiency, 
                             metric_names = ["n_valid"],
                             algo_names = sim_names,
                             plot_array_metric = avg_valid_runs_all_sims,
                             suffix="")
    print(paths_metric_data)
    metric_data_loaded = retrieve_metric_data_from_csv(paths_metric_data,
                                                       n_algos=len(sim_names))
    
    make_comparison_plot(max_evaluations = 1,  # percentage
                         save_folder = save_folder_efficiency, 
                         subplot_metrics = metric_data_loaded,
                         subplot_names = ["Number Valid Failures"], 
                         algo_names = sim_names, 
                         distance_tick =0.1, 
                         suffix="",
                         is_percentage=True,
                          shift_error=True,
                          plot_error=False)
    
    make_comparison_plot(max_evaluations = 1,  # percentage
                         save_folder = save_folder_efficiency, 
                         subplot_metrics = metric_data_loaded,
                         subplot_names = ["Number Valid Failures"], 
                         algo_names = sim_names, 
                         distance_tick =0.1, 
                         suffix="_error",
                         is_percentage=True,
                          shift_error=True,
                          plot_error=True)

    print("avg_valid_runs_all_sims:", avg_valid_runs_all_sims)
    
    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data = [
        ["timestamp", str(timestamp)],
        ["paths:", str(paths)],
        ["runs:", str(selector)]
    ]
    # Write data to CSV file
    with open(save_folder_efficiency + os.sep + "metadata.csv", mode='w', newline='') as file:
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
    get_valid_over_time_dss(args.paths, 
                            args.paths_second_dss,
                             args.validation_folders,
                             args.sim_map,
                             args.sim_names,
                             args.save_folder_all,
                             args.seeds,
                             args.n_runs)