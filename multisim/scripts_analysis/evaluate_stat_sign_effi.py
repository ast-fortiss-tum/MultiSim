import csv
import itertools as it

from bisect import bisect_left
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss

from pandas import Categorical
from opensbt.statistical_tests.wilcoxon import run_wilcoxon_and_delaney
from datetime import datetime
import os

from opensbt.utils.files import find_file_contains, find_files_with_parent

def merge_overviews(file1,
                    file2, 
                    output_folder = "./merged_overview",
                    name = None,
                    batch_size = 20):
    
    if name is None:
        name = os.path.splitext(os.path.basename(file1))[0]
    
    print("name is", name)

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Remove the "Avg" and "Std" rows for processing
    df1_no_stats = df1[df1['run'].str.isdigit()]
    df2_no_stats = df2[df2['run'].str.isdigit()]

    # Append the datasets
    combined_df = pd.concat([df1_no_stats, df2_no_stats], ignore_index=True)

    # Renumber the "run" column sequentially
    combined_df['run'] = range(1, len(combined_df) + 1)

    # Recalculate Avg and Std
    avg_row = combined_df.mean(numeric_only=True).to_dict()
    std_row = combined_df.std(numeric_only=True).to_dict()

    # Cut the results to 6 digits after the decimal point
    avg_row = {key: round(value, 6) if isinstance(value, float) else value for key, value in avg_row.items()}
    std_row = {key: round(value, 6) if isinstance(value, float) else value for key, value in std_row.items()}

    # Add the Avg and Std rows
    avg_row['run'] = 'avg'
    std_row['run'] = 'std'

    # Append Avg and Std to the combined dataframe
    stats_df = pd.DataFrame([avg_row, std_row])
    final_df = pd.concat([combined_df, stats_df], ignore_index=True)
    
    avg_row = final_df[final_df['run'] == 'avg']
    std_row = final_df[final_df['run'] == 'std']

    fv_iter = avg_row.iloc[0, 1:].tolist()[0]
    lv_iter = avg_row.iloc[0, 1:].tolist()[1]

    fv_std_iter = std_row.iloc[0, 1:].tolist()[0]
    lv_std_iter = std_row.iloc[0, 1:].tolist()[1]

    row_batch = [
        {
         'run': 'avg_iter', 
         'first_valid': fv_iter/batch_size, 
         'last_valid': lv_iter/batch_size,
         'test_first': -1, 
         'test_last': -1
        },
         {
         'run': 'std_iter', 
         'first_valid': fv_std_iter/batch_size, 
         'last_valid': lv_std_iter/batch_size,
         'test_first': -1, 
         'test_last': -1
        },
    ]

    custom_rows_df = pd.DataFrame(row_batch)
    final_df = pd.concat([final_df, custom_rows_df], ignore_index=True)
    
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Save the result to a new CSV
    final_df.to_csv(f'{output_folder}/{name}.csv', index=False)

def significance_from_overview(files1_csv, files2_csv, metric):
    data1 =  []
    for file in files1_csv:
        data = pd.read_csv(file)
        data = data[~data["run"].isin(["avg", "std"])]
        data1 += data[metric].values.tolist()
    data2 =  []
    for file in files2_csv:
        data = pd.read_csv(file)
        data = data[~data["run"].isin(["avg", "std"])]
        data2 += data[metric].values.tolist()
    print("data1", data1)
    print("data2", data2)

    # print(VD_A(data1, data2))
    return run_wilcoxon_and_delaney(data1,data2)

def calc_significance_from_overview_all(files1_csv, 
                                  files2_csv,
                                  name1,
                                  name2, 
                                  save_folder,
                                  metric_names =["first_valid", "last_valid"]):
    results = {
        f"{name1}_-_{name2}": metric_names,
        "p_value": [],
        "effect": []
    }

    for metric in metric_names:
        p_value, effect = significance_from_overview(files1_csv, files2_csv, metric)
        print(f"{metric}:")
        print(f"P-Value is: ", p_value)
        print(f"Delaney's effect size is:",effect)
        print("\n")
        results["effect"].append(effect)
        results["p_value"].append(p_value)
 
    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    Path(save_folder).mkdir(exist_ok=True, parents=True)
    
    # date = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    # Save the DataFrame to a CSV file
    results_df.to_csv(save_folder + f"/significance_{name1}-vs-{name2}.csv", index=False)

if __name__ == '__main__':
    # msims = ["DB", "UD", "UB"]
    # dsims = ["DSS-DB" , "DSS-UD", "DSS-UB"]
    # ssims = ["U", "B", "D"]
    files_msim = {
        "DB" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/bd" , ["rel_1.0"])
        ],
        "UD" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/ud" , ["rel_1.0"])
        ],
        "UB" : [
             find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/bu" , ["rel_1.0"])
        ]
    }
    
    files_dss = {
        "DSS-DB" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency_dss/db" , ["rel_1.0"])
        ],
        "DSS-UD" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency_dss/ud" , ["rel_1.0"])
        ],
        "DSS-UB" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency_dss/bu" , ["rel_1.0"])
        ]
    }
    files_ssim = {
        "SSIM-U" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/u" , ["rel_1.0"])
        ],
        "SSIM-B" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/b" , ["rel_1.0"])
        ],
        "SSIM-D" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-2025/efficiency/d" , ["rel_1.0"])
        ]
    }
   
    # print(files_msim)
    # print(files_dss)
    # print(files_ssim)

    save_folder = r"../msim-results/analysis/analysis_runs_01-2025/"
    save_folder_overview =  save_folder + os.sep + "overview_eff" + os.sep
    save_folder_stats =  save_folder + os.sep + "stats_eff" + os.sep
    
    write_merge_overview = False
    write_merge_overview_efficiency = False

    consider_eval_time = True

    files_all = {**files_dss, **files_msim, **files_ssim}

    if write_merge_overview:
        # merge runs
        print(files_all)
        for key in files_all:
            print(key)
            merge_overviews(files_all[key][0],files_all[key][1],
                            output_folder=save_folder_overview)
            
    if write_merge_overview_efficiency:
        # merge runs
        print(files_all)
        for key in files_all:
            print(key)
            merge_overviews(files_all[key][0],files_all[key][1],
                            output_folder=save_folder_overview)
            
    for msim in list(files_msim.keys()):

        files1_csv = files_msim[msim]
        print(msim)
        

        # merge all runs for analysis
        print("dss keys", list(files_msim.keys()))

        print(files_dss)
        for other in list(files_dss.keys()):
            files2_csv = files_dss[other]

            #print("file",files2_csv[0])
            #print("file",files2_csv[1])
            print(files1_csv)
            print(files2_csv)

            calc_significance_from_overview_all(files1_csv=files1_csv,
                                        files2_csv=files2_csv,
                                        name1 = msim,
                                        name2 = other,
                                        save_folder=save_folder_stats
                                        )
        for other in list(files_ssim.keys()):
            files2_csv = files_ssim[other]
            print(other)

            calc_significance_from_overview_all(files1_csv=files1_csv,
                                        files2_csv=files2_csv,
                                        name1 = msim,
                                        name2 = other,
                                        save_folder=save_folder_stats
                                        )
        for other in list(files_msim.keys()):
            if msim == other:
                continue
            
            files2_csv = files_msim[other]
            print(other)

            calc_significance_from_overview_all(files1_csv=files1_csv,
                                        files2_csv=files2_csv,
                                        name1 = msim,
                                        name2 = other,
                                        save_folder=save_folder_stats
                                        )
        # write training metadata
        with open(os.path.dirname(save_folder) + os.sep  + "metadata_efficiency.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["training_data", files_all])
   