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
                                  metric_names =["valid_rate", "n_valid", "fmap_cov", "fmap_s", "fmap_fs",
                                                "avg_euclid"] ):
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
            find_file_contains("../msim-results/analysis/analysis_runs_20-12-2024_BD" , ["bd_u_1"])
        ],
        "UD" : [
            find_file_contains("../msim-results/analysis/analysis_runs_08-01-2025_UD/overview_ud_b" , ["ud_b_1"])
        ],
        "UB" : [
             find_file_contains("../msim-results/analysis/analysis_runs_26-12-2024_BU" , ["bu_d_1"])
        ]
    }
    
    files_dss = {
        "DSS-DB" : [
            find_file_contains("../msim-results/analysis/analysis_runs_20-12-2024_D/overview_qu_dss_db_u/" , ["dss_db_u_1"])
        ],
        "DSS-UD" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-01-2025_U/" , ["ud_b_1"])
        ],
        "DSS-UB" : [
            find_file_contains("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" , ["bu_d_1"]),
            find_file_contains("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , ["bu_d_1"])
        ]
    }
    files_ssim = {
        "SSIM-U" : [
            find_file_contains("../msim-results/analysis/analysis_runs_01-01-2025_U/" , ["u_bd_1"])
        ],
        "SSIM-B" : [
            find_file_contains("../msim-results/analysis/analysis_runs_20-12-2024_B/" , ["b_ud_1"])
        ],
        "SSIM-D" : [
            find_file_contains("../msim-results/analysis/analysis_runs_20-12-2024_D/" , ["d_ub_1"])
        ]
    }
   
    # print(files_msim)
    # print(files_dss)
    # print(files_ssim)

    save_folder = r"../msim-results/analysis/analysis_runs_02-2025/"
    save_folder_overview =  save_folder + os.sep + "overview" + os.sep
    save_folder_stats =  save_folder + os.sep + "stats" + os.sep
    
    write_merge_overview = False
    write_merge_overview_efficiency = False
    write_significance_effect = False
    write_significance_efficiency = True

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
            
    if write_significance_effect:         
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
            with open(os.path.dirname(save_folder) + os.sep  + "metadata.csv", mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["training_data", files_all])
            
    ####################
    ######## efficiency
    if write_significance_efficiency:
        # msims = ["DB", "UD", "UB"]
        # dsims = ["DSS-DB" , "DSS-UD", "DSS-UB"]
        # ssims = ["U", "B", "D"]
        save_folder_overview =  save_folder + os.sep + "overview_efficiency_time" + os.sep 
        save_folder_stats =  save_folder + os.sep + "stats_efficiency_time" + os.sep
        
        if consider_eval_time:
            suffix = "_time"
        else:
            suffix = ""    

        files_msim = {
            "DB" : [
                find_files_with_parent(f"../msim-results/analysis/analysis_5-runs_msim_29-09-24/" ,
                                        "overview_first_last_valid",
                                        "efficiency_bd" + suffix)[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_more_10-12-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_bd" + suffix)[0]
            ]
            # ,
            # "UD" : [
            #     find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_29-09-24/" , 
            #                             "overview_first_last_valid",
            #                             "efficiency_ud" + suffix)[0],
            #     find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_more_10-12-24/" ,    
            #                         "overview_first_last_valid",
            #                             "efficiency_ud" + suffix)[0]
            # ],
            # "UB" : [
            #     find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_29-09-24/" ,
            #                         "overview_first_last_valid",
            #                         "efficiency_ub" + suffix)[0],
            #     find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_more_10-12-24/" ,
            #                             "overview_first_last_valid",
            #                         "efficiency_ub" + suffix)[0]
            # ]
        }

        files_dss = {
            "DSS-DB" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" ,
                                        "overview_first_last_valid",
                                        "efficiency_dss_db" + suffix)[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_dss_db" + suffix)[0]
            ],
            "DSS-UD" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" ,
                                        "overview_first_last_valid",
                                        "efficiency_dss_ud" + suffix)[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_dss_ud" + suffix)[0]
            ],
            "DSS-UB" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" ,
                                        "overview_first_last_valid",
                                        "efficiency_dss_bu" + suffix)[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_dss_bu" + suffix)[0]
            ]
        }
        print(files_dss)
        # files_dss = {
        #     "DSS-DB" : [
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" , ["bd_u_1"]),
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , ["db_u_1"])
        #     ],
        #     "DSS-UD" : [
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" , ["du_b_1"]),
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , ["ud_b_1"])
        #     ],
        #     "DSS-UB" : [
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_all_10-12-2024/" , ["bu_d_1"]),
        #         find_files_with_parent("../msim-results/analysis/analysis_5-runs_dss_more_10-12-24/" , ["bu_d_1"])
        #     ]
        # }
        files_ssim = {
            "SSIM-U" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_29-09-24/" ,
                                        "overview_first_last_valid",
                                        "efficiency_u")[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_ssim_more_10-12-24/" ,
                                    "overview_first_last_valid",
                                    "efficiency_u")[0]
            ],
            "SSIM-B" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_29-09-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_b")[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_ssim_more_10-12-24/" , 
                                        "overview_first_last_valid",
                                        "efficiency_b")[0]
                                        ],
            "SSIM-D" : [
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_msim_29-09-24/" ,
                                    "overview_first_last_valid",
                                        "efficiency_d")[0],
                find_files_with_parent("../msim-results/analysis/analysis_5-runs_ssim_more_10-12-24/" ,
                                    "overview_first_last_valid",
                                        "efficiency_d")[0],
            ]
        }

        files_all = {**files_msim, **files_ssim, **files_dss}

        if write_merge_overview_efficiency:
            # merge runs
            for key in files_all:
                print(key)
                merge_overviews(files_all[key][0],files_all[key][1],
                                output_folder=save_folder_overview,
                                name=f"overview_efficiency_first_last_valid_1.0_{key}")
                
        for msim in list(files_msim.keys()):
            files1_csv = files_msim[msim]
            print(msim)
            
            for other in list(files_ssim.keys()):
                files2_csv = files_ssim[other]
                print(other)

                calc_significance_from_overview_all(files1_csv=files1_csv,
                                            files2_csv=files2_csv,
                                            name1 = msim,
                                            name2 = other,
                                            save_folder=save_folder_stats,
                                            metric_names=["first_valid", "last_valid"]
                                            )
            for other in list(files_dss.keys()):
                files2_csv = files_dss[other]
                print(other)

                calc_significance_from_overview_all(files1_csv=files1_csv,
                                            files2_csv=files2_csv,
                                            name1 = msim,
                                            name2 = other,
                                            save_folder=save_folder_stats,
                                            metric_names=["first_valid", "last_valid"]
                                            )
        # write training metadata
        with open(os.path.dirname(save_folder) + os.sep  + "metadata_efficiency.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["training_data", files_all])
        
