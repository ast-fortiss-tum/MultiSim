import os
import matplotlib.pyplot as plt
from pathlib import Path
from features import opensbt_feature
from features.feature_map import FeatureMap
from opensbt.quality_indicators.quality import EvaluationResult
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log
from opensbt.visualization.visualization3d import visualize_3d
from config import CONSIDER_HIGH_VAL_OS_PLOT, LAST_ITERATION_ONLY_DEFAULT, OUTPUT_PRECISION, PENALTY_MAX, PENALTY_MIN
from opensbt.visualization.output import write_metric_history

WRITE_ALL_INDIVIDUALS = True
BACKUP_FOLDER =  "backup" + os.sep
METRIC_PLOTS_FOLDER =  "metric_plots" + os.sep

def calculate_fm_sp(result, 
                    save_folder,
                    max_fitness=0, 
                    min_fitness=-3):
    res = result
    problem = res.problem
    hist = res.history
    if hist is not None:
        sp = []
        n_evals = []
        for i, generation in enumerate(hist):
            n_eval = generation.evaluator.n_eval
            all_population_gen = res.archive[0:n_eval]

            save_folder_gen = save_folder + os.sep + f"gen_{i}" + os.sep
            Path(save_folder_gen).mkdir(parents=True, exist_ok=True)

            # get fmap and plot
            map: FeatureMap = opensbt_feature.plot_feature_map_from_pop(all_population_gen,
                                                        save_folder=save_folder_gen,
                                                        max_fitness=max_fitness,
                                                        min_fitness=min_fitness,
                                                        iterations=i)
            n_evals.append(n_eval)
            sp.append(map.get_fm_sparseness())  
        return EvaluationResult("fm_sp", n_evals, sp)
    else:
        return None

def calculate_fm_fail_sp(result, 
                        save_folder,
                        max_fitness=0, 
                        min_fitness=-3):
    res = result
    problem = res.problem
    hist = res.history

    # print("######### calculate_fm_fail_sp ###########")
    # print("res.history", hist)
    
    if hist is not None:
        sp = []
        n_evals = []
        for i, generation in enumerate(hist):
            n_eval = generation.evaluator.n_eval
            all_population_gen = res.archive[0:n_eval]

            save_folder_gen = save_folder + os.sep + f"gen_{i}" + os.sep
            Path(save_folder_gen).mkdir(parents=True, exist_ok=True)

            # get fmap and plot
            map: FeatureMap = opensbt_feature.plot_feature_map_from_pop(all_population_gen,
                                                            save_folder=save_folder_gen,
                                                            max_fitness=max_fitness,
                                                            min_fitness=min_fitness,
                                                            iterations=i)

            n_evals.append(n_eval)
            sp.append(map.get_failure_sparsness())  
        return EvaluationResult("fm_fail_sp", n_evals, sp)
    else:
        return None

def calculate_fm_coverage(result, 
                          save_folder,
                    max_fitness=0, 
                    min_fitness=-3):
    res = result
    problem = res.problem
    hist = res.history
    if hist is not None:
        sp = []
        n_evals = []
        for i, generation in enumerate(hist):
            n_eval = generation.evaluator.n_eval
            all_population_gen = res.archive[0:n_eval]

            save_folder_gen = save_folder + os.sep + f"gen_{i}" + os.sep
            Path(save_folder_gen).mkdir(parents=True, exist_ok=True)

            # get fmap and plot
            map: FeatureMap = opensbt_feature.plot_feature_map_from_pop(all_population_gen,
                                                            save_folder=save_folder_gen,
                                                            max_fitness=max_fitness,
                                                            min_fitness=min_fitness,
                                                            iterations=i)

            n_evals.append(n_eval)
            sp.append(map.get_fm_coverage())  
        return EvaluationResult("fm_cov", n_evals, sp)
    else:
        return None
           
def fmap_failure_sp(res: Result, 
                    save_folder: str, 
                    max_fitness=0, 
                    min_fitness=-3, 
                    filename='fm_fail_sp'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    fmap_plot_folder = save_folder + os.sep + "feature_map"
    Path(fmap_plot_folder).mkdir(parents=True, exist_ok=True)

    eval_result = calculate_fm_fail_sp(res,
                                    fmap_plot_folder,
                                    max_fitness, 
                                    min_fitness)

    n_evals, fm_sp = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)

    write_metric_history(n_evals, fm_sp, filename, save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, fm_sp, color='black', lw=0.7)
    plt.scatter(n_evals, fm_sp, facecolor='none', edgecolor='black', marker='o')
    plt.title("Feature Map Failure Sparseness Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Failure Sparseness")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final IGD value: {fm_sp[-1]}")       

    return n_evals, fm_sp 

def fmap_coverage(res: Result, save_folder: str, 
                                    max_fitness=0, 
                                        min_fitness=-3, 
                                        filename='fm_cov'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)
  
    fmap_plot_folder = save_folder + os.sep + "feature_map"
    Path(fmap_plot_folder).mkdir(parents=True, exist_ok=True)

    eval_result = calculate_fm_coverage(res,
                                    fmap_plot_folder,
                                    max_fitness, 
                                    min_fitness)

    n_evals, fm_sp = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)

    write_metric_history(n_evals, fm_sp, filename, save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, fm_sp, color='black', lw=0.7)
    plt.scatter(n_evals, fm_sp, facecolor='none', edgecolor='black', marker='o')
    plt.title("Feature Map Coverage Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Coverage")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final IGD value: {fm_sp[-1]}")
        
    return n_evals, fm_sp 

def fmap_sparseness(res: Result, save_folder: str, 
                                    max_fitness=0, 
                                        min_fitness=-3, 
                                        filename='fm_sp'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)
    
    fmap_plot_folder = save_folder + os.sep + "feature_map"
    Path(fmap_plot_folder).mkdir(parents=True, exist_ok=True)

    eval_result = calculate_fm_sp(res,
                                    fmap_plot_folder,
                                    max_fitness, 
                                    min_fitness)

    n_evals, fm_sp = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)

    write_metric_history(n_evals, fm_sp, filename, save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, fm_sp, color='black', lw=0.7)
    plt.scatter(n_evals, fm_sp, facecolor='none', edgecolor='black', marker='o')
    plt.title("Feature Map Sparseness Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Sparseness")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final IGD value: {fm_sp[-1]}")

    return n_evals, fm_sp 