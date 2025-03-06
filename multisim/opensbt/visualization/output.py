import copy
import csv
import json
import math
import os


import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.quality_indicators.metrics.igde import IGDE
from opensbt.visualization import plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.quality_indicators.quality import EvaluationResult, Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free, duplicate_free_with_index
import logging as log
from opensbt.visualization.visualization3d import visualize_3d
import shutil
import config
import uuid
from config import CONSIDER_HIGH_VAL_OS_PLOT, LAST_ITERATION_ONLY_DEFAULT, OUTPUT_PRECISION, PENALTY_MAX, PENALTY_MIN

WRITE_ALL_INDIVIDUALS = True
BACKUP_FOLDER =  "backup" + os.sep
METRIC_PLOTS_FOLDER =  "metric_plots" + os.sep

def is_valid_dir_path(path):
    if path[0:2] == "C:" or path[0:2] == "c/" or path[0:2] == r"c\\" or "home" in path or "./" in path :
        return True
    else:
        return False

def delete_files_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            log.info('Failed to delete %s. Reason: %s' % (file_path, e))

def create_save_folder(problem: Problem, 
                       results_folder: str, 
                       algorithm_name: str, 
                       is_experimental = False,
                       folder_name = None): # define optional the folder name instead using time
    # log.info("results_folder", results_folder)
    # log.info("is_experimental", is_experimental)
    # log.info("folder_name", folder_name)
    # input()
    problem_name = problem.problem_name
    # algorithm_name = type(res.algorithm).__name__
    # if results folder is already a valid folder, do not create it in parent, use it relative
    
    if results_folder is None:
        results_folder = ""
    
    # HACK Temporary disabled
    # if results_folder is not None and os.path.isdir(results_folder):
    #     save_folder = results_folder 
    #     #+ problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
    #     #   "%d-%m-%Y_%H-%M-%S") + os.sep
    elif is_experimental:
        if is_valid_dir_path(results_folder):
            save_folder = results_folder + problem_name + os.sep + algorithm_name + os.sep + "temp" + os.sep
        else:
            save_folder = str(
            os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + "temp" + os.sep
    else:
    
        if is_valid_dir_path(results_folder):
            if folder_name is None:
                save_folder = results_folder + problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S") + os.sep
            else:
                save_folder = results_folder + problem_name + os.sep + algorithm_name + os.sep + folder_name + os.sep
        else:
            if folder_name is None:
                save_folder = str(
                os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S") + os.sep
            else:
                save_folder = str(os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + folder_name + os.sep
    # delete existing folder if exists
    # if os.path.isdir(save_folder):
    #     delete_files_folder(save_folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    log.info(f"save_folder created: {save_folder}")
    return save_folder

def get_pc_name():
    import os
    import socket

    # Get the current username
    username = os.getlogin()

    # Get the hostname
    hostname = socket.gethostname()
    return f"{username}@{hostname}"

def write_calculation_properties(res: Result, save_folder: str, algorithm_name: str, algorithm_parameters: Dict, **kwargs):
    problem = res.problem
    # algorithm_name = type(res.algorithm).__name__
    is_simulation = problem.is_simulation()
    
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
    uid = str(uuid.uuid4())

    with open(save_folder + 'calculation_properties.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Id', uid])
        write_to.writerow(['Timestamp', date_time])
        write_to.writerow(['PC', get_pc_name()])
        write_to.writerow(['Problem', problem.problem_name])
        write_to.writerow(['Algorithm', algorithm_name])
        write_to.writerow(['Search variables', problem.design_names])        
        write_to.writerow(['Search space', [v for v in zip(problem.xl,problem.xu)]])
        
        if is_simulation:
            write_to.writerow(['Fitness function', str(problem.fitness_function.name)])
        else:
            write_to.writerow(['Fitness function', "<No name available>"])

        if hasattr(res.problem, "migrate_function"):
            write_to.writerow(['Migrate function', str(problem.migrate_function.__class__.__name__)])

        write_to.writerow(['Critical function', str(problem.critical_function.name)])
        # write_to.writerow(['Number of maximal tree generations', str(max_tree_iterations)])
        write_to.writerow(['Search time', str("%.2f" % res.exec_time + " sec")])

        for item,value in algorithm_parameters.items():
            write_to.writerow([item, value])

        _additional_descritption(res, save_folder, algorithm_name, **kwargs)

        f.close()

    _calc_properties(res, save_folder, algorithm_name, **kwargs)


def _calc_properties(res, save_folder, algorithm_name, **kwargs):
    pass

def _additional_descritption(res, save_folder, algorithm_name, **kwargs):
    pass

'''Output of the simulation data for all solutions (for the moment only partial data)'''
def write_simulation_output(res: Result, 
                            save_folder: str, 
                            mode = "all", 
                            dup_free = True, 
                            limited = True,
                            write_summary = True,
                            seperate_folder = False):
    problem = res.problem
    if not problem.is_simulation():
        return
    
    max = config.MAX_NUM_GIFS
    inds = get_pop_using_mode(res=res, mode=mode)

    if dup_free and limited:
        all_population = duplicate_free(inds)[:max]
    elif dup_free:
        all_population = duplicate_free(inds)
    else:
        all_population = inds

    if len(all_population) == 0:
        return
    
    if isinstance(all_population.get("SO")[0], SimulationOutput):
        # Single Sim Problem
        save_folder_simout = save_folder + os.sep + "simout" + os.sep
        Path(save_folder_simout).mkdir(parents=True, exist_ok=True)
        write_simout(save_folder_simout, 
                    all_population.get("SO"), 
                    index=0,
                    write_summary = write_summary)

    else:
        # Multi Sim Problem
        for i in range(len(all_population.get("SO")[0])):
            if seperate_folder:
                save_folder_simout = save_folder + os.sep + "simout" + os.sep + f"simulator_{i}" + os.sep
            else:
                save_folder_simout = save_folder + os.sep + "simout" + os.sep
            Path(save_folder_simout).mkdir(parents=True, exist_ok=True)
            
            write_simout(save_folder_simout, 
                        all_population.get("SO")[:,i], 
                        index=i,
                        write_summary = write_summary)
            
def write_simout(path, simout_pop, index=None, write_summary = True):
    for i, simout in enumerate(simout_pop):
        simout_dumped = simout.to_json()
        with open(path + os.sep + f'simout'f'_{i}{f"_S{index}" if index is not None else ""}.json', 'w') as f:
            f.write(simout_dumped)

    if write_summary:
        with open(path + f'simulation_output_S{index}.csv','w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            header = ['Index']
            other_params = simout_pop[0].otherParams

            # write header
            for item, value in other_params.items():
                if isinstance(value,float) or isinstance(value,int) or isinstance(value,bool):
                    header.append(item)
            write_to.writerow(header)

            # write values
            for index in range(len(simout_pop)):
                row = [index]
                other_params = simout_pop[index].otherParams
                for item, value in other_params.items():
                    if isinstance(value,float):
                        row.extend(["%.2f" % value])
                    if isinstance(value,int) or isinstance(value,bool):
                        row.extend([value])
                write_to.writerow(row)
            f.close()

def digd_analysis(res: Result, save_folder: str, input_crit=None, filename='digd'):
    # log.info("------ Performing igd analysis ------")

    eval_result = Quality.calculate_digd(res,input_crit=input_crit)
    if eval_result is None:
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'digd_all',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Design Space Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("dIGD")
    # plt.yscale("log")
    plt.savefig(save_folder + METRIC_PLOTS_FOLDER +  filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final dIGD value: {gd[-1]}")

def igde_analysis_hitherto(res: Result, save_folder: str, reference_set=None, n_evals_by_axis=None):
    log.info("------ Performing IGDE analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_igde(res, reference_set=reference_set, n_evals_by_axis=n_evals_by_axis)

    if eval_result is None:
        log.info("No IDGE values computed")
        return
    
    n_evals, igde = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, igde, 'igde',save_folder)

    f = plt.figure()
    plt.plot(n_evals, igde, color='black', lw=0.7)
    plt.scatter(n_evals, igde, facecolor="none", edgecolor='black', marker="o")
    plt.title("Coverage Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel(COVERAGE_METRIC_NAME)
    plt.savefig(save_folder_plot + COVERAGE_METRIC_NAME.lower() + '_global.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final {COVERAGE_METRIC_NAME} value: {igde[-1]}")

def gd_analysis(res: Result, save_folder: str, input_pf=None, filename='gd', mode='default', critical_only = False):
    log.info("------ Performing gd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_gd(res, input_pf=input_pf, critical_only=critical_only, mode=mode)
    if eval_result is None:
        log.info("No GD values computed")
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'gd_all' + '_' + mode,save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("GD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '_' + mode + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final GD value: {gd[-1]}")


def gd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='gd_global', mode='default'):
    log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_gd_hitherto(res, input_pf=input_pf, mode=mode)
    if eval_result is None:
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'gd_global' + '_' + mode,save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("GD")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + filename + '_' + mode + '.png')
    plt.clf()
    plt.close(f)

def igd_analysis(res: Result, save_folder: str, input_pf=None, filename='igd'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_igd(res, input_pf=input_pf)
    if eval_result is None:
        return

    n_evals, igd = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, igd,'igd_all',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, igd, color='black', lw=0.7)
    plt.scatter(n_evals, igd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final IGD value: {igd[-1]}")

def igd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='igd_global'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_igd_hitherto(res, input_pf=input_pf)
    if eval_result is None:
        return

    n_evals, igd = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, igd,'igd_global',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, igd, color='black', lw=0.7)
    plt.scatter(n_evals, igd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    # plt.yscale("log")
    plt.savefig(save_folder_plot + METRIC_PLOTS_FOLDER + filename + '.png')
    plt.clf()
    plt.close(f)


def write_metric_history(n_evals, hist_F, metric_name, save_folder):
    history_folder = save_folder + "history" + os.sep
    Path(history_folder).mkdir(parents=True, exist_ok=True)
    with open(history_folder+ '' + metric_name + '.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['n_evals', metric_name]
        write_to.writerow(header)
        for i,val in enumerate(n_evals):
            write_to.writerow([n_evals[i], hist_F[i]])
        f.close()

def hypervolume_analysis(res, save_folder, critical_only = False, ref_point_hv = None, relative = False):
    # log.info("------ Performing hv analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_hv_hitherto(res, critical_only=critical_only, 
                                                ref_point = ref_point_hv)
 
    if eval_result is None:
        return
    n_evals, hv = eval_result.steps, eval_result.values

    if relative:
        max_evals = len(res.archive)
        n_evals = [round(eval/max_evals, 3) for eval in n_evals]
        eval_result = EvaluationResult(eval_result.name, n_evals, eval_result.values)
        
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv, 'hv_all', save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder_plot + 'hypervolume.png')

    # output to console
    log.info(f"Final HV value: {hv[-1]}")

def hypervolume_analysis_local(res, save_folder):
    log.info("------ Performing hv analysis ------")
    
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_hv(res)

    if eval_result is None:
        return

    n_evals, hv = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv,'hv_local_all',save_folder)    

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder_plot + 'hypervolume_local.png')


def spread_analysis(res, save_folder):
    # log.info("------ Performing sp analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_sp(res)
    if eval_result is None:
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp',save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")
    plt.savefig(save_folder_plot + 'spread.png')

    # output to console
    log.info(f"Final SP value: {uniformity[-1]}")

def spread_analysis_hitherto(res, save_folder, hitherto = False):
    log.info("------ Performing sp analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_sp_hitherto(res)
    if eval_result is None:
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp_global',save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")

    plt.savefig(save_folder_plot  + 'spread_global.png')

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=min(width, height),
                             height=min(width, height))
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def create_markers(fail_name = "Critical", not_fail_name = "Not critical"):
    patch_not_critical_region = mpatches.Patch(color=color_not_critical, label=f'{not_fail_name} regions',
                                               alpha=0.05)
    patch_critical_region = mpatches.Patch(color=color_critical, label=f'{fail_name} regions', alpha=0.05)

    circle_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                      edgecolor=color_critical, linewidth=1, label=f'{fail_name} testcases')

    circle_not_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                          edgecolor=color_not_critical, linewidth=1, label=f'{not_fail_name} testcases')

    circle_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_optimal,
                                     edgecolor='none', linewidth=1, label='Optimal testcases')

    circle_not_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_not_optimal,
                                         edgecolor='none', linewidth=1, label='Not optimal testcases')

    line_pareto = Line2D([0], [0], label='Pareto front', color='blue')

    marker_list = [patch_not_critical_region, patch_critical_region, circle_critical, circle_not_critical,
                   circle_optimal, circle_not_optimal, line_pareto]

    return marker_list

def backup_object(object, save_folder, name):
    save_folder_object = save_folder + BACKUP_FOLDER
    Path(save_folder_object).mkdir(parents=True, exist_ok=True)   

    import dill
    with open(save_folder_object + os.sep + name, "wb") as f:
        dill.dump(object, f)

def write_summary_results(res, save_folder):

    all_population = res.obtain_archive()
    
    best_population = get_nondominated_population(all_population)

    critical_best,_ = best_population.divide_critical_non_critical()
    critical_all,_ = all_population.divide_critical_non_critical()
    
    n_crit_all_dup_free = len(duplicate_free(critical_all))
    n_all_dup_free = len(duplicate_free(all_population))
    n_crit_best_dup_free = len(duplicate_free(critical_best))
    n_best_dup_free = len(duplicate_free(best_population))
    dup_free = duplicate_free(all_population)

    # write down when first critical solutions found + which fitness value it has
    iter_crit, inds_critical = res.get_first_critical()

    avg_fit_all = np.sum(dup_free.get("F"),axis=0) / len(dup_free.get("F"))
    min_fit_all = np.min(dup_free.get("F"),axis=0)
    max_fit_all = np.max(dup_free.get("F"),axis=0)

    '''Output of summery of the performance'''
    with open(save_folder + 'summary_results.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Number Critical Scenarios', len(critical_all)])
        write_to.writerow(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])

        write_to.writerow(['Number All Scenarios', len(all_population)])
        write_to.writerow(['Number All Scenarios (duplicate free)', n_all_dup_free])

        write_to.writerow(['Number Best Critical Scenarios', len(critical_best)])
        write_to.writerow(['Number Best Critical Scenarios (duplicate free)', n_crit_best_dup_free])

        write_to.writerow(['Number Best Scenarios', len(best_population)])
        write_to.writerow(['Number Best Scenarios (duplicate free)',n_best_dup_free])

        write_to.writerow(['Ratio Critical/All scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
        write_to.writerow(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

        write_to.writerow(['Ratio Best Critical/Best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])
        write_to.writerow(['Ratio Best Critical/Best Scenarios (duplicate free)', '{0:.2f}'.format(n_crit_best_dup_free/n_best_dup_free)])

        
        write_to.writerow(['Iteration first critical found', '{}'.format(iter_crit)])
        write_to.writerow(['Fitness value of critical (first of population of interest)','{}'.format(str(inds_critical[0].get("F")) if len(inds_critical) > 0 else None) ])
        write_to.writerow(['Input value of critical (first of population of interest)','{}'.format(str(inds_critical[0].get("X")) if len(inds_critical) > 0 else None) ])
        
        write_to.writerow(['Average Fitness',f"%s" % avg_fit_all])
        write_to.writerow(['Min Fitness', f"%s" % min_fit_all])
        write_to.writerow(['Max Fitness', f"%s" % max_fit_all])
        
        f.close()

    log.info(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])
    log.info(['Number All Scenarios (duplicate free)', n_all_dup_free])
    log.info(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

def design_space(res, save_folder, classification_type=ClassificationType.DT, iteration=None):
    save_folder_design = save_folder + "design_space" + os.sep
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    if iteration is not None:
        save_folder_design_iteration = save_folder_design + 'TI_' + str(iteration) + os.sep
        Path(save_folder_design_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_design_iteration
 
    problem = res.problem
    design_names = problem.design_names
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    all_population = res.obtain_archive()
    critical_all, _ = all_population.divide_critical_non_critical()

    if classification_type == ClassificationType.DT:
        save_folder_classification = save_folder + "classification" + os.sep
        Path(save_folder_classification).mkdir(parents=True, exist_ok=True)
        regions = decision_tree.generate_critical_regions(all_population, problem, save_folder=save_folder_classification)
    
    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_var - 1):
        for axis_y in range(axis_x + 1, n_var):
            if classification_type == ClassificationType.DT:
                for region in regions:
                    x_rectangle = region.xl[axis_x]
                    y_rectangle = region.xl[axis_y]
                    width_rectangle = region.xu[axis_x] - region.xl[axis_x]
                    height_rectangle = region.xu[axis_y] - region.xl[axis_y]
                    region_color = color_not_critical

                    if region.is_critical:
                        region_color = color_critical
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor=region_color, lw=1.5, ls='-',
                                                  facecolor='none', alpha=0.2))
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor='none',
                                                  facecolor=region_color, alpha=0.05))

            ax = plt.subplot(111)
            plt.title(f"{res.algorithm.__class__.__name__}\nDesign Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

            if classification_type == ClassificationType.DT:
                critical, not_critical = all_population.divide_critical_non_critical()
                if len(not_critical) != 0:
                    ax.scatter(not_critical.get("X")[:, axis_x], not_critical.get("X")[:, axis_y],
                                s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_not_critical, marker='o')
                if len(critical) != 0:
                    ax.scatter(critical.get("X")[:, axis_x], critical.get("X")[:, axis_y], s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_critical, marker='o')

                
                opt = get_nondominated_population(all_population)
                critical_opt, not_critical_opt = opt.divide_critical_non_critical()

                if len(critical_opt) != 0:
                    ax.scatter(critical_opt.get("X")[:, axis_x], critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_critical, marker='o')
                                
                if len(not_critical_opt) != 0:
                    ax.scatter(not_critical_opt.get("X")[:, axis_x], not_critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_not_critical, marker='o')


            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers(critical_name = "")
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.png')
            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.pdf', format="pdf")

            plt.clf()
    
    # output 3d plots
    if n_var == 3:
        visualize_3d(all_population, save_folder_design, design_names, mode="critical", markersize=20, do_save=True)
    plt.close(f)

def design_space_from_pop(pop, 
                        design_names,
                        feature_names,
                        n_var,
                        xl,
                        xu,
                        save_folder, 
                        classification_type = ClassificationType.DT, 
                        iteration = None,
                        title_plot = "",
                        criticality_threshold_min = decision_tree.CRITICALITY_THRESHOLD_MIN,
                        suffix_class_folder = "",
                        suffix_design_folder = "",
                        fail_name = "Critical",
                        not_fail_name = "Not critical",
                        max_plots = 10): # limit number of plots
    save_folder_design = save_folder + "design_space" + suffix_design_folder + os.sep
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    if iteration is not None:
        save_folder_design_iteration = save_folder_design + 'TI_' + str(iteration) + os.sep
        Path(save_folder_design_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_design_iteration

    all_population = pop
    critical_all, _ = all_population.divide_critical_non_critical()

    if classification_type == ClassificationType.DT:
        save_folder_classification = save_folder + "classification" + os.sep + suffix_class_folder + os.sep
        Path(save_folder_classification).mkdir(parents=True, exist_ok=True)
        regions = decision_tree.generate_critical_regions_raw(all_population, 
                                                            feature_names, 
                                                            xl,
                                                            xu,
                                                            save_folder=save_folder_classification,
                                                            criticality_threshold_min=criticality_threshold_min)
    
    counter = 0
    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_var - 1):
        for axis_y in range(axis_x + 1, n_var):
            if counter > max_plots:
                plt.close(f)

                return
            else:
                counter += 1

            if classification_type == ClassificationType.DT:
                for region in regions:
                    x_rectangle = region.xl[axis_x]
                    y_rectangle = region.xl[axis_y]
                    width_rectangle = region.xu[axis_x] - region.xl[axis_x]
                    height_rectangle = region.xu[axis_y] - region.xl[axis_y]
                    region_color = color_not_critical

                    if region.is_critical:
                        region_color = color_critical
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor=region_color, lw=1.5, ls='-',
                                                  facecolor='none', alpha=0.2))
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor='none',
                                                  facecolor=region_color, alpha=0.05))

            ax = plt.subplot(111)
            plt.title(f"{title_plot}\nDesign Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + f" of which are {fail_name.lower()})")

            if classification_type == ClassificationType.DT:
                critical, not_critical = all_population.divide_critical_non_critical()
                if len(not_critical) != 0:
                    ax.scatter(not_critical.get("X")[:, axis_x], not_critical.get("X")[:, axis_y],
                                s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_not_critical, marker='o')
                if len(critical) != 0:
                    ax.scatter(critical.get("X")[:, axis_x], critical.get("X")[:, axis_y], s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_critical, marker='o')

                
                opt = get_nondominated_population(all_population)
                critical_opt, not_critical_opt = opt.divide_critical_non_critical()

                if len(critical_opt) != 0:
                    ax.scatter(critical_opt.get("X")[:, axis_x], critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_critical, marker='o')
                                
                if len(not_critical_opt) != 0:
                    ax.scatter(not_critical_opt.get("X")[:, axis_x], not_critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_not_critical, marker='o')


            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers(fail_name, not_fail_name)
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.png')
            # plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.pdf', format="pdf")

            plt.clf()
    
    # output 3d plots
    if n_var == 3:
        visualize_3d(all_population, save_folder_design, design_names, mode="critical", markersize=20, do_save=True)
    plt.close(f)
    
def backup_problem(res,save_folder):
    save_folder_problem = save_folder + BACKUP_FOLDER
    Path(save_folder_problem).mkdir(parents=True, exist_ok=True)   

    import dill
    with open(save_folder_problem + os.sep + "problem", "wb") as f:
        dill.dump(res.problem, f)
        
def objective_space(res, save_folder, iteration=None, show=False, last_iteration=LAST_ITERATION_ONLY_DEFAULT):
    save_folder_objective = save_folder + "objective_space" + os.sep
    Path(save_folder_objective).mkdir(parents=True, exist_ok=True)   
    save_folder_plot = save_folder_objective

    if iteration is not None:
        save_folder_iteration = save_folder_objective + 'TI_' + str(iteration) + os.sep
        Path(save_folder_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_iteration
 
    hist = res.history
    problem = res.problem
    pf = problem.pareto_front()
    n_obj = problem.n_obj
    objective_names = problem.objective_names
    
    if n_obj == 1:
        plot_single_objective_space(result=res,
                                   save_folder_plot=save_folder_plot,
                                   objective_names=objective_names,
                                   show=show,
                                   pf=pf)
    else:
        plot_multi_objective_space(res, n_obj, save_folder_objective, objective_names, show, pf, last_iteration)

        # output 3d plots
        if n_obj == 3:
            all_population = res.obtain_archive()
            visualize_3d(all_population, 
                save_folder_objective, 
                objective_names, 
                mode="critical", 
                markersize=20, 
                do_save=True,
                dimension="F",
                angles=[(45,-45),(45,45),(45,135)],
                show=show)
        # combine os plots over time in one plot
        def plot_overview_iterations(save_folder):
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image
            import os
            count= 0

            for filename in os.listdir(save_folder):
                if os.path.isdir(save_folder + os.sep + filename) and filename.startswith("iteration"):
                    count +=1

            n_col = min(1,count)
            n_row = int(np.ceil(count/n_col))

            fw = 6 * n_col
            fl = 5 * n_row

            # Create a figure and axes
            fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(fw,fl))  # Adjust the number of rows and columns as needed
            print(f"type: {type(axs)}")
            if isinstance(axs,np.ndarray):
                ax = axs.ravel()
            else:
                ax = [axs]

            for a in ax:
                a.axis('off')

            for i in range(count):
                def get_path(file = None):
                    return save_folder + os.sep + f"iteration_{i}" + os.sep + (file if file is not None else "")
                
                folder_path = get_path()
                for file in os.listdir(folder_path):
                    if file.endswith(".png"):
                        break
                # Load images
                image = Image.open(get_path(file))
                width = image.size[0] // 2
                height = image.size[1]  // 2
                image = image.resize((width,height))
                image_array = np.array(image)
                ax[i].axis('off')
                ax[i].imshow(image_array)
                ax[i].set_title(f'Iteration {i}')
            fig.tight_layout()
            # Show the plot
            # plt.show()
            plt.savefig(save_folder + os.sep + "os_overview.png", format="png")
                
        if not last_iteration:
            plot_overview_iterations(save_folder=save_folder_objective)
    log.info(f"Objective Space: {save_folder_plot}")
    

def plot_multi_objective_space(res, n_obj, save_folder_objective, objective_names, show, pf, last_iteration):
    all_population = Population()
    # n_evals_all = 0
    for i, generation in enumerate(res.history):
        # TODO first generation has somehow archive size of 0
        # all_population = generation.archive #Population.merge(all_population, generation.pop)
        n_eval = generation.evaluator.n_eval
        # TODO assure that every algorithm stores in n_eval the number of evaluations SO FAR performed!!
        all_population = res.archive[0:n_eval]
        critical_all, _ = all_population.divide_critical_non_critical()
        
        # plot only last iteration if requested
        if last_iteration and i < len(res.history) -1 :
            continue
        
        save_folder_iteration = save_folder_objective + f"iteration_{i}" + os.sep
        Path(save_folder_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_iteration

        f = plt.figure(figsize=(12, 10))
        for axis_x in range(n_obj - 1):
            for axis_y in range(axis_x + 1, n_obj):
                ax = plt.subplot(111)
                plt.title(f"{res.algorithm.__class__.__name__}\nObjective Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

                if True: #classification_type == ClassificationType.DT:
                    critical, not_critical = all_population.divide_critical_non_critical()

                    critical_clean = duplicate_free(critical)
                    not_critical_clean = duplicate_free(not_critical)
                    
                    if len(not_critical_clean) != 0:
                        ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                                    facecolors=color_not_optimal,
                                    edgecolors=color_not_critical, marker='o')
                    if len(critical_clean) != 0:
                        ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                                    facecolors=color_not_optimal, edgecolors=color_critical, marker='o')

                if pf is not None:
                    ax.plot(pf[:, axis_x], pf[:, axis_y], color='blue', lw=0.7, zorder=1)

                if True: #classification_type == ClassificationType.DT:
                    optimal_pop = get_nondominated_population(all_population)
                    critical, not_critical = optimal_pop.divide_critical_non_critical()
                    critical_clean = duplicate_free(critical)
                    not_critical_clean = duplicate_free(not_critical)
                    
                    if len(not_critical_clean) != 0:
                        ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                                facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
                    if len(critical_clean) != 0:
                        ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                                facecolors=color_optimal, edgecolors=color_critical, marker='o')

                #limit axes bounds, since we do not want to show fitness values as 1000 or int.max, 
                # that assign bad quality to worse scenarios
                if CONSIDER_HIGH_VAL_OS_PLOT:                    
                    pop_f_x = all_population.get("F")[:,axis_x]
                    clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == PENALTY_MAX))
                    max_x_f_ind = max(clean_pop_x)
                    clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == PENALTY_MIN))
                    min_x_f_ind = min(clean_pop_x)

                    pop_f_y = all_population.get("F")[:,axis_y]
                    clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == PENALTY_MAX))
                    max_y_f_ind = max(clean_pop_y)
                    clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == PENALTY_MIN))
                    min_y_f_ind = min(clean_pop_y)

                    eta_x = abs(max_x_f_ind - min_x_f_ind) / 10
                    eta_y = abs(max_y_f_ind- min_y_f_ind) / 10
                    
                    plt.xlim(min_x_f_ind - eta_x, max_x_f_ind  + eta_x)
                    plt.ylim(min_y_f_ind - eta_y, max_y_f_ind  + eta_y)
                    
                plt.xlabel(objective_names[axis_x])
                plt.ylabel(objective_names[axis_y])

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                marker_list = create_markers()
                if pf is not None:
                    markers = marker_list[2:]
                else:
                    markers = marker_list[2:-1]

                plt.legend(handles=markers,
                        loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

                if show:
                    plt.show()
                plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.png')
                plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.pdf', format='pdf')
            
                plt.clf()
        plt.close(f)

def plot_single_objective_space(result, save_folder_plot, objective_names, show, pf):
    res = result
    problem = res.problem

    x_axis_width = 10
    plt.figure(figsize=(x_axis_width,6))
    plt.axis('auto')

    # ax = plt.subplot(111)
    # ax.axis('auto')
    fig = plt.gcf()

    # Set the figure size to stretch the x-axis physically
    fig.set_size_inches(x_axis_width, fig.get_figheight(), forward=True)

    all_population = res.obtain_archive()
    critical, _ = all_population.divide_critical_non_critical()

    plt.title(f"{res.algorithm.__class__.__name__}\nObjective Space | {problem.problem_name}" + \
                    " (" + str(len(all_population)) + " testcases, " + \
                        str(len(critical)) + " of which are critical)")
        
    n_evals_all = 0

    # we plot the fitness over time as it is only one value for each iteration
    for i, generation in enumerate(res.history):
        n_eval = generation.evaluator.n_eval
        n_evals_all += n_eval
        # print(f"[visualizer] n_eval: {n_eval}")
        all_population = res.archive[0:n_evals_all]
        axis_y = 0
        critical, not_critical = all_population.divide_critical_non_critical()
        critical_clean = duplicate_free(critical)
        not_critical_clean = duplicate_free(not_critical)
        
        if len(not_critical_clean) != 0:
            plt.scatter([i]*len(not_critical_clean), not_critical_clean.get("F")[:, axis_y], s=40,
                        facecolors=color_not_optimal,
                        edgecolors=color_not_critical, marker='o')
        if len(critical_clean) != 0:
            plt.scatter([i]*len(critical_clean), critical_clean.get("F")[:, axis_y], s=40,
                        facecolors=color_not_optimal, edgecolors=color_critical, marker='o')
        optimal_pop = get_nondominated_population(all_population)
        critical, not_critical = optimal_pop.divide_critical_non_critical()

        critical_clean = duplicate_free(critical)
        not_critical_clean = duplicate_free(not_critical)
        
        if len(not_critical_clean) != 0:
            plt.scatter([i]*len(not_critical_clean), not_critical_clean.get("F")[:, axis_y], s=40,
                    facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
        if len(critical_clean) != 0:
            plt.scatter([i]*len(critical_clean), critical_clean.get("F")[:, axis_y], s=40,
                    facecolors=color_optimal, edgecolors=color_critical, marker='o')

    
        # box = fig.get_position()
        # fig.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        marker_list = create_markers()
        if pf is not None:
            markers = marker_list[2:]
        else:
            markers = marker_list[2:-1]
    plt.xlabel("Iteration")
    plt.ylabel(problem.objective_names[0])
    plt.legend(handles=markers,
            #loc='center left',
            loc='best',
            bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

    if show:
        plt.show()
    plt.savefig(save_folder_plot + objective_names[0] + '_iterations.png')
    plt.clf()

def optimal_individuals(res, save_folder):
    """Output of optimal individuals (duplicate free)"""
    problem = res.problem
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'optimal_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_"+ objective_names[i])

        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        clean_pop = duplicate_free(res.opt)

        for index in range(len(clean_pop)):
            row = [index]
            row.extend(["%.6f" % X_i for X_i in clean_pop.get("X")[index]])
            row.extend(["%.6f" % F_i for F_i in clean_pop.get("F")[index]])
            row.extend(["%i" % clean_pop.get("CB")[index]])
            write_to.writerow(row)
        f.close()

def all_individuals(res, save_folder):
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}")
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        index = 0
        all_individuals = res.obtain_archive()
        for index, ind in enumerate(all_individuals):
                row = [index]
                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
                row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in ind.get("F")])
                row.extend(["%i" % ind.get("CB")])
                write_to.writerow(row)
        f.close()

def write_disagree_testcases(res, save_folder, dup_free = True):
    problem = res.problem
    hist = res.history    # TODO check why when iterating over the algo in the history set is different
    design_names = problem.design_names
    objective_names = problem.objective_names
    if problem.is_simulation() and problem.__class__.__name__!= "ADASProblem":
        with open(save_folder + 'all_disagree_testcases.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Index']
            for i in range(problem.n_var):
                header.append(design_names[i])
            for i in range(problem.n_obj):
                header.append(f"Fitness_{objective_names[i]}")
            # column to indicate wheter individual is critical or not 
            header.append(f"Critical")

            write_to.writerow(header)
            
            inds = get_pop_using_mode(res=res, 
                                    mode="all")
            
            if dup_free:
                clean_pop, clean_indeces = duplicate_free_with_index(inds)
            else:
                clean_pop = inds
                clean_indeces = [i for i in range(0,len(clean_pop))]
                
            # get disagreements
            if len(clean_pop) == 0:
                return
            
            crit_all = clean_pop.get("CB_all")
            n_sims = len(problem.simulate_functions)

            disagree_map = {}
            # combination
            for i in range(n_sims-1):
                for j in range(i+1, n_sims):
                    # Find indices where there is disagreement
                    indices_1 = np.where((crit_all[:, i] == 1) & (crit_all[:, j] == 0))[0]
                    indices_2 = np.where((crit_all[:, j] == 1) & (crit_all[:, i] == 0))[0]
                    
            # Update the disagree_map with the indices
            disagree_map[(i, j)] = indices_1.tolist()
            disagree_map[(j, i)] = indices_2.tolist()

            for key, value in disagree_map.items():
                indices = value
                for i in indices:
                    row = [clean_indeces[i]]
                    row.extend(["%.6f" % X_i for X_i in clean_pop.get("X")[i]])
                    row.extend(["%.6f" % F_i for F_i in clean_pop.get("F")[i]])
                    row.extend(["%i" % clean_pop.get("CB")[i]])
                    write_to.writerow(row)
            f.close()
            
def all_critical_individuals(res, save_folder):
    """Output of all critical individuals"""
    problem = res.problem
    hist = res.history    # TODO check why when iterating over the algo in the history set is different
    design_names = problem.design_names
    objective_names = problem.objective_names

    all = res.obtain_archive()
    critical = all.divide_critical_non_critical()[0]

    with open(save_folder + 'all_critical_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
                header.append(f"Fitness_"+ objective_names[i])
                
        write_to.writerow(header)

        index = 0
        # for algo in hist:
        for i in range(len(critical)):
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in critical.get("X")[i]])
            row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in critical.get("F")[i]])
            write_to.writerow(row)
            index += 1
        f.close()

''' Write down the population for each generation'''
def write_generations(res, save_folder):

    save_folder_history = save_folder + "generations" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names
    
    for i, algo in enumerate(hist):
        with open(save_folder_history + f'gen_{i+1}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Index']
            for i in range(problem.n_var):
                header.append(design_names[i])
            for i in range(problem.n_obj):
                header.append(f"Fitness_{objective_names[i]}")
            # column to indicate wheter individual is critical or not 
            header.append(f"Critical")

            write_to.writerow(header)
            index = 0
            for i in range(len(algo.pop)):
                row = [index]
                row.extend(["%.6f" % X_i for X_i in algo.pop.get("X")[i]])
                row.extend(["%.6f" % F_i for F_i in algo.pop.get("F")[i]])
                row.extend(["%i" % algo.pop.get("CB")[i]])
                write_to.writerow(row)
                index += 1
            f.close()

def plot_generations_scenario(res, save_folder):
    
    if isinstance(res.opt.get("SO")[0], SimulationOutput):
        save_folder_history = save_folder + "generations_gif" + os.sep
        Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

        hist = res.history
        
        for i, algo in enumerate(hist):
            save_folder_generation = save_folder_history + os.sep + f"gen_{i}" + os.sep
            Path(save_folder_generation).mkdir(parents=True, exist_ok=True) 
            pop = algo.pop
            for index, simout in enumerate(pop.get("SO")):
                file_name = str(index) + str("_trajectory")
                param_values = pop.get("X")[index]
                plotter.plot_gif(param_values, 
                                simout, 
                                save_folder_generation, 
                                file_name, 
                                skip=1, 
                                plot_animation=False,
                                focus_on_road=True)    
            
def plot_roads_overview_generation(res, save_folder):
        
    save_folder_history = save_folder + "generations_gif" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    if len(res.opt) == 0:
        return
    
    # if is not multisim
    if isinstance(res.opt.get("SO")[0], SimulationOutput):
        is_multisim = False
    else:
        is_multisim = True

    hist = res.history
    
    for i, algo in enumerate(hist):
        save_folder_generation = save_folder_history + os.sep + f"gen_{i}" + os.sep
        Path(save_folder_generation).mkdir(parents=True, exist_ok=True) 
        pop = algo.pop

        num_subplots = len(pop)
        
        if is_multisim:  
            n_sims = len(pop.get("SO")[0])
            fig, axs = plt.subplots(n_sims, num_subplots, figsize=(7*num_subplots, n_sims*10))
            for i_sim in range(n_sims):
                for index, simout in enumerate(pop.get("SO")):
                    file_name = str(index) + str("_trajectory")
                    param_values = pop.get("X")[index]
                    plotter.plot_gif(param_values, 
                                    simout[i_sim], # take the first if is multisim
                                    save_folder_generation, 
                                    file_name, 
                                    skip=1, 
                                    plot_animation=False,
                                    ax=axs[i_sim, index] if num_subplots > 1 else axs,
                                    focus_on_road=True)            
            plt.savefig(save_folder_generation + f"roads_gen_{i}.png")
            plt.clf()
            plt.close(fig)
        else:       
            fig, axs = plt.subplots(1, num_subplots, figsize=(7*num_subplots, 10))
            
            for index, simout in enumerate(pop.get("SO")):
                file_name = str(index) + str("_trajectory")
                param_values = pop.get("X")[index]
                plotter.plot_gif(param_values, 
                                simout, # take the first if is multisim
                                save_folder_generation, 
                                file_name, 
                                skip=1, 
                                plot_animation=False,
                                ax=axs[index] if num_subplots > 1 else axs,
                                focus_on_road=True)            
            plt.savefig(save_folder_generation + f"roads_gen_{i}.png")
            plt.clf()
            plt.close(fig)

def export_roads(res, save_folder):
    # only for single simulation problems
    is_multisim = False if isinstance(res.opt.get("SO")[0], SimulationOutput) else True
 
    save_folder_history = save_folder + "exported" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    crit_pop, non_crit_pop = res.obtain_all_population(). \
                divide_critical_non_critical()
    
    roads = {}
        
    # crate json file for each generation, write roads
    for index, simout in enumerate(crit_pop.get("SO")):
        simout = simout if not is_multisim else simout[0]
        roads[f"road_{index}"] = simout.otherParams["road"]
    data = json.dumps(roads, indent=4)
    with open(save_folder_history + f"critical_roads.json", 'w+') as f:
        f.write(data)         
 
    roads = {}
        
    # crate json file for each generation, write roads
    for index, simout in enumerate(non_crit_pop.get("SO")):
        simout = simout if not is_multisim else simout[0]
        roads[f"road_{index}"] = simout.otherParams["road"]
    data = json.dumps(roads, indent=4)
    with open(save_folder_history + f"non_critical_roads.json", 'w+') as f:
        f.write(data)     

def plot_critical_roads(res, save_folder):
    problem = res.problem
    is_simulation = problem.is_simulation()
    
    if is_simulation:
        save_folder_gif = save_folder + "critical_gif" + os.sep
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
        
        max = config.MAX_NUM_CRIT_GIFS
        inds = get_pop_using_mode(res=res, 
                                  mode="crit")
        clean_pop = duplicate_free(inds)[:max]
        if len(clean_pop) == 0:
            return

        if isinstance(clean_pop.get("SO")[0], SimulationOutput):
            # single simulation problem
            for index, simout in enumerate(clean_pop.get("SO")):
                file_name = str(index) + str("_trajectory")
                param_values = clean_pop.get("X")[index]
                plotter.plot_gif(param_values, 
                            simout, 
                            save_folder_gif, 
                            file_name, 
                            skip=1,
                            focus_on_road=True)  
        else:
            # multi simulation problem
            for index in range(len(clean_pop.get("SO"))):
                for sim_ind, simout in enumerate(clean_pop.get("SO")[index]):
                    file_name = str(index) + str(f"_trajectory_S{sim_ind + 1}")
                    param_values = clean_pop.get("X")[index]
                    plotter.plot_gif(param_values, 
                                    simout, 
                                    save_folder_gif, 
                                    file_name, 
                                    skip=1,
                                    focus_on_road=True)   
        # for index, simout in enumerate(clean_pop.get("SO_1")):
        #     file_name = str(index) + str("_trajectory_S2")
        #     param_values = clean_pop.get("X")[index]
        #     plotter.plot_gif(param_values, simout, save_folder_gif, file_name)
    else:
        log.info("No simulation visualization available. The experiment is not a simulation.")

def export_generations_roads(res, save_folder):
    # only for single simulation problems
    is_multisim = False if isinstance(res.opt.get("SO")[0], SimulationOutput) else True
 
    save_folder_history = save_folder + os.sep + "exported" + os.sep + "generations_roads" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    hist = res.history
    for i, algo in enumerate(hist):
    
        pop = algo.pop
        roads = {}
            
        # crate json file for each generation, write roads
        for index, simout in enumerate(pop.get("SO") if not is_multisim \
                                                     else pop.get("SO")[0]):
            roads[f"road_{index}"] = simout.otherParams["road"]
        data = json.dumps(roads, indent=4)
        with open(save_folder_history + f"roads_gen_{i}", 'w+') as f:
            f.write(data)

def simulations(res, save_folder, 
                mode=config.MODE_GIF_WRITING, 
                plot_animation = True,
                dup_free = True,
                limited = True,
                folder_name = "gif"):
    '''Visualization of the results of simulations'''
    ''' Plots scenarios only once when duplicates available'''
    
    problem = res.problem
    is_simulation = problem.is_simulation()
    
    if is_simulation:
        save_folder_gif = save_folder + folder_name + os.sep
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
        
        max = config.MAX_NUM_GIFS
        inds = get_pop_using_mode(res=res, 
                                  mode=mode)
        
        if dup_free and limited:
            clean_pop = duplicate_free(inds)[:max]
        elif dup_free:
            clean_pop = duplicate_free(inds)
        else:
            clean_pop = inds
            
        if len(clean_pop) == 0:
            return

        if isinstance(clean_pop.get("SO")[0], SimulationOutput):
            # single simulation problem
            for index, simout in enumerate(clean_pop.get("SO")):
                file_name = str(index) + str("_trajectory")
                param_values = clean_pop.get("X")[index]
                plotter.plot_gif(param_values, 
                            simout, 
                            save_folder_gif, 
                            file_name, 
                            skip=1,
                            focus_on_road=True,
                            plot_animation=plot_animation)  
        else:
            # multi simulation problem
            for index in range(len(clean_pop.get("SO"))):
                for sim_ind, simout in enumerate(clean_pop.get("SO")[index]):
                    file_name = str(index) + str(f"_trajectory_S{sim_ind + 1}")
                    param_values = clean_pop.get("X")[index]
                    plotter.plot_gif(param_values, 
                                    simout, 
                                    save_folder_gif, 
                                    file_name, 
                                    skip=1,
                                    focus_on_road=True,
                                    plot_animation=plot_animation)  
        # for index, simout in enumerate(clean_pop.get("SO_1")):
        #     file_name = str(index) + str("_trajectory_S2")
        #     param_values = clean_pop.get("X")[index]
        #     plotter.plot_gif(param_values, simout, save_folder_gif, file_name)
    else:
        log.info("No simulation visualization available. The experiment is not a simulation.")

def scenario_plot_disagreements(res, save_folder, dup_free = True, plot_animation = True):
    problem = res.problem
    is_simulation = problem.is_simulation()
    # print(f"class problem: {problem.__class__.__name__}")

    if is_simulation and problem.__class__.__name__!= "ADASProblem":
        save_folder_gif = save_folder + "gif_disagree" + os.sep
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
        
        inds = get_pop_using_mode(res=res, 
                                  mode="all")
        
        if dup_free:
            clean_pop, clean_indeces = duplicate_free_with_index(inds)
        else:
            clean_pop = inds
            clean_indeces = [i in range(0,len(clean_pop))]
            
        # get disagreements
        if len(clean_pop) == 0:
            return
        
        crit_all = clean_pop.get("CB_all")
        n_sims = len(problem.simulate_functions)

        disagree_map = {}
        # combination
        for i in range(n_sims-1):
            for j in range(i+1, n_sims):
                # Find indices where there is disagreement
                indices_1 = np.where((crit_all[:, i] == 1) & (crit_all[:, j] == 0))[0]
                indices_2 = np.where((crit_all[:, j] == 1) & (crit_all[:, i] == 0))[0]
                
        # Update the disagree_map with the indices
        disagree_map[(i, j)] = indices_1.tolist()
        disagree_map[(j, i)] = indices_2.tolist()

        for key, value in disagree_map.items():
            sim_0 = key[0]
            sim_1 = key[1]
            indices = value
            for index in indices[:config.MAX_PLOT_DISAGREE]:
                simout_0 = clean_pop.get("SO")[index][sim_0]
                simout_1 = clean_pop.get("SO")[index][sim_1]

                param_values = clean_pop.get("X")[index]

                file_name = str(clean_indeces[index]) + str(f"_dis_trajectory_S{sim_0 + 1}")
                plotter.plot_gif(param_values, 
                                 simout_0, 
                                 save_folder_gif, 
                                 file_name, 
                                 skip=1, 
                                 focus_on_road=True,
                                 plot_animation=plot_animation)   

                file_name = str(clean_indeces[index]) + str(f"_dis_trajectory_S{sim_1+ 1}")
                plotter.plot_gif(param_values, simout_1, save_folder_gif, file_name, skip=1, focus_on_road=True,
                                 plot_animation=plot_animation)    
    else:
        log.info("No simulation visualization available. The experiment is not a simulation.")

def get_pop_using_mode(res: Result, mode: str):
    inds = Population()
    # print(f"mode: {mode}")
    if mode == "all":
        inds = res.obtain_archive()
    elif mode == "opt":
        inds = res.opt
    elif mode == "crit":
        all = res.obtain_archive()
        inds, _ = all.divide_critical_non_critical()
    else:
        print("Mode is not accepted. Accepted modes are: all, opt, crit.")
    return inds

def comparison_trace(res, save_folder, mode="opt", type="X"):
    problem = res.problem
    is_simulation = problem.is_simulation()
    if is_simulation:
        max = config.MAX_NUM_GIFS
        inds = get_pop_using_mode(res=res, 
                                  mode=mode)
        clean_pop = duplicate_free(inds)[:max]
        if len(clean_pop) == 0:
            return
        
        # only for multi simulation problem
        if not isinstance(clean_pop.get("SO")[0], SimulationOutput):
            for index in range(len(clean_pop.get("SO"))):
                save_folder_gif = save_folder + os.sep + "trace_comp" +  os.sep + f"ind_{index + 1}"
                Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
                                    
                f = plt.figure()
                cmap = plt.get_cmap('gnuplot')
                colors = [cmap(i) for i in np.linspace(0, 1, len(clean_pop.get("SO")[0]))]

                for sim_ind, simout in enumerate(clean_pop.get("SO")[index]):
                    sim_name = simout.otherParams["simulator"] \
                               if "simulator" in simout.otherParams  \
                               else f"Simulator {sim_ind + 1}"
                    
                    if type.lower() == "x":
                        plt.plot([i for i in range(len(simout.location["ego"]))],
                                [v[0] for v in simout.location["ego"]],
                                label=sim_name,
                                color=colors[sim_ind])
                    elif type.lower() == "y":
                        plt.plot([i for i in range(len(simout.location["ego"]))],
                            [v[1] for v in simout.location["ego"]],
                            color=colors[sim_ind],
                            label=sim_name)
                    elif type.lower() == "v":
                      plt.plot([i for i in range(len(simout.speed["ego"]))],
                            [v for v in simout.speed["ego"]],
                            color=colors[sim_ind],
                            label=sim_name)
                    elif type.lower() in simout.otherParams:
                        plt.plot([i for i in range(len(simout.otherParams[type.lower()]))],
                            [v for v in simout.otherParams[type.lower()]],
                            color=colors[sim_ind],
                            label=sim_name)
                    else:
                        log.info("Type is unknown")
                        return
                plt.xlabel('Timestep')
                plt.ylabel(f'{type.upper()}')
                plt.title(f'{type.upper()} for Different Simulators')
                plt.legend()
                plt.savefig(save_folder_gif + os.sep + f"{type.upper()}_trace_{index + 1}")
                plt.clf()
                plt.close(f)
        else:
            return "Problem is not a multi simulation problem. No comparison possible."
    else:
        log.info("No simulation visualization available. The experiment is not a simulation.")


def write_fitness_all_sims(res, save_folder):
    
    all = res.obtain_all_population()

    if all.get("F_all")[0] is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        n_sim = len(all.get("F_all")[0])

        with open(save_folder + 'fitness_all_sims.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            sim_names = [simout.otherParams["simulator"] for simout in all.get("SO")[0]]
            header = sim_names
            write_to.writerow(header)

            for ind in all:
                f_values = ind.get("F_all")
                write_to.writerow(f_values)

        f.close()
    else:
        log.info("This is not supported for this type of problem.")


def write_criticality_all_sims(res, save_folder):
    
    all = res.obtain_all_population()

    if all.get("CB_all")[0] is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        n_sim = len(all.get("CB_all")[0])

        with open(save_folder + 'criticality_all_sims.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            sim_names = [simout.otherParams["simulator"] for simout in all.get("SO")[0]]
            header = sim_names
            write_to.writerow(header)

            for ind in all:
                labels = ind.get("CB_all")
                write_to.writerow(labels)

        f.close()
    else:
        log.info("This is not supported for this type of problem.")


def write_multisim_analysis(res, save_folder, no_duplicates=True):
    all = res.obtain_archive()

    if all.get("CB_all")[0] is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        n_crit_sims = []
        ratio_n_crit_sims = []
        n_non_crit_sims = []

        if no_duplicates:
            crit_all = duplicate_free(population=all). \
                    get("CB_all")
        else:
            crit_all = all.get("CB_all")
        
        n_sims = crit_all.shape[1]
        n_all = crit_all.shape[0]

        logical_comb = np.ones(n_all, dtype=bool)
        logical_comb_neg = np.ones(n_all, dtype=bool)

        # agreements 
        for i in range(n_sims):
            crit_ar = crit_all[:,i]
            v = np.count_nonzero(crit_ar)
            n_crit_sims.append(v)
            n_non_crit_sims.append(n_all - v)

            ratio_n_crit_sims.append(float("%.2f" % (v/n_all)))

            logical_comb = np.logical_and(logical_comb, crit_ar)
            logical_comb_neg = np.logical_and(logical_comb_neg, ~crit_ar)
            # print(f"~crit_ar: {~crit_ar}")
            # print(f"log_comb_neg: {logical_comb_neg}")
            # last_array = copy.deepcopy(crit_ar)
        
        # disagreements
        # for now for two sims
            
        disagree_map = {}
        # combination
        for i in range(n_sims-1):
            for j in range(i+1, n_sims):
                value_1 = np.count_nonzero(np.logical_and(crit_all[:,i],~crit_all[:,j]))
                value_2 = np.count_nonzero(np.logical_and(~crit_all[:,i],crit_all[:,j]))
                disagree_map.update({(i,j): value_1})
                disagree_map.update({(j,i): value_2})

        n_crit_aggreement = np.count_nonzero(logical_comb)
        n_non_crit_agreement = np.count_nonzero(logical_comb_neg)
        # print(f"count non zero negated: {np.count_nonzero(logical_comb_neg)}")
        n_aggreement = n_crit_aggreement + n_non_crit_agreement
        
        ###### average fitness analysis

        avg_fit_all = np.sum(all.get("F_all")/len(all.get("F_all")), axis=0)


        ######### ratio

        with open(save_folder + f'multisim_analysis.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Metric', 'Value', 'Ratio All']
            write_to.writerow(header)
            write_to.writerow(['Number Tests All', len(all), 1])
            write_to.writerow(['Number Tests All (duplicate free)', n_all, n_all/len(all)])
            write_to.writerow(['Number Single Critical', n_crit_sims, ratio_n_crit_sims])
            write_to.writerow(['Number Single Non-Critical', n_non_crit_sims, [float("%.2f" % (1 - v)) for v in ratio_n_crit_sims]])

            write_to.writerow(['Agreement Critical + Non-Critical',n_aggreement, "%.2f" % (n_aggreement/n_all)])
            write_to.writerow(['Agreement Critical',n_crit_aggreement,"%.2f" % (n_crit_aggreement/n_all)])
            write_to.writerow(['Agreement Non-Critical',n_non_crit_agreement,"%.2f" % (n_non_crit_agreement/n_all)])

            for _, (comb, value) in enumerate(disagree_map.items()):
                write_to.writerow([f'Critical Sim {comb[0] + 1}, Non-critical {comb[1] + 1}', int(value),"%.2f" % (int(value)/n_all)])
            write_to.writerow(['Average Fitness', ["%s" % a for a in avg_fit_all],None])

        f.close()
        
def write_config(save_folder):
    # Open the file for reading
    with open(os.getcwd() + os.sep + 'config.py', 'r') as file:
        # Read the content of the file
        file_content = file.readlines()

    # Remove lines starting with '#' or containing 'import'
    filtered_content = [line for line in file_content if not (line.startswith('#') or 'import' in line)]

    # Remove multiple consecutive newlines
    cleaned_content = []
    previous_line_empty = False
    for line in filtered_content:
        if line.strip():  # Check if the line is not empty
            cleaned_content.append(line)
            previous_line_empty = False
        elif not previous_line_empty:  # If line is empty and previous line wasn't empty
            cleaned_content.append(line)
            previous_line_empty = True

    # Open another file for writing (or create it if it doesn't exist)
    with open(save_folder + os.sep + 'config.txt', 'w') as output_file:
        # Write the cleaned content to the output file
        output_file.writelines(cleaned_content)
        
def calculate_n_crit_distinct(res: Result, 
                             save_folder: str, 
                             bound_min = None, 
                             bound_max = None, 
                             n_cells=config.N_CELLS,
                             optimal=False,
                             var = "F"):

    log.info(f"------ Performing number critical analysis ({var}) ------")
    log.info(f"------ Optimal: {optimal}------")
    log.info(f"------ N_cells: {n_cells}------")

    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_n_crit_distinct(res, 
                                            bound_min, 
                                            bound_max, 
                                            n_cells=n_cells,
                                            optimal=optimal,
                                            var = var
                                            )

    if eval_result is None:
        log.info("No number distinct criticals computed")
        return
    
    n_evals, cid = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, cid, f"n_crit{'_opt' if optimal else ''}_{var}",save_folder)

    f = plt.figure()
    plt.plot(n_evals, cid, color='black', lw=0.7)
    plt.scatter(n_evals, cid, facecolor="none", edgecolor='black', marker="o")
    plt.title(f"Failure Analysis ({var})\n{res.problem.problem_name}")
    plt.xlabel("Function Evaluations")
    plt.ylabel(f"Number Critical (Cell Size = {n_cells})")
    plt.savefig(save_folder_plot + f'n_crit{"_opt" if optimal else ""}_{var}.png')
    plt.close()
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final n_critical value: {cid[-1]}")