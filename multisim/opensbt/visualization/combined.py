import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from opensbt.model_ga.individual import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from opensbt.model_ga.population import *
from opensbt.utils.sorting import get_nondominated_population
import os
import csv
from opensbt.visualization.output import *
from opensbt.quality_indicators.quality import EvaluationResult
from matplotlib import pyplot as plt
import scipy
from scipy.interpolate import interp1d
import warnings
import sys
import matplotlib
from opensbt.utils.duplicates import duplicate_free

BACKUP_FOLDER = "backup" + os.sep

# union critical solutions from all runs to approximate "real" critical design space
def calculate_combined_crit_pop(run_paths):
    len(f"run_paths: {run_paths}")
    crit_pop = Population()
    for run_path in run_paths:
        crit_run = read_pf_single(run_path + os.sep + "all_critical_testcases.csv")
        crit_pop = Population.merge(crit_pop, crit_run)
    # crit_pop = duplicate_free(crit_pop)
    # inds = [Individual(ind.get("X").tolist()) for ind in crit_pop]
    # assign critical label for the visualization
    for i in range(0,len(crit_pop)):
        crit_pop[i].set("CB",True)
    return crit_pop

# TODO add information on the deviation of the values wrt. to differen runs in the plots
def calculate_combined_pf(run_paths, critical_only=False):
    pf_pop = Population()
    for run_path in run_paths:
        pf_run = read_pf_single(run_path + os.sep + "optimal_testcases.csv")
        pf_pop = Population.merge(pf_pop, pf_run)

    print(f"len: {len(pf_pop)}")
    print(f"X: {len(pf_pop[0].get('X'))}")
    print(f"F: {len(pf_pop[0].get('F'))}")

    print(pf_pop)
    pf_pop = get_nondominated_population(pf_pop)
    if critical_only:
        crit_inds = np.where(pf_pop.get("CB"))[0]
        pf_pop = pf_pop[crit_inds]
    inds = [ind.get("F").tolist() for ind in pf_pop]
    pf = np.array(inds, dtype=float)
    # print(pf)
    return pf


''' 
    output mean/std/min/max for final metric value instead for several number of evaluations as in plot_combined_analysis
'''

def plot_combined_analysis_last_min_max(metric_name, run_paths_array, save_folder):
    plot_array = []
    
    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        values_all = []

        y_stacked = np.zeros((len(run_paths), 1))
        f = plt.figure(figsize=(7, 5))

        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER, metric_name)
            n_evals, hv = eval_res.steps, eval_res.values
            values_all.append(hv[-1])

            # print(f"n_eval: {n_evals}")
            # print(f"n_eval: {hv}")

            plt.plot(n_evals, hv, marker='.', linestyle='--', label='run ' + str(key_run + 1))

        max_value = max(values_all)
        min_value = min(values_all)

        max_value_runs = [max_value]
        min_value_runs = [min_value]

        #TODO print min max

        def std_dev(y_values):
            y_mean = np.sum(y_values, axis=0) / np.count_nonzero(y_values, axis=0)
            y_error = []
            for i in range(len(y_mean)):
                square_sum = 0
                n = 0
                for j in range(len(run_paths)):
                    if np.all(y_values[j, i]):
                        n += 1
                        deviation = y_values[j, i] - y_mean[i]
                        square_sum += deviation ** 2
                if n <= 1:
                    variance = 0
                else:
                    variance = square_sum / (n - 1)
                standart_deviation = np.sqrt(variance)
                y_error.append(standart_deviation / np.sqrt(n))
            return y_error, y_mean
        
        y_stacked[:,0] = values_all
        y_error, y_mean = std_dev(y_stacked)

        x_plot = n_evals # TODO make labels RS and NSGA-II
        plt.plot(x_plot, y_mean[0:len(x_plot)], color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        Path(save_folder).mkdir(parents=True, exist_ok=True)   

        plt.title(f"{metric_name.upper()} Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel(f"{metric_name.upper()}")
        plt.savefig(save_folder + f'{metric_name}_combined_single' + str(algo) + '.png')
        plt.clf()
        plt.close(f)
        plot_array.append([x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], min_value_runs, max_value_runs])

    return plot_array

def write_last_metric_values(metric_name_load, run_paths_array, save_folder, metric_name_label=None):
    values_algo = {}
    algos = list(run_paths_array.keys())
    print(algos)
    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        values_algo[algo] = []
        for run_path in run_paths:
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            _, v = eval_res.steps, eval_res.values
            values_algo[algo].append(v[-1])

    with open(save_folder + f'overview_{metric_name_label}.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['run', f'{algos[0]}',f'{algos[1]}']
        write_to.writerow(header)

        for i in range(0,len(values_algo[algo])):
            algo1 = algos[0]
            algo2 = algos[1]
            write_to.writerow([f'{i+1}', values_algo[algo1][i],  values_algo[algo2][i]])

def plot_combined_analysis(metric_name_load, 
                           run_paths_array, 
                           save_folder, 
                           n_func_evals_lim, 
                           n_fitting_points, 
                           metric_name_label=None, 
                           step_chkp=None, 
                           error_mean=True):
    plot_array = []

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        num_evals_limit = []

        for run_path in run_paths:
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            n_evals, hv = eval_res.steps, eval_res.values
            num_evals_limit.append(n_evals[-1])
        
        if n_func_evals_lim == 1:
            min_num_evals = n_func_evals_lim + 0.1
        else:
            min_num_evals = n_func_evals_lim + 1

        step = min_num_evals / n_fitting_points

        if step_chkp is not None:
            step = step_chkp 
        
        x = np.arange(step, min_num_evals, step=step)
        y_stacked = np.zeros((len(run_paths), len(x)))

        f = plt.figure(figsize=(7, 5))
        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER, metric_name_load)
            n_evals, hv = eval_res.steps, eval_res.values
            spl = scipy.interpolate.interp1d(np.array(n_evals), np.array(hv),fill_value="extrapolate")
            x_run = np.arange(step, min_num_evals, step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y
            # print(f"y: {y}")
            # print(f"y_stacked: {y_stacked}")

            plt.plot(n_evals, hv, marker='.', linestyle='--', label='run ' + str(key_run + 1))
        
        y_mean = np.sum(y_stacked, axis=0) / np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(run_paths)):
                if np.all(y_stacked[j, i]):
                    n += 1
                    deviation = y_stacked[j, i] - y_mean[i]
                    square_sum += deviation ** 2
            if n <= 1:
                variance = 0
            else:
                variance = square_sum / (n - 1)
            standard_deviation = np.sqrt(variance)
            if error_mean:
                y_error.append(standard_deviation / np.sqrt(n))
            else:
                y_error.append(standard_deviation)

        x_plot = np.arange(step, min_num_evals, step=step) 
        plt.plot(x_plot, y_mean[0:len(x_plot)], color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        Path(save_folder).mkdir(parents=True, exist_ok=True)   

        if metric_name_label is None:
            metric_name_label = metric_name_load.upper()

        plt.title(f"{metric_name_label} Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel(f"{metric_name_label}")
        plt.savefig(save_folder + f'{metric_name_label}_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)
        plot_array.append([x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)]])

    return plot_array

def plot_combined_hypervolume_lin_analysis(run_paths_array, save_folder):
    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        if len(run_paths) == 0:
            print("Path list is empty")
            return

        f = plt.figure(figsize=(7, 5))
        plt.title(f"Performance Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel("Hypervolume")

        n_runs = len(run_paths)
        hv_run = []
        evals_run = []

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, n_runs)]

        for ind, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER, "hv")
            n_evals, hv = eval_res.steps, eval_res.values
            plt.plot(n_evals, hv, marker='o', linestyle=":", linewidth=0.5, markersize=3, color=colors[ind],
                     label=f"run: " + str(ind + 1))
            hv_run.append(hv)
            evals_run.append(n_evals)

        def get_interpol_value(pos, n_evals, hv):
            # print(hv)
            # print(n_evals)
            for ind, eval in enumerate(n_evals):
                if n_evals[ind] > pos:
                    if ind == 0:
                        value = (hv[ind]) / 2
                    else:
                        diff = pos - n_evals[ind - 1]
                        grad = (hv[ind] - hv[ind - 1]) / (n_evals[ind] - n_evals[ind - 1])
                        value = hv[ind - 1] + grad * diff
                    return value

        step = 10
        last_n_evals = [n_evals[-1] for n_evals in evals_run]
        # n_steps = floor(min(last_n_evals)/step)
        last_n_eval = min(last_n_evals)
        n_evals_comb = np.arange(0, last_n_eval, step)

        for ind in range(0, n_runs):
            n_evals = evals_run[ind]
            hv = hv_run[ind]
            hv_run[ind] = [get_interpol_value(val, n_evals, hv) for val in n_evals_comb]
            # print(hv_comb_all)

        hv_comb_all = np.sum(hv_run, axis=0)
        hv_comb = np.asarray(hv_comb_all) / n_runs

        plt.plot(n_evals_comb, hv_comb, marker='o', linewidth=1, markersize=3, color='black', label=f"combined")

        plt.legend(loc='best')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(save_folder + 'hypervolume_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)

def write_analysis_results(result_runs_all, save_folder):
    evaluations_all = {}
    evaluations_all_no_duplicates = {}

    n_critical_all = {}
    n_critical_all_no_duplicates = {}

    mean_critical_all = {}
    mean_critical_all_no_duplicates = {}

    mean_evaluations_all = {}
    mean_evaluations_all_no_duplicates = {}

    ratio_critical_single_run = {}
    ratio_critical_single_run_no_duplicates = {}
    # temporary
    algo1 = list(result_runs_all.keys())[0]
    algo2 = list(result_runs_all.keys())[1]

    for algo, result_runs in result_runs_all.items():
        if len(result_runs) == 0:
            print("Result list is empty")
            return
        n_runs = len(result_runs)

        # n evaluations analysis
        n_evals = [res.algorithm.evaluator.n_eval for res in result_runs]
        min_n_evals = np.min(n_evals)
        max_n_evals = np.max(n_evals)
        mean_n_evals = np.sum(n_evals) / len(result_runs)

        # time analysis
        exec_time = [res.exec_time for res in result_runs]
        min_exec_time = np.min(exec_time)
        max_exec_time = np.max(exec_time)
        mean_exec_time = np.sum(exec_time) / len(result_runs)

        # criticality analysis
        evaluations_all[algo] = [res.obtain_all_population() for res in result_runs]
        evaluations_all_no_duplicates[algo] = [duplicate_free(res.obtain_all_population()) for res in result_runs]


        n_critical_all[algo] = np.asarray(
            [len(evals.divide_critical_non_critical()[0]) for evals in evaluations_all[algo]], dtype=object)
        n_critical_all_no_duplicates[algo] = np.asarray(
            [len(evals.divide_critical_non_critical()[0]) for evals in evaluations_all_no_duplicates[algo]], dtype=object)

        mean_critical_all[algo] = np.sum(n_critical_all[algo])  # /len(result_runs)
        mean_critical_all_no_duplicates[algo] = np.sum(n_critical_all_no_duplicates[algo])  # /len(result_runs)

        mean_evaluations_all[algo] = np.sum(len(evals) for evals in evaluations_all[algo])  # /len(result_runs)
        mean_evaluations_all_no_duplicates[algo] = np.sum(len(evals) for evals in evaluations_all_no_duplicates[algo])  # /len(result_runs)
        
        ratio_critical_single_run[algo] = [ n_critical_all[algo][i] / len(evaluations_all[algo][i]) for i in range(0,n_runs)]
        ratio_critical_single_run_no_duplicates[algo] = [ n_critical_all_no_duplicates[algo][i] / len(evaluations_all_no_duplicates[algo][i]) for i in range(0,n_runs)]

        '''Output of summery of the performance'''
        with open(save_folder + f'analysis_results_{algo}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Attribute', 'Value']
            write_to.writerow(header)
            write_to.writerow(['min_n_evals', min_n_evals])
            write_to.writerow(['max_n_evals', max_n_evals])
            write_to.writerow(['mean_n_evals', mean_n_evals])

            write_to.writerow(['min_exec_time [s]', min_exec_time])
            write_to.writerow(['max_exec_time [s]', max_exec_time])
            write_to.writerow(['mean_exec_time [s]', mean_exec_time])

            # write_to.writerow(['Mean number critical Scenarios', len(mean_critical)])
            # write_to.writerow(['Mean evaluations all scenarios', len(mean_evaluations_all)])
            # write_to.writerow(['Mean ratio critical/all scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
            # write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])
            f.close()

    ratio_critical_both = np.sum(n_critical_all[algo2]) if np.sum(n_critical_all[algo1]) == 0 \
                            else np.sum(n_critical_all[algo2]) / np.sum(n_critical_all[algo1]) 

    ratio_critical_both_no_duplicates = np.sum(n_critical_all_no_duplicates[algo2]) if np.sum(n_critical_all_no_duplicates[algo1]) == 0 \
                            else np.sum(n_critical_all_no_duplicates[algo2]) / np.sum(n_critical_all_no_duplicates[algo1]) 
    
    ratio_critical_both_average = np.sum(ratio_critical_single_run[algo2]) / np.sum(ratio_critical_single_run[algo1]) 
    ratio_critical_both_average_no_duplicates = np.sum(ratio_critical_single_run_no_duplicates[algo2]) / np.sum(ratio_critical_single_run_no_duplicates[algo1]) 

    '''Output of summery of the performance'''
    with open(save_folder + f'analysis_combined.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        write_to.writerow([f'Critical Scenarios {algo1}', np.sum(n_critical_all[algo1])])
        write_to.writerow([f'Critical Scenarios {algo1} (duplicate free)', np.sum(n_critical_all_no_duplicates[algo1])])

        write_to.writerow([f'Critical Scenarios {algo2}', np.sum(n_critical_all[algo2])])
        write_to.writerow([f'Critical Scenarios {algo2} (duplicate free)', np.sum(n_critical_all_no_duplicates[algo2])])

        write_to.writerow([f'Ratio Critical Scenarios {algo2}/{algo1} (union)', '{0:.2f}'.format(ratio_critical_both)])
        write_to.writerow([f'Ratio Critical Scenarios {algo2}/{algo1} (union, duplicate free)', '{0:.2f}'.format(ratio_critical_both_no_duplicates)])

        write_to.writerow([f'Ratio Critical Scenarios {algo2}/{algo1} (average)', '{0:.2f}'.format(ratio_critical_both_average)])
        write_to.writerow([f'Ratio Critical Scenarios {algo2}/{algo1} (average, duplicate free)', '{0:.2f}'.format(ratio_critical_both_average_no_duplicates)])

        # write_to.writerow([f'Mean evaluations all scenarios {algo1}', mean_evaluations_all[algo1]])
        # write_to.writerow([f'Mean evaluations all scenarios {algo2}', mean_evaluations_all[algo2]])
        # write_to.writerow([f'Mean critical all {algo1}', '{0:.2f}'.format(mean_critical_all[algo1])])
        # write_to.writerow([f'Mean critical all {algo2}', '{0:.2f}'.format(mean_critical_all[algo2])])
        # write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])

        f.close()


def read_metric_single(filename, metric_name):
    table = pd.read_csv(filename, names=["n_eval", metric_name])
    n_evals = np.array(table["n_eval"][1:].values.tolist(), dtype=float)
    hv = np.array(table[metric_name][1:].values.tolist(), dtype=float)
    return n_evals, hv

def read_pf_single(filename, with_critical_column = False):
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
        if with_critical_column:
            F = table.iloc[i, n_var + 1:].to_numpy()
        else:
            F = table.iloc[i, n_var + 1:-1].to_numpy()

        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        individuals.append(ind)
    return Population(individuals=individuals)


def make_comparison_single(max_evaluations, save_folder, subplot_metrics, subplot_names, algo_names, suffix=""):
    font = {'family':'sans-serif', 'size': 17, 'weight':'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 35

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    step = 200

    algos = algo_names
    colors = ['#9a226a','#1347ac']
    n_subplots = len(subplot_metrics)
    fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
    for i in range(len(algos)):
        for key, metric in enumerate(subplot_metrics):
            # plt.subplot(n_subplots, 1, key + 1)
            ax[key].plot(metric[i][0], metric[i][1], color=colors[i], marker=None, lw=1)
            ax[key].errorbar(metric[i][0], metric[i][1], metric[i][2], ecolor=colors[i], fmt='.k', capsize=5)
            ax[key].set_xticks(np.arange(0, stop + step, step))
            ax[key].set_xlim([0, stop])
            ax[key].locator_params(tight=True, axis='y', nbins=4)
            if key != 2:
                ax[key].xaxis.set_ticklabels([])

    for i in range(n_subplots):
        ax[i].set_ylabel(subplot_names[i], **font)
        # ax[i].xaxis.set_minor_locator()
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(which='major', axis='y', length=7)
        ax[i].tick_params(which='minor', axis='y', length=4)
        ax[i].tick_params(which='minor', axis='x', length=0)
    ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
    ax[n_subplots - 1].legend(algos, loc='best')


    ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
    ax[n_subplots - 1].set_xlim([0, stop])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.clf()
    plt.close(fig)


def make_comparison_plot(max_evaluations, save_folder, subplot_metrics, subplot_names, algo_names, distance_tick, suffix=""):
    font = {'family':'sans-serif', 'size': 16, 'weight':'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 15
    size_title = 12
    size_ticks = 10

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    step = distance_tick

    algos = algo_names
    colors = ['#9a226a','#1347ac']
    n_subplots = len(subplot_metrics)

    # only one metric to plot
    if n_subplots == 1:    
        fig, ax = plt.subplots(n_subplots, figsize=(7,5))
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                ax.plot(metric[i][0], metric[i][1], color=colors[i], marker=None, lw=1)
                ax.errorbar(metric[i][0], metric[i][1], metric[i][2], ecolor=colors[i], fmt='.k', capsize=5)
                ax.set_xticks(np.arange(0, stop + step, step))
                ax.set_xlim([0, stop])
                ax.locator_params(tight=True, axis='y', nbins=4)
                
        ax.set_ylabel(subplot_names[0], **font)
        # ax[i].xaxis.set_minor_locator()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', axis='y', length=7, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='y', length=4, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='x', length=0, labelsize=size_ticks)

        ax.set_xlabel("Number of Evaluations", **font)
        ax.title.set_size(size_title)
        ax.legend(algos, loc='best')
        ax.set_xticks(np.arange(0, stop, step))
        ax.set_xlim([0, stop])
    else:
        # several metrics to plot
        fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                # plt.subplot(n_subplots, 1, key + 1)
                ax[key].plot(metric[i][0], metric[i][1], color=colors[i], marker=None, lw=1)
                ax[key].errorbar(metric[i][0], metric[i][1], metric[i][2], ecolor=colors[i], fmt='.k', capsize=5)
                ax[key].set_xticks(np.arange(0, stop + step, step))
                ax[key].set_xlim([0, stop])
                ax[key].locator_params(tight=True, axis='y', nbins=4)
                if key != 2:
                    ax[key].xaxis.set_ticklabels([])

        for i in range(n_subplots):
            ax[i].set_ylabel(subplot_names[i], **font)
            # ax[i].xaxis.set_minor_locator()
            ax[i].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i].tick_params(which='major', axis='y', length=7)
            ax[i].tick_params(which='minor', axis='y', length=4)
            ax[i].tick_params(which='minor', axis='x', length=0)

        ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
        ax[n_subplots - 1].legend(algos, loc='best')


        ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
        ax[n_subplots - 1].set_xlim([0, stop])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    Path(save_folder).mkdir(parents=True, exist_ok=True)   

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.savefig(save_folder + f'subplots_combined{suffix}.pdf', format = "pdf")

    plt.clf()
    plt.close(fig)


def make_subplots(max_evaluations, save_folder, subplot_metrics, subplot_names, algo_names, suffix=""):
    font = {'family':'sans-serif', 'size': 17, 'weight':'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 35

    offset_x_ax = 0.1 * max_evaluations
    stop = max_evaluations + offset_x_ax
    step = 200

    algos = algo_names
    colors = ['#9a226a','#1347ac']
    n_subplots = len(subplot_metrics)
    fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
    for i in range(len(algos)):
        for key, metric in enumerate(subplot_metrics):
            # plt.subplot(n_subplots, 1, key + 1)
            ax[key].plot(metric[i][0], metric[i][1], color=colors[i], marker=None, lw=1)
            ax[key].errorbar(metric[i][0], metric[i][1], metric[i][2], ecolor=colors[i], fmt='.k', capsize=5)
            ax[key].set_xticks(np.arange(0, stop + step, step))
            ax[key].set_xlim([0, stop])
            ax[key].locator_params(tight=True, axis='y', nbins=4)
            if key != 2:
                ax[key].xaxis.set_ticklabels([])

    for i in range(n_subplots):
        ax[i].set_ylabel(subplot_names[i], **font)
        # ax[i].xaxis.set_minor_locator()
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].tick_params(which='major', axis='y', length=7)
        ax[i].tick_params(which='minor', axis='y', length=4)
        ax[i].tick_params(which='minor', axis='x', length=0)
    ax[n_subplots - 1].set_xlabel("Number of Evaluations", **font)
    ax[n_subplots - 1].legend(algos, loc='best')


    ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
    ax[n_subplots - 1].set_xlim([0, stop])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.clf()
    plt.close(fig)

''' 
    Write plot idge plot data in csv form. (TODO make more generic, i.e. independent of number of metrics)
'''
def write_metric_data_to_csv_igde(save_folder, metric_name, algo_names, plot_array, suffix=""):
    algo_1 = algo_names[0]
    algo_2 = algo_names[1]

    m = metric_name

    header_igd = [f'{algo_1}_n_evals', 
                  f'{algo_1}_{m}', 
                  f'{algo_1}_{m}_sigma', 
                  #f'{algo_1}__{m}_min', 
                  #f'{algo_1}__{m}_max', 
                  f'{algo_2}__n_evals', 
                  f'{algo_2}__{m}',
                  f'{algo_2}__{m}_sigma', 
                  #f'{algo_2}__{m}_min',
                  #f'{algo_2}__{m}_max'
    ]
    # metric_names = ['hv', 'gd', 'sp']
    paths = []
    metric = plot_array
    filename = save_folder + metric_name + os.sep + 'combined_' + metric_name + suffix +  '.csv'
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        write_to.writerow(header_igd)
        for i in range(len(metric[0][0])):
            write_to.writerow([metric[0][0][i],  # algo1
                               metric[0][1][i], 
                               metric[0][2][i],
                               #metric[0][3][i],
                               #metric[0][4][i], 
                               metric[1][0][i],  # algo2
                               metric[1][1][i], 
                               metric[1][2][i],
                               #metric[1][3][i], 
                               #metric[1][4][i]
                               ])
        f.close()
        paths.append(filename)

    return paths


def write_metric_data_to_csv(save_folder, metric_names, algo_names, plot_array_hv, plot_array_igd, plot_array_sp, suffix=""):
    algo_1 = algo_names[0]
    algo_2 = algo_names[1]
    
    # m_1 = metric_names[0]
    # m_2 = metric_names[1]
    # m_3 = metric_names[2]

    headers = []
    for m in metric_names:
        header =  [f'{algo_1}_n_evals', f'{algo_1}_{m}', f'{algo_1}_{m}_sigma', f'{algo_2}__n_evals', f'{algo_2}__{m}',
                 f'{algo_2}__{m}_sigma']
        headers.append(header)

    # header_hv = [f'{algo_1}_n_evals', f'{algo_1}_{m_1}', f'{algo_1}_{m_1}_sigma', f'{algo_2}__n_evals', f'{algo_2}__{m_1}',
    #              f'{algo_2}__{m_1}_sigma']
    # header_igd = [f'{algo_1}_n_evals', f'{algo_1}_{m_2}', f'{algo_1}_{m_2}_sigma', f'{algo_2}__n_evals', f'{algo_2}__{m_2}',
    #               f'{algo_2}__{m_2}_sigma']
    # header_sp = [f'{algo_1}_n_evals', f'{algo_1}_{m_3}', f'{algo_1}_{m_3}_sigma', f'{algo_2}__n_evals', f'{algo_2}__{m_3}',
    #              f'{algo_2}__{m_3}_sigma']
    # headers = [header_hv, header_igd, header_sp]
    # metric_names = ['hv', 'gd', 'sp']

    paths = []

    for key, metric in enumerate([plot_array_hv, plot_array_igd, plot_array_sp]):
        filename = save_folder + metric_names[key] + os.sep + 'combined_' + metric_names[key] + suffix +  '.csv'
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            write_to.writerow(headers[key])
            for i, value in enumerate(metric[0][0]):
                write_to.writerow([metric[0][0][i], metric[0][1][i], metric[0][2][i], metric[1][0][i], metric[1][1][i], metric[1][2][i]])
            f.close()
            paths.append(filename)

    return paths

def retrieve_metric_data_from_csv(paths):
    storing_arrays = []

    # plot_array_hv = []
    # plot_array_igd = []
    # plot_array_sp = []

    # storing_arrays = [plot_array_hv, plot_array_igd, plot_array_sp]

    for key, path in enumerate(paths):
        table = pd.read_csv(path)
        metric_nsga_two = [table.iloc[:,0].values.tolist(), table.iloc[:,1].values.tolist(), table.iloc[:,2].values.tolist()]
        metric_nsga_two_dt = [table.iloc[:,3].values.tolist(), table.iloc[:,4].values.tolist(), table.iloc[:,5].values.tolist()]
        storing_arrays.append([metric_nsga_two, metric_nsga_two_dt])
    return storing_arrays #plot_array_hv, plot_array_igd, plot_array_sp
