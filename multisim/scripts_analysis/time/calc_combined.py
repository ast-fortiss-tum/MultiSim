import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
import csv
import pandas as pd

"""value_pairs_runs:

    { 
        algo : [ 
                    [ [ 10, 2], [20, 4], ... ]  # run 1, 
                    [ [ 10, 0], [20, 2], ... ]  # run 2
                    ...
        ]
    }
"""
def calc_average_over_runs(metric_name, 
                                metric_values_array_algo, 
                                save_folder, 
                                n_func_evals_lim, 
                                n_fitting_points, 
                                metric_name_label=None, 
                                step_chkp=None, 
                                error_mean=False,
                                is_percentage = False):
        
    plot_array = []

    for key, (algo, runs_metric_values) in enumerate(metric_values_array_algo.items()):
        num_evals_limit = []

        for run_metric_values in runs_metric_values:
            n_evals = [v[0] for v in run_metric_values]
            values = [v[1] for v in run_metric_values]
            num_evals_limit.append(n_evals[-1])
            
        if is_percentage:
            min_num_evals = 1.1
            step = round(min_num_evals / n_fitting_points, 2)
        else:
            min_num_evals = n_func_evals_lim + 1
            step = min_num_evals // n_fitting_points

        if step_chkp is not None:
            step = step_chkp 
        x = np.arange(step, min_num_evals, step=step)
        y_stacked = np.zeros((len(runs_metric_values), len(x)))

        f = plt.figure(figsize=(7, 5))
        
        for key_run, run_metric_values in enumerate(runs_metric_values):
            n_evals = [v[0] for v in run_metric_values]
            values = [v[1] for v in run_metric_values]
            spl = scipy.interpolate.interp1d(np.array(n_evals), np.array(values),fill_value="extrapolate")
            x_run = np.arange(step, min_num_evals, step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y
            # print(f"y: {y}")
            # print(f"y_stacked: {y_stacked}")

            plt.plot(n_evals, values, marker='.', linestyle='--', label='run ' + str(key_run + 1))
        
        y_mean = np.sum(y_stacked, axis=0) / np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(runs_metric_values)):
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
            metric_name_label = metric_name.upper()

        plt.title(f"{metric_name_label} Analysis ({algo})")
        plt.xlabel("Evaluations")
        plt.ylabel(f"{metric_name_label}")
        plt.savefig(save_folder + f'{metric_name_label}_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)
        plot_array.append([x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)]])

    return plot_array


def make_comparison_plot(max_evaluations, 
                        save_folder, 
                        subplot_metrics, 
                        subplot_names, 
                        algo_names, 
                        distance_tick,
                        shift_error=False, 
                        suffix="",
                        colors=None,
                        cmap=None,
                        is_percentage = False,
                        plot_error = True,
                        x_title = "x_value"):

    font = {'family': 'sans-serif', 'size': 14, 'weight': 'bold'}
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.weight'] = "bold"
    matplotlib.rcParams['font.size'] = 13
    size_title = 12
    size_ticks = 10
    
    offset_x_ax = 0.1 * max_evaluations

    if is_percentage:
        stop = 1.1
        max_evaluations = 1
        if distance_tick is None:
            step = 0.1
        else:
            step = distance_tick
    else:    
        stop = max_evaluations + offset_x_ax
        if distance_tick is None:
            step = 0.1*max_evaluations
        else:
            step = distance_tick
            
    if shift_error is True:
        if is_percentage:
            offset_algo_x = 0.008
            offset_algo_y = 0.008
            marker_error = ''
        else:       
            offset_algo_x = 0.004* max_evaluations
            offset_algo_y = 0.005
            marker_error = ''
    else: 
        offset_algo_x = 0
        offset_algo_y = 0
        marker_error = '.'
        
    line_width = 3
    error_lw = 2
    error_ls = ''
    
    algos = algo_names
    if colors is None:
            colors = ['#ffbb00',  # Vibrant orange-yellow
                    '#1347ac',  # Deep blue
                    '#9a226a',  # Rich magenta
                    
                    '#666666',  # Neutral gray
                    '#c2185b',  # Raspberry pink
                    '#e30202',   # red
                    
                    '#3e4a61',  # Color 1 from viridis
                    '#6a8c7c',  # Color 2 from viridis
                    '#f5f6f3'   # Color 3 from viridis
                    ]
    line_styles = [
        
        '-',  # Solid line,
        '-',  # Solid line
        '-',  # Solid line
        
        '--', # Dashed line,
        '--', # Dashed line
        '--', # Dashed line
        
        '-.', # Dash-dot line,
        '-.', # Dash-dot line
        '-.', # Dash-dot line
        
        ':',  # Dotted line
        '-',  # Solid line (reused for wrapping)
        '--', 
        '-.', 
        ':'
    ]
    n_subplots = len(subplot_metrics)

    # only one metric to plot
    if n_subplots == 1:
        fig, ax = plt.subplots(n_subplots, figsize=(7, 5))
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                ax.plot(metric[i][0], 
                        metric[i][1],
                        color= cmap[algos[i]] if cmap is not None else colors[i], 
                        marker=None,
                        lw=1,
                        linestyle = line_styles[i])
                if plot_error:
                    ax.errorbar(np.asarray(metric[i][0]) + offset_algo_x*i, 
                            np.asarray(metric[i][1]) + offset_algo_y*i, 
                                metric[i][2], 
                                ecolor= cmap[algos[i]] if cmap is not None else colors[i], 
                                marker=marker_error,
                                fmt='.k', 
                                capsize=5,
                                linestyle=error_ls,
                                linewidth=error_lw)
                ax.set_xticks(np.arange(0, stop + step, step))
                ax.set_xlim([0, stop])
                ax.locator_params(tight=True, axis='y', nbins=4)
        
        
        ax.set_ylabel(subplot_names[0], **font)
        # ax[i].xaxis.set_minor_locator()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', axis='y', length=7, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='y', length=4, labelsize=size_ticks)
        ax.tick_params(which='minor', axis='x', length=0, labelsize=size_ticks)

        if is_percentage:
            ax.set_xlabel("Percentage Budget", **font)
        else:
            ax.set_xlabel(x_title, **font)
        ax.title.set_size(size_title)
        ax.legend([algo.upper() for algo in algos], loc='best')
        ax.set_xticks(np.arange(0, stop, step))
        ax.set_xlim([0, stop])

        # # Dynamically set integer y-axis ticks, starting at 0
        # y_min, y_max = ax.get_ylim()  # Get current y-axis limits
        # ax.set_yticks(np.arange(0, 5 * round(y_max / 5) + 1, round(y_max / 5) ))
    else:
        # several metrics to plot
        fig, ax = plt.subplots(n_subplots, 1, figsize=(9, 9))
        for i in range(len(algos)):
            for key, metric in enumerate(subplot_metrics):
                # plt.subplot(n_subplots, 1, key + 1)
                # log.info(f"metric: {metric}")
                ax[key].plot(np.asarray(metric[i][0]), 
                             np.asarray(metric[i][1]),
                             color= cmap[algos[i]] if cmap is not None else colors[i],
                             marker=None, 
                             lw=line_width,
                             linestyle=line_styles[i])
                if plot_error:
                    ax[key].errorbar(
                                    np.asarray(metric[i][0])+ offset_algo_x*i, 
                                    np.asarray(metric[i][1]) + offset_algo_y*i, 
                                    metric[i][2], 
                                    marker=marker_error,
                                    ecolor= cmap[algos[i]] if cmap is not None else colors[i], 
                                    fmt='.k', 
                                    capsize=5,
                                    linestyle=error_ls,
                                    linewidth=error_lw)
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
        if is_percentage:
            ax[n_subplots - 1].set_xlabel("Percentage Budget", **font)
        else:
            ax[n_subplots - 1].set_xlabel(x_title, **font)
            
        ax[n_subplots - 1].legend([algo.upper() for algo in algos], loc='best', framealpha=0.4)

        ax[n_subplots - 1].set_xticks(np.arange(0, stop+step, step))
        ax[n_subplots - 1].set_xlim([0, stop])

        # y_min, y_max =  ax[n_subplots - 1].get_ylim()  # Get current y-axis limits
        # ax[n_subplots - 1].set_yticks(np.arange(0, 5 * round(y_max / 5) + 1,  round(y_max / 5) ))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    plt.savefig(save_folder + f'subplots_combined{suffix}.png')
    plt.savefig(save_folder + f'subplots_combined{suffix}.pdf', format="pdf")

    plt.clf()
    plt.close(fig)

def write_metric_data_to_csv(save_folder, metric_names, algo_names, plot_array_metric, suffix=""):
    headers = []
    for m in metric_names:
        header = []
        for algo in algo_names:
            header += [f'{algo}_n_evals', f'{algo}_{m}', f'{algo}_{m}_sigma']
        headers.append(header)

    paths = []

    for key, metric in enumerate([plot_array_metric]):
        filename = save_folder + \
            metric_names[key] + os.sep + 'combined_' + \
            metric_names[key] + suffix + '.csv'
        Path(save_folder + \
            metric_names[key]).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)
            write_to.writerow(headers[key])
            for i, _ in enumerate(metric[0][0]):
                res_algo = []
                for j, _ in enumerate(algo_names):
                    res_algo += [metric[j][0][i],
                                 metric[j][1][i], metric[j][2][i]]
                write_to.writerow(res_algo)
            f.close()
            paths.append(filename)

    return paths

def retrieve_metric_data_from_csv(paths, n_algos):
    storing_arrays = []

    # plot_array_hv = []
    # plot_array_igd = []
    # plot_array_sp = []

    # storing_arrays = [plot_array_hv, plot_array_igd, plot_array_sp]

    for key, path in enumerate(paths):
        table = pd.read_csv(path)
        by_metric = []
        for i in range(n_algos):
            metric_algo = [
                        table.iloc[:, 3*i].values.tolist(), 
                        table.iloc[:,3*i + 1].values.tolist(), 
                        table.iloc[:, 3*i + 2].values.tolist()
            ]
            by_metric.append(metric_algo)
        storing_arrays.append(by_metric)
    return storing_arrays  # plot_array_hv, plot_array_igd, plot_array_sp

