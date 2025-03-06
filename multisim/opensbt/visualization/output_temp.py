

''' Updated for testing algorithm'''

import os
from pathlib import Path

from matplotlib import patches, pyplot as plt
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.visualization.output import HandlerCircle, create_markers
from opensbt.visualization.output import *

#TODO merge with existing functions in output.py (functions here do not use result object)

def objective_space(problem, population, save_folder):
    save_folder_objective = save_folder + "objective_space" + os.sep
    Path(save_folder_objective).mkdir(parents=True, exist_ok=True)   
    save_folder_plot = save_folder_objective

    pf = problem.pareto_front()
    n_obj = problem.n_obj
    objective_names = problem.objective_names

    critical, not_critical = population.divide_critical_non_critical()

    critical_clean = duplicate_free(critical)
    not_critical_clean = duplicate_free(not_critical)

    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_obj - 1):
        for axis_y in range(axis_x + 1, n_obj):
            ax = plt.subplot(111)
            plt.title("Objective Space" + " (" + str(len(population)) + " individuals, " + str(len(critical)) + " of which are critical)")

            if pf is not None:
                ax.plot(pf[:, axis_x], pf[:, axis_y], color='blue', lw=0.7, zorder=1)

            if len(not_critical) != 0:
                ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
            if len(critical) != 0:
                ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_critical, marker='o')
            
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
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={patches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.png')
            plt.clf()
            
    plt.close(f)

def plot_critical_all(problem, population, save_folder):
    save_folder_design = save_folder
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    design_names = problem.design_names
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    all_population = population
    critical_all = population
    
    # clean up
    plt.clf()

    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_var - 1):
        for axis_y in range(axis_x + 1, n_var):
            ax = plt.subplot(111)
            plt.title("Design Space" + " (" + str(len(all_population)) + " individuals, " + str(len(critical_all)) + " of which are critical)")

            critical_all = duplicate_free(critical_all)
            
            ax.scatter(critical_all.get("X")[:, axis_x], critical_all.get("X")[:, axis_y], s=40,
                            facecolors=color_optimal,
                            edgecolors=color_critical, marker='o')

            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers()
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.png')
            plt.clf()

    plt.close(f)

    ''' Updated for testing algorithm'''

def design_space(problem, population, best_population, save_folder, classification_type=ClassificationType.DT, suffix=""):
    save_folder_design = save_folder + "design_space" + os.sep
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    design_names = problem.design_names
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    all_population = population
    critical, not_critical = all_population.divide_critical_non_critical()

    critical_clean = duplicate_free(critical)
    not_critical_clean = duplicate_free(not_critical)

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
                    plt.gca().add_patch(patches.Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor=region_color, lw=1.5, ls='-',
                                                  facecolor='none', alpha=0.2))
                    plt.gca().add_patch(patches.Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor='none',
                                                  facecolor=region_color, alpha=0.05))

            ax = plt.subplot(111)
            plt.title("Design Space" + " (" + str(len(all_population)) + " individuals, " + str(len(critical)) + " of which are critical)")

            if best_population != []:
                critical, not_critical = best_population.divide_critical_non_critical()  # classification of optimal individuals

                if len(not_critical) != 0:
                    ax.scatter(not_critical.get("X")[:, axis_x], not_critical.get("X")[:, axis_y], s=40,
                                facecolors=color_optimal,
                                edgecolors=color_not_critical, marker='o')

                if len(critical_clean) != 0:
                    ax.scatter(critical_clean.get("X")[:, axis_x], critical_clean.get("X")[:, axis_y], s=40,
                                facecolors=color_optimal,
                                edgecolors=color_critical, marker='o')

            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers()
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + suffix + '.png')
            plt.clf()

    plt.close(f)