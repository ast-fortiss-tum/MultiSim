from pathlib import Path
import numpy as np
from pymoo.visualization.scatter import Scatter
import os
import pandas as pd
import pymoo

from opensbt.model_ga.individual import Individual
from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.utils.sorting import get_nondominated_population
import logging as log
from opensbt.visualization.configuration import *
from opensbt.utils.duplicates import duplicate_free
from matplotlib import patches, pyplot as plt
from datetime import datetime
from opensbt.visualization.output import HandlerCircle, create_markers

# 3d plot colors (taken from model example plots in metrics paper SSBSE23)
color_reference_set_edge = "#f84394"
color_reference_set_fill = "#ffffff"
color_critical_set_edge = "#ef2222"
color_critical_set_fill = "#fab7b7"
    
plt.rcParams.update({'axes.titlesize': 17})
plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
# plt.rcParams['xtick.major.size'] = 10
# plt.rcParams['xtick.major.width'] = 4
# plt.rcParams['xtick.minor.size'] = 10
# plt.rcParams['xtick.minor.width'] = 4

def read_testcases(filename):
    individuals = []
    table = pd.read_csv(filename)
    var_names = []
    n_var = -1
    k = 0
    # identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        var_names.append(col)
        k = k + 1
    print(f"n_var: {n_var}")
    print(f"Length of table: {len(table)}")
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        CB = table.iloc[i, -1]
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        ind.set("CB", CB)
        individuals.append(ind)
    
    print("Csv file successfully read")
    return PopulationExtended(individuals=individuals), var_names

def generate_object_space_plot(population, 
                               oracle,
                                limits, 
                                objective_names, 
                                n_obj, 
                                offset, 
                                pf, 
                                algorithm_name, 
                                save_folder, 
                                iteration=None):
    # save_folder_objective = save_folder + "os" + os.sep
    # Path(save_folder_objective).mkdir(parents=True, exist_ok=True)   
    # save_folder_plot = save_folder_objective
    Path(save_folder).mkdir(parents=True, exist_ok=True)   
    critical_all, non_critical_all = population.divide_critical_non_critical()
    print(f"Length critical: {len(critical_all)}")
    print(f"Length non critical: {len(non_critical_all)}")
    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_obj - 1):
        for axis_y in range(axis_x + 1, n_obj):
            ax = plt.subplot(111)
            plt.title(f"{algorithm_name} \nObjective Space" + " (" + str(len(population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

            critical, not_critical = population.divide_critical_non_critical()
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

            optimal_pop = get_nondominated_population(population)
            critical, not_critical = optimal_pop.divide_critical_non_critical()
            critical_clean = duplicate_free(critical)
            not_critical_clean = duplicate_free(not_critical)
            
            if len(not_critical_clean) != 0:
                ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
            if len(critical_clean) != 0:
                ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_critical, marker='o')

            # Limit axes bounds, since we do not want to show fitness values as 1000 or int.max, 
            # that assign bad quality to worse scenarios
            CONSIDER_HIGH_VAL = True
            if CONSIDER_HIGH_VAL:
                MAX_VALUE = 1000
                MIN_VALUE = -1000
                
                pop_f_x = population.get("F")[:,axis_x]
                clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == MAX_VALUE))
                max_x_f_ind = max(clean_pop_x)
                clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == MIN_VALUE))
                min_x_f_ind = min(clean_pop_x)

                pop_f_y = population.get("F")[:,axis_y]
                clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == MAX_VALUE))
                max_y_f_ind = max(clean_pop_y)
                clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == MIN_VALUE))
                min_y_f_ind = min(clean_pop_y)

                eta_x = abs(max_x_f_ind - min_x_f_ind) / 10
                eta_y = abs(max_y_f_ind- min_y_f_ind) / 10
                
                plt.xlim(min_x_f_ind - eta_x, max_x_f_ind  + eta_x)
                plt.ylim(min_y_f_ind - eta_y, max_y_f_ind  + eta_y)
            
            plt.xlabel(objective_names[axis_x])
            plt.ylabel(objective_names[axis_y])

            if limits is not None:
                offset_x = (- limits[0][0]  + limits[0][1])* offset
                offset_y = (- limits[1][0]  + limits[1][1])* offset
                
                plt.xlim(limits[0][0] - offset_x , limits[0][1] + offset_x )
                plt.ylim(limits[1][0] - offset_y , limits[1][1] + offset_y)
            
            if oracle is not None:
                left = oracle[1][axis_x]
                bottom = oracle[1][axis_y]
                width = - oracle[1][axis_x] + oracle[0][axis_x]  
                height = - oracle[1][axis_y] + oracle[2][axis_y] 
                rect = plt.Rectangle((left, bottom), width, height,
                     facecolor=None, 
                     edgecolor="purple",
                     alpha=0.1)
                ax.add_patch(rect)      
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            marker_list = create_markers()
            if pf is not None:
                markers = marker_list[2:]
            else:
                markers = marker_list[2:-1]

            plt.legend(handles=markers,
                    loc='best',
                    # loc='center left', 
                    # bbox_to_anchor=(1, 0.5), 
                    handler_map={patches.Circle: HandlerCircle()})

            plt.savefig(save_folder + objective_names[axis_x] + '_' + objective_names[axis_y] + '.png')
            plt.savefig(save_folder + objective_names[axis_x] + '_' + objective_names[axis_y] + '.pdf', format='pdf')

            plt.clf()
            
    plt.close(f)

#file =  "all_testcases.csv"
file1 =  "NSGA-II" + os.sep + "all_testcases.csv"
file2 =  "NLSGA-II" + os.sep + "all_testcases.csv"
file = "NSGA-II_Filtered" + os.sep + "all_testcases.csv"
files = [file]
limits = [(-1, 0), (-5, 0)]
offset = 0.05 #in per cent for each axis
objective_names = ["Adapted Distance", "Velocity (negated)"]
algorithm_name = "NLSGA-II"
oracle = [(-0.6,0),(-1, -100),(-1,0),(-0.6,0)]

for file in files:
    main_folder = file.split(os.sep)[0]
    algorithm_name = main_folder
    # file = "results_avp" + os.sep + "all_individuals_small.csv"
    save_folder = os.getcwd() + os.sep + "scripts" + os.sep + "output" + os.sep+ "os_plot"  + os.sep + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + os.sep

    ###########
    data = os.getcwd() + os.sep + "scripts" + os.sep +  "data" + os.sep +  file
    pop, _ = read_testcases(data)
    n_var = len(objective_names)
   
    generate_object_space_plot(population=pop,
                            limits=limits,
                            oracle=oracle,
                            offset=offset,
                            n_obj=2,
                            pf=None,
                            objective_names=objective_names,
                            algorithm_name=algorithm_name,
                            save_folder=save_folder
                            )