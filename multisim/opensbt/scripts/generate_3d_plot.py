from pathlib import Path
import numpy as np
from pymoo.visualization.scatter import Scatter
import os
import pandas as pd
from opensbt.model_ga.individual import Individual
from opensbt.model_ga.population import Population
from opensbt.utils.sorting import get_nondominated_population
import logging as log
from opensbt.visualization.configuration import *
from opensbt.utils.duplicates import duplicate_free
from matplotlib import pyplot as plt
from datetime import datetime
from opensbt.evaluation.critical import *

# 3d plot colors (taken from model example plots in metrics paper SSBSE23)
color_reference_set_edge = "#f84394"
color_reference_set_fill = "#ffffff"
color_critical_set_edge = "#ef2222"
color_critical_set_fill = "#fab7b7"
    
plt.rcParams.update({'axes.titlesize': 'small'})
plt.rcParams.update({'axes.labelsize': 'large'})
plt.rcParams['xtick.labelsize']=25
plt.rcParams['ytick.labelsize']=25
# plt.rcParams['xtick.major.size'] = 10
# plt.rcParams['xtick.major.width'] = 4
# plt.rcParams['xtick.minor.size'] = 10
# plt.rcParams['xtick.minor.width'] = 4

def read_testcases(filename, critical_fnc: Critical = None):
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
    print(f"length of table: {len(table)}")
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        CB = table.iloc[i, -1]
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        if critical_fnc is not None:
            ind.set("CB", critical_fnc.eval(ind.get("F")))
        else:
            ind.set("CB", CB)
        individuals.append(ind)
    
    print("Csv file successfully read")
    return Population(individuals=individuals), var_names

# make 3d design space plot
def visualize_3d(population, save_folder, labels, mode="critical", markersize=20, axis_limits = None, orientations=[(45,-45)], do_save=False):
    save_folder_design = save_folder + "design_space" + os.sep

    pop = duplicate_free(population)
    print(f"Number all: {len(pop)}")

    opt = get_nondominated_population(population)
    print(f"Number optimal: {len(opt)}")

    X, F = pop.get("X", "F")
    CB = np.array(pop.get("CB"),dtype=bool)
    X_opt, F_opt= opt.get("X", "F")
    CB_opt = np.array(opt.get("CB"),dtype=bool)

    X_plus_opt = np.ma.masked_array(X_opt, mask=np.dstack([np.invert(CB_opt)] * n_var))
    X_minus_opt = np.ma.masked_array(X_opt, mask=np.dstack([CB_opt] * n_var))

    mask_plus = np.invert(CB)
    # print(f"mask_plus_len:{len(mask_plus)}")

    mask_minus = CB
    # print(f"mask_minus_len:{len(mask_minus)}")

    X_plus = np.ma.masked_array(X, mask=np.dstack([mask_plus] * n_var))
    # print(X_plus)

    X_minus = np.ma.masked_array(X, mask=np.dstack([mask_minus] * n_var))
    # print(X_minus)

    if do_save:
        Path(save_folder_design).mkdir(parents=True, exist_ok=True)   

        # select the view(s)
        angles = orientations

        for angle in angles:
            #plot_des = Scatter(title='Design Space', labels = labels, angle=angle)
            plot_des = Scatter(labels = labels, angle=angle)
            if np.ma.count(X_plus, axis=0)[0] != 0:
                # plot_des.add(X_plus, facecolor=color_critical_set_fill, edgecolor=color_critical_set_edge, s=markersize)
                plot_des.add(X_plus, facecolor=color_reference_set_fill, edgecolor=color_reference_set_edge, s=markersize)
                if mode == 'all':
                    if np.ma.count(X_minus, axis=0)[0] != 0:
                        plot_des.add(X_minus, facecolor=color_not_optimal, edgecolor=color_critical_set_edge, s=markersize)

            if mode != "critical":    
                if np.ma.count(X_minus_opt, axis=0)[0] != 0:
                        plot_des.add(X_minus_opt, facecolor=color_optimal, edgecolor=color_critical_set_edge, s=markersize)
        
            if mode=="opt" or mode== "all":
                if np.ma.count(X_plus_opt, axis=0)[0] != 0:
                    print("added optimal and critical")
                    plot_des.add(X_plus_opt, facecolor=color_optimal, edgecolor=color_critical_set_edge, s=markersize)

            plot_des.show()
           
            fig = plot_des.fig
            ax = plot_des.ax

            if axis_limits is not None:
                ls = axis_limits

                ax.set_xlim(ls[0][0], ls[0][1])
                ax.set_ylim(ls[1][0], ls[1][1])
                ax.set_zlim(ls[2][0], ls[2][1])

                print("Axis limits set.")

            plot_des.save(save_folder_design + f"design_space_3d_angle{angle}.png")
            plot_des.save(save_folder_design + f"design_space_3d_angle{angle}.pdf",format="pdf")
    else:
        for angle in angles:
            #plot_des = Scatter(title='Design Space', labels = labels, angle=angle)
            plot_des = Scatter(labels = labels, angle=angle)

            if np.ma.count(X_plus, axis=0)[0] != 0:
                plot_des.add(X_plus, facecolor=color_not_optimal, edgecolor=color_critical_set_edge, s=markersize)
                if mode == 'all':
                    if np.ma.count(X_minus, axis=0)[0] != 0:
                        plot_des.add(X_minus, facecolor=color_not_optimal, edgecolor=color_critical_set_edge, s=markersize)

            if mode != "critical":    
                if np.ma.count(X_minus_opt, axis=0)[0] != 0:
                        plot_des.add(X_minus_opt, facecolor=color_optimal, edgecolor=color_critical_set_edge, s=markersize)
        
            if mode=="opt" or mode== "all":
                if np.ma.count(X_plus_opt, axis=0)[0] != 0:
                    print("added optimal and critical")
                    plot_des.add(X_plus_opt, facecolor=color_optimal, edgecolor=color_critical_set_edge, s=markersize)

            plot_des.show()

'''
  read in csv file of test cases and create 3d plot for test inputs 
  three modes:
    'opt' : plot optimal and critical
    'crit' : plot all critical
    'all': plot all
'''

file = "final" + os.sep + "testcases" + os.sep + "gs_25" + os.sep + "all_testcases.csv"
file_rs10 = "final" + os.sep + "testcases" + os.sep + "gs_10" + os.sep + "all_testcases.csv"
file_rs20 = "final" + os.sep + "testcases" + os.sep + "gs_20" + os.sep + "all_testcases.csv"
file_rs25 = "final" + os.sep + "testcases" + os.sep + "gs_25" + os.sep + "all_testcases.csv"

file_fps500 = "final" + os.sep + "testcases" + os.sep + "fps_500" + os.sep + "all_testcases.csv"
file_fps1000 = "final" + os.sep + "testcases" + os.sep + "fps_1000" + os.sep + "all_testcases.csv"
file_fps8000 = "final" + os.sep + "testcases" + os.sep + "fps_8000" + os.sep + "all_testcases.csv"

file_run5_rs = "final" + os.sep + "testcases" + os.sep + "run_5" + os.sep + "RS.csv"
file_run5_nsga2 = "final" + os.sep + "testcases" + os.sep + "run_5" + os.sep + "NSGAII.csv"

#file = "final" + os.sep + "testcases" + os.sep + "sampling_rs_25" + os.sep + "test.csv"
#file = "final" + os.sep + "testcases" + os.sep + "fps_1000" + os.sep + "all_testcases.csv"

files = [file_run5_rs,
        file_run5_nsga2,
        file_rs25]

limits = [(0.1, 1), (0.5, 2.0), (0.0, 5.0)]

orientations = [(45,-45)]#(45,45),(45,135),(45,225)]#(0,0),(0,90),(0,180),(0,270)]

labels = ["Ego_Velocity_Gain", "Ped_Velocity", "Ped_Wait"]

#critical_fnc = CriticalAdasAdaptedDistanceVelocityContrained()
critical_fnc = CriticalAdasAdaptedDistanceVelocityContrainedMax()

for file in files:
    main_folder = file.split(os.sep)[0]

    # file = "results_avp" + os.sep + "all_individuals_small.csv"
    save_folder = os.getcwd() + os.sep + "scripts" + os.sep + "output" + os.sep+ "3d_plot" + os.sep + main_folder + os.sep + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + os.sep

    ###########
    data = os.getcwd() + os.sep + "scripts" + os.sep +  "data" + os.sep +  file
    pop, _ = read_testcases(data, critical_fnc=critical_fnc)
    n_var = len(labels)
    visualize_3d(pop,save_folder,labels, axis_limits=limits, orientations=orientations, do_save=True)
