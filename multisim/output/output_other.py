import copy
import csv
import json
import math
import os
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from features import opensbt_feature
from features.feature_map import FeatureMap
from model_ga.population import PopulationExtended
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.quality_indicators.metrics.igde import IGDE
from opensbt.utils.sampling import CartesianSampling
from problem.adas_problem import ADASProblem
from visualization import plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from analysis.quality import Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log
from opensbt.visualization.visualization3d import visualize_3d
import shutil
import config
from config import CONSIDER_HIGH_VAL_OS_PLOT, LAST_ITERATION_ONLY_DEFAULT, OUTPUT_PRECISION, PENALTY_MAX, PENALTY_MIN
from opensbt.visualization.output import BACKUP_FOLDER, METRIC_PLOTS_FOLDER, write_metric_history

def calculate_avg_euclid(res: Result, 
                             save_folder: str, 
                             critical_only = True,
                             var = "X",
                             title_suffix = ""):

    log.info(f"------ Calculating Avg Euclidean Distance ({var}) ------")

    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_avg_euclid(res, 
                                            critical_only=critical_only,
                                            var = var) 
    if eval_result is None:
        log.info("No avg euclidean distance computed")
        return
    
    n_evals, cid = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, cid, f"avg_euclid_{var}_{'C' if critical_only else ''}",save_folder)

    f = plt.figure()
    plt.plot(n_evals, cid, color='black', lw=0.7)
    plt.scatter(n_evals, cid, facecolor="none", edgecolor='black', marker="o")
    plt.title(f"Diversity Analysis ({var})\n{res.problem.problem_name}")
    plt.xlabel("Function Evaluations")
    plt.ylabel(f"Avg Euclidean Distance")
    plt.savefig(save_folder_plot + f"avg_euclid_{var}_{'C' if critical_only else ''}{title_suffix}.png")
    plt.close()
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final avg euclid distance value: {cid[-1]}")     
    return n_evals, cid 