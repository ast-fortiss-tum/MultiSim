import copy
import csv
import json
import math
import os
import random
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from fitness import fitness
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.problem.adas_multi_sim_aggreement_problem_diverse import ADSMultiSimAgreementProblemDiverse
from opensbt.quality_indicators.metrics.igde import IGDE
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.visualization import plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.quality_indicators.quality import Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free, duplicate_free_with_index
import logging as log
from opensbt.visualization.visualization3d import visualize_3d
import shutil
import config
import uuid
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from config import CONSIDER_HIGH_VAL_OS_PLOT, EVALUATE_DISAGREEMENTS_PREDICT, LAST_ITERATION_ONLY_DEFAULT, OUTPUT_PRECISION, PENALTY_MAX, PENALTY_MIN, \
CRITICAL_XTE

WRITE_ALL_INDIVIDUALS = True
BACKUP_FOLDER =  "backup" + os.sep
METRIC_PLOTS_FOLDER =  "metric_plots" + os.sep

def do_disagree(ind):
    return abs(ind.get("F")[0]) <= CRITICAL_XTE and abs(ind.get("F")[1]) > CRITICAL_XTE or \
            abs(ind.get("F")[0]) > CRITICAL_XTE and abs(ind.get("F")[1]) <= CRITICAL_XTE

def evaluate_disagreement(res, ind):
    if not ind.get("SO")[0].otherParams["predict_disagreement"]:
        # if there is no disagreement the test was already evaluated
        ind.set("DISAGREE_real", do_disagree(ind))
        return do_disagree(ind)
    else:
        ind_temp = copy.deepcopy(ind)
        problem = ADSMultiSimAgreementProblemDiverse(
            problem_name=res.problem.problem_name,
            scenario_path=res.problem.scenario_path,
            xl=res.problem.xl,
            xu=res.problem.xu,
            simulation_variables=res.problem.simulation_variables,
            fitness_function=res.problem.fitness_function,
            critical_function=res.problem.critical_function,
            simulate_functions=res.problem.simulate_functions,
            migrate_function=res.problem.migrate_function,
            simulation_time=res.problem.simulation_time,
            sampling_time=res.problem.sampling_time
        )
        # disable predictor
        pop = Population(individuals=[ind_temp])
        # print(pop)
        evaluate_individuals(pop, problem)
        ind.set("DISAGREE_real", do_disagree(pop[0]))
        return do_disagree(pop[0])

def write_predictions(res, save_folder):
    log_dis_file = save_folder + os.sep + f'disagree_predict.csv'
    with open(log_dis_file, mode = 'w') as f:
        write_to = csv.writer(f)
        if config.EVALUATE_DISAGREEMENTS_PREDICT:
            write_to.writerow(["test_input"] + [f"certainty_cl{i}" for i in range(2)] + ["th_disagree"] + ["false_evaluated"] + ["simulated"])
        else:
            write_to.writerow(["test_input"] + [f"certainty_cl{i}" for i in range(2)] + ["th_disagree"] + ["false_evaluated"])

        for ind in res.archive:
            so = ind.get("SO")[0]
            
            if EVALUATE_DISAGREEMENTS_PREDICT:
                write_to.writerow([ind.get("X")] + \
                        [a.tolist() for a in so.otherParams["predict_uncertainties"]] + \
                        [so.otherParams["predict_disagreement"]] + \
                        [do_disagree(ind)]+ \
                        [evaluate_disagreement(res, ind)]
                )  # simulation results when no disagreement was predicted
            else:
                write_to.writerow([ind.get("X")] + \
                        [a.tolist() for a in so.otherParams["predict_uncertainties"]] + \
                        [so.otherParams["predict_disagreement"]] + \
                        [do_disagree(ind)]
                )  # simulation results when no disagree
def write_prediction_summary(res, save_folder):
    """
        cm[0, 0]: True Positives (TP)
        cm[0, 1]: False Negatives (FN)
        cm[1, 0]: False Positives (FP)
        cm[1, 1]: True Negatives (TN)
    """

    # get finally evaluated agreements/disagreements
    if config.EVALUATE_DISAGREEMENTS_PREDICT:
        final_disagree = res.archive.get("DISAGREE_real")
    else:
        final_disagree = [do_disagree(ind) for ind in res.archive]
    predictions  = [ind.get("SO")[0].otherParams["predict_disagreement"] for ind in res.archive]

    # Calculate confusion matrix
    cm = confusion_matrix(final_disagree, predictions, labels=[True, False])
    accuracy = accuracy_score(final_disagree, predictions)
    precision = precision_score(final_disagree, predictions)
    recall = recall_score(final_disagree, predictions)

    # Display the result
    print("Confusion Matrix:")
    print(cm)

    log_dis_file = save_folder + os.sep + f'predict_summary.csv'
    with open(log_dis_file, mode = 'w') as f:
        write_to = csv.writer(f)
        write_to.writerow(["metric", "value"])
        write_to.writerow(["N", len(predictions)])
        # we can only evaluate false, true negatives
        write_to.writerow(["TP", cm[0,0]])
        write_to.writerow(["FP", cm[1,0]])
        write_to.writerow(["TN", cm[1,1]])
        write_to.writerow(["FN", cm[0,1]])
        write_to.writerow(["accuracy", accuracy])
        write_to.writerow(["precision", precision])
        write_to.writerow(["recall", recall])