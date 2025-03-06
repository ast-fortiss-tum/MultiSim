import random
import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended
from sims.udacity.udacity_simulator_remote import UdacitySimulatorRemote
from pathlib import Path
import sys
import matplotlib
import pymoo
import numpy as np
import os

from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.model_ga.individual import IndividualSimulated
from problem_utils.naming import generate_problem_name
from road_generator.roads.simulator_road import SimulatorRoad
from self_driving.road import Road
from sims.udacity.udacity_simulator import UdacitySimulator
from mock.mock_simulation import MockSimulator
from opensbt.utils.evaluation import evaluate_individuals

from sims.donkey_simulation import DonkeySimulator
from sims.beamng_simulation import BeamngSimulator

from fitness.fitness import MaxXTEFitness, MaxXTECriticality
from config import CRITICAL_XTE, NUM_CONTROL_NODES, ROAD_WIDTH
from opensbt.problem.adas_problem import ADASProblem

from datetime import datetime
import argparse

from scripts.sample_roads import sample_roads, sample_roads_test, critical_roads, road_mix_1, road_mix_2

from shapely.geometry import Point
from opensbt.utils.log_utils import setup_logging
from config import LOG_FILE
import config

import pandas as pd
from pymoo.core.individual import Individual
from pymoo.core.population import Population
import subprocess
import dill
from config import BEAMNG_STEP_SIZE, NUM_CONTROL_NODES
from scripts.evaluate_validity import analyse_and_output_results, sel_sim_fnc
from opensbt.problem.adas_problem import ADASProblem
import copy
from mock.mock_simulation import MockSimulator
from opensbt.utils.duplicates import duplicate_free
import logging as log
from features.feature_map import FeatureMap
import time

logger = log.getLogger(__name__)

log.getLogger('engineio').setLevel(log.ERROR)
log.getLogger('socketio').setLevel(log.ERROR)

matplotlib.use("Agg")

setup_logging(LOG_FILE)

########################

def get_name_simulator(simulate_function):
    if simulate_function == DonkeySimulator.simulate:
        return "Donkey"
    elif simulate_function == UdacitySimulator.simulate or \
         simulate_function == UdacitySimulatorRemote.simulate:
        return "Udacity"
    elif simulate_function == BeamngSimulator.simulate:
        return "Beamng"
    elif simulate_function == MockSimulator.simulate:
        return "Mock"
    else:
        print("Simulator not known.")
        sys.exit(1)

def get_inds_from_fmap(fmap: FeatureMap, 
                                percentage, 
                                threshold,
                                only_failing_cells = False,
                                adaptive = True,
                                threshold_adaptive = 0.9):
    all_inds = []
    for cell,_ in fmap.input_data_all.items():
        in_data_all = fmap.input_data_all[cell]
        data_all = fmap.data_all[cell]

        if only_failing_cells:
            relevant = [in_data_all[id] for id, fitness in enumerate(data_all) if abs(fitness) > threshold]
            fitness_relevant = [fitness for fitness in data_all if abs(fitness) > threshold]
        else:
            relevant = [in_data_all[id] for id, fitness in enumerate(data_all)]
            fitness_relevant = data_all


        # if standard deviation is low, take just the percentage
        if adaptive:
            if np.asarray(fitness_relevant).std() < threshold_adaptive:         
                # all but only test input for each fitness
                encountered_fitness = set()

                # List to store selected individuals
                selected_individuals = []

                # Loop through individuals and fitness values
                for individual, fitness in zip(relevant, fitness_relevant):
                    if fitness not in encountered_fitness:
                        encountered_fitness.add(fitness)  # Mark this fitness as encountered
                        selected_individuals.append(individual)  # Select this individual for the unique fitness value
                relevant = selected_individuals
                # # if percentage value is an integer, threat as absolute count
                # if int(percentage) == percentage:
                #     num_samples = min(len(relevant), percentage)
                # else:
                #     num_samples = int(max(len(relevant) * percentage,
                #                         min(len(relevant),1)))
                num_samples = len(relevant)
            else:
                # take all
                num_samples = len(relevant)
        else:
            # if percentage value is an integer, threat as absolute count
            if int(percentage) == percentage:
                num_samples = min(len(relevant), percentage)
            else:
                num_samples = int(max(len(relevant) * percentage,
                                    min(len(relevant),1)))
        if num_samples == 0:
            selected_np = []
        else:
            if not adaptive:
                selected = random.sample(relevant, int(num_samples))
            else:
                selected = relevant
                
            # if only_failing_cells:
            #     selected = random.sample(relevant, int(num_samples))
            # else:
            #     # sort
            #     selected = np.sort(np.asarray(relevant))[:int(num_samples)]
                
            selected_np = []
            for element in selected:
                ind = Individual()
                ind.set("X", np.array(element))
                ind.set("F", None) # add later
                selected_np.append(ind)
            # print("selected", selected_np)

        all_inds = all_inds + selected_np
    
    print("all_inds", all_inds)
    return all_inds

def read_pf_single(filename):
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
        F = table.iloc[i, n_var + 1:-1].to_numpy()
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        individuals.append(ind)
    print(f"Reading of {len(individuals)} tests completed")
    return Population(individuals=individuals)

def validate_results_from_pop(pop_mix,
                    path_problem,
                    n_repeat,
                    save_folder_parent,
                    simulate_function, # to select specific simulator. function in problem will be overriden
                    problem_obj = None,
                    folder_validation = "validation",
                    write_results_extended = False):
        
    start_time = time.time()

    if path_problem is not None:
        with open(path_problem, 'rb') as f:
            problem_read = dill.load(f)
            # print("Loaded problem:", problem_read) 
    
        problem = ADASProblem(
            problem_name=f"",
            scenario_path="",
            xl=problem_read.xl,
            xu=problem_read.xu,
            simulation_variables=problem_read.simulation_variables,
            fitness_function=problem_read.fitness_function,
            critical_function=problem_read.critical_function,
            simulate_function=simulate_function,
            simulation_time=30,
            sampling_time=0.25
        )
    if problem_obj is not None:

        problem = ADASProblem(
            problem_name=f"",
            scenario_path="",
            xl=problem_obj.xl,
            xu=problem_obj.xu,
            simulation_variables=problem_obj.simulation_variables,
            fitness_function=problem_obj.fitness_function,
            critical_function=problem_obj.critical_function,
            simulate_function=simulate_function,
            simulation_time=30,
            sampling_time=0.25
        )
        
    problem.problem_name = generate_problem_name(problem,
                                                category="Result_Validation",
                                                fitness_name="NOTKNOWN",
                                                suffix=f"step{BEAMNG_STEP_SIZE}")
    
    res = validate_results_pop(pop_mix, 
                        problem = problem,
                        n_repeat = n_repeat,
                        save_folder_parent = save_folder_parent,
                        folder_validation = folder_validation,
                        write_results_extended = write_results_extended,
                        simulator_name = get_name_simulator(simulate_function),
                        simulate_function = simulate_function)
    return res

def validate_results_pop(pop_mix, 
                        problem, 
                        n_repeat, 
                        save_folder_parent,
                        folder_validation,
                        write_results_extended,
                        simulator_name,
                        simulate_function):
    
    start_time = time.time()
    ######################################

    n_roads = len(pop_mix)

    log.info(f"Performing validation on {n_roads} roads...")

    #####################
    inds = []
    for ind in pop_mix:
        inds = inds + [copy.deepcopy(ind) for _ in range(0,n_repeat)]

    pop = PopulationExtended(individuals=inds)
    
    print(pop)
    if len(pop) != 0:
        evaluate_individuals(population = pop, problem=problem)

    if save_folder_parent is not None:
        Path(save_folder_parent + os.sep + folder_validation + os.sep).mkdir(parents=True, exist_ok=True)
        save_folder_validation = save_folder_parent + os.sep + folder_validation + os.sep
    else:
        save_folder_validation = None
    
    simulator_name = get_name_simulator(simulate_function)
    name_validation = save_folder_validation + "fmap_valid_" + simulator_name.lower()
    
    time_validation = time.time() - start_time
    log.info(f"Time validation: --- %s seconds ---" % time_validation)
    
    res = analyse_and_output_results(pop,  # roads for repeated execution
                        problem, 
                        n_repeat, 
                        n_roads, 
                        save_folder_validation,
                        angles_list=pop_mix.get("X"),
                        write_results_extended = write_results_extended, # the actual roads once each roads
                        simulator_name = simulator_name,
                        name_validation = name_validation,
                        time_validation = time_validation)
    return res

def validate_results(path_problem,
                    path_testcases,
                    n_repeat,
                    save_folder_parent,
                    simulate_function, # to select specific simulator. function in problem will be overriden
                    problem_obj = None,
                    folder_validation = "validation",
                    write_results_extended = False,
                    path_fmap = None,
                    validate_portion = False,
                    percentage_validation = 1):

    if validate_portion:
        fmap: FeatureMap = FeatureMap.from_json(path_fmap)
        individuals = get_inds_from_fmap(fmap, 
                                                  percentage=percentage_validation,
                                                  threshold=CRITICAL_XTE)
        pop_mix = Population(individuals)
        # read in idnviduals from feature map
    else:
        pop_mix = read_pf_single(path_testcases)
    
    # if len(pop_mix) == 0:
    #     log.info("No test cases available. Terminating...")
    #     sys.exit(1)
    
    pop_mix = duplicate_free(pop_mix)

    assert path_problem is not None or problem_obj is not None, "Path to problem, or problem object has to be given."
    
    return validate_results_from_pop(pop_mix=pop_mix,
                                    path_problem=path_problem,
                                    n_repeat= n_repeat,
                                    save_folder_parent = save_folder_parent,
                                    simulate_function = simulate_function, # to select specific simulator. function in problem will be overriden
                                    problem_obj = problem_obj,
                                    folder_validation = folder_validation,
                                    write_results_extended = write_results_extended)
if __name__ == "__main__":
    # test data
    # read roads from mixed and single runs from csv file into population
    path_mix = "./scripts/data/all_critical_testcases_ub.csv"
    path_s1 = "./scripts/data/all_critical_testcases_u.csv"
    path_s2 = "./scripts/data/all_critical_testcases_b.csv"

    testproblem= r"C:\\Users\sorokin\Documents\\testing\\opensbt-multisim\\results\single\\UB_A7_0-360_XTE_STEER_gen15_pop15\\NSGAII-D\\15-06-2024_23-30-59\backup\\problem"
    testdata = path_mix
    testsavefolder = r"C:\\Users\sorokin\Documents\\testing\\opensbt-multisim\\results\single\\UB_A7_0-360_XTE_STEER_gen15_pop15\\NSGAII-D\\15-06-2024_23-30-59\\"

    #############

    parser = argparse.ArgumentParser(description="Pass parameters for validation of simulator on different roads.")
    parser.add_argument('-s', dest='simulator', type=str,  default="Mock", action='store',
                        help='Specify the simulator to use.')
    parser.add_argument('-n', dest='n_repeat', type=int, default=10, action='store', help='Specify the number of repetitions.')
    parser.add_argument('-t', dest='path_testcases', type=str, default=testdata, 
                        action='store', help='Specify the path to the testcases.')
    parser.add_argument('-p', dest='path_problem', type=str, default=testproblem, action='store', help='Specify the path to the problem.')
    parser.add_argument('-o', dest='save_folder_parent', type=str, default=testsavefolder, action='store', help='Specify the path where to store validation results.')
    # Optional arguments (with abbreviations)
    parser.add_argument('-fv', '--folder_validation', type=str, default='validation', help='Folder name for validation output.', dest='folder_validation')
    parser.add_argument('-we', '--write_extended', action='store_true', help='Flag to write extended results.', dest='write_results_extended')
    parser.add_argument('-pfmap', '--path_fmap', type=str, default=None, help='Path to fmap file, if needed.', dest='path_fmap')
    parser.add_argument('-vp', '--validate_portion', action='store_true', help='Flag to validate a portion of the test cases.', dest='validate_portion')
    parser.add_argument('-pv', '--percentage_validation', type=float, default=1.0, help='Percentage of test cases to validate (1.0 = 100%%).', dest='percentage_validation')

    args = parser.parse_args()

    #####################

    # load the fitness and criticality function
    path_problem = args.path_problem
    path_testcases = args.path_testcases
    n_repeat = args.n_repeat
    save_folder_parent = args.save_folder_parent

    simulate_function = sel_sim_fnc(args.simulator)
    
    folder_validation=args.folder_validation
    write_results_extended=args.write_results_extended
    path_fmap=args.path_fmap
    validate_portion=args.validate_portion
    percentage_validation=args.percentage_validation
    
    validate_results(path_problem,
                    path_testcases,
                    n_repeat,
                    save_folder_parent,
                    simulate_function,
                    problem_obj = None,
                    folder_validation = folder_validation,
                    write_results_extended = write_results_extended,
                    path_fmap = path_fmap,
                    validate_portion = validate_portion,
                    percentage_validation = percentage_validation)