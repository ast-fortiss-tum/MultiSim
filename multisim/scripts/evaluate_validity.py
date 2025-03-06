from copy import deepcopy
from pathlib import Path
import sys
import matplotlib
import pymoo
import numpy as np
import os

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from fitness import fitness
from opensbt.algorithm.algorithm import AlgorithmType
from problem_utils.naming import generate_problem_name
from road_generator.roads.simulator_road import SimulatorRoad
from self_driving.road import Road
from sims.udacity.udacity_simulator import UdacitySimulator
from opensbt.utils.archive import MemoryArchive
from sims.udacity.udacity_simulator_remote import UdacitySimulatorRemote
from mock.mock_simulation import MockSimulator
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.utils.evaluation import evaluate_individuals

from sims.donkey_simulation import DonkeySimulator
from sims.beamng_simulation import BeamngSimulator

from fitness.fitness import MaxXTEFitness, MaxXTECriticality
from config import MAX_SEG_LENGTH, MIN_SEG_LENGTH, NUM_CONTROL_NODES, ROAD_WIDTH
from opensbt.problem.adas_problem import ADASProblem

from datetime import datetime
import argparse

from scripts.sample_roads import sample_roads, sample_roads_test, \
                                 critical_roads, road_mix_1, road_mix_2, \
                                 roads_msim_run0, roads_ssim1_run0, roads_ssim2_run0, \
                                 critical_roads_2, roads_polgyon_error, one_road, disagree_candidate, \
                                 test_road, test_road_2

from shapely.geometry import Point
from opensbt.utils.log_utils import setup_logging
from config import LOG_FILE
import config
import matplotlib.pyplot as plt

from config import CRITICAL_XTE, CRITICAL_STEERING, WRITE_ALL_INDIVIDUALS, \
    EXPERIMENTAL_MODE, DO_PLOT_GIFS
    
import csv

def analyse_and_output_results(pop, 
                    problem, 
                    n_repeat, 
                    n_roads, 
                    save_folder = None, 
                    write_results_extended = False, 
                    angles_list = None,
                    simulator_name = None,
                    name_validation = None,
                    time_validation = -1):

    ############## create result
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from opensbt.utils.result_utils import create_result
    # create some algorithm instance for storage
    hist_holder = []
    inner_algorithm = NSGA2(
            pop_size=None,
            n_offsprings=None,
            sampling=None,
            eliminate_duplicates=True,
            archive = MemoryArchive())
    for i in range(n_roads):        
        inner_algorithm = deepcopy(inner_algorithm)
        inner_algorithm.pop = pop[i*(n_repeat): (i+1)*n_repeat]
        inner_algorithm.archive = pop[i*(n_repeat): (i+1)*n_repeat]
        hist_holder.extend([inner_algorithm])

    res = create_result(hist_holder=hist_holder,
                inner_algorithm=inner_algorithm,execution_time=-1,
                problem=problem)

    ############## writing basic
    fvalues_roads = [] 
    avg_fitness = []
    max_fitness = []
    min_fitness = []
    
    ind_min_fitness = []
    ind_max_fitness = []

    std_fitness = []
    critical_ratio = [] 

    simouts_roads = []

    for i in range(n_roads):
        pop_road = pop[i*(n_repeat): (i+1)*n_repeat]
        fvalues = pop_road.get("F")
        fvalues_roads.append(fvalues)
        
        simouts_roads.append(pop_road.get("SO"))

        avg_fitness.append(np.mean(fvalues, axis = 0))
        max_fitness.append(np.max(fvalues, axis = 0))
        min_fitness.append(np.min(fvalues, axis = 0))

        ind_min_fitness.append(np.argmin(fvalues, axis = 0)[0])
        ind_max_fitness.append(np.argmax(fvalues, axis = 0)[0])
        
        std_fitness.append(np.std(fvalues, axis = 0))
        critical_ratio.append(len(pop_road.divide_critical_non_critical()[0])/len(pop_road))

    # print(f"fitness values: {fvalues}")

    data = {
            "roads_angles" : angles_list,
            "n_repeat" : n_repeat,
            "critical_ratio": critical_ratio,
            "critical_ratio_all": np.mean(critical_ratio),
            "avg_fitness" : avg_fitness, 
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "ind_min_fitness" : ind_min_fitness,
            "ind_max_fitness" : ind_max_fitness,
            "std_fitness" : std_fitness,
            "std_fitness_avg" : np.mean(std_fitness),
            "all_fitness" : np.asarray(fvalues_roads).tolist(),
            "time_validation" : time_validation
    }

    data_simout = {
        "roads_angles" : angles_list,
        "simout_roads" : [[so.to_dict() for so in so_array] for so_array in simouts_roads]
    }

    #############
    # truncate


    #############
    # write
    import json
    from opensbt.visualization import output
    from opensbt.utils.encoder_utils import NumpyEncoder

    if save_folder is None:
        save_folder = output.create_save_folder(problem, 
                                                results_folder = os.sep + "results" + os.sep,
                                                    algorithm_name = "", 
                                                    is_experimental=EXPERIMENTAL_MODE)
    print(f"save_folder is: {save_folder}")    
    time  = datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S") 
    
    with open(save_folder + os.sep + f"validation_results.json", 'w') as f:
        json.dump(data, 
                f, 
                indent=4,
                cls=NumpyEncoder)
        
    with open(save_folder + os.sep + f"validation_simout.json", 'w') as f:
        json.dump(data_simout, 
                f, 
                indent=4,
                cls=NumpyEncoder)
        
    # write as csv file
    '''Output of summery of the performance'''
    with open(save_folder + os.sep + f"validation_results.csv", 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(["n_roads", n_roads])
        write_to.writerow(["n_repeat", data["n_repeat"]])
        write_to.writerow(["critical_ratio", data["critical_ratio"]]),
        write_to.writerow(["critical_ratio_all", data["critical_ratio_all"]])
        write_to.writerow(["std_fitness", data["std_fitness"]])
        write_to.writerow(["timestamp", time])
        f.close()

    ############ default writing
    import logging as log
    from features import opensbt_feature

    def write_results(res ,save_folder):
        print("save_folder in write_results", save_folder)
        algorithm_name = "VALIDATION"
        algorithm_parameters = {
        }
        
        # output.export_generations_roads(res, save_folder)
        # output.export_roads(res, save_folder)
        # output.plot_critical_roads(res, save_folder)
        #persist results object
        res.persist(save_folder + "backup")

        # output.igd_analysis(res, save_folder)
        # output.gd_analysis(res,save_folder)
        # output.hypervolume_analysis(res, save_folder)
        # output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        # output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        # if DO_PLOT_GIFS:
        #     output.simulations(res, save_folder)
        output.all_critical_individuals(res, save_folder)
        output.write_generations(res, save_folder)
        output.write_criticality_all_sims(res, save_folder)
        output.write_fitness_all_sims(res, save_folder)
        
        
        # output.calculate_n_crit_distinct(res,save_folder,
                                            # bound_min=problem.xl,
                                            # bound_max=problem.xu, 
                                            # var="X")
        
        # output.calculate_n_crit_distinct(res,save_folder,
        #                                 bound_min=config.ideal,
        #                                 bound_max=config.nadir, 
        #                                 var="F")

        # output.calculate_n_crit_distinct(res,
        #                                  save_folder,
        #                                  bound_min=config.ideal,
        #                                  bound_max=config.nadir, 
        #                                  var="F")

            
        # import config as config
        # for type in ["X", "Y", "V", "XTE", "throttles", "steerings"]:
        #     output.comparison_trace(res, save_folder, mode=config.MODE_PLOT_RESULTS, type=type)
        
        output.write_multisim_analysis(res, save_folder)
        # output.scenario_plot_disagreements(res, save_folder)
        output.write_config(save_folder)
        
        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

        # output.write_disagree_testcases(res, save_folder)
        output.plot_generations_scenario(res, save_folder)

        opensbt_feature.plot_feature_map(res, 
                                         save_folder, 
                                         max_fitness=0, 
                                         min_fitness=-3,
                                         name_fmap=name_validation,
                                         title_suffix="Validation " + simulator_name + " ")


    if write_results_extended:
        write_results(res, save_folder)
    else:
        # just backup
        res.persist(save_folder + "backup")
        output.write_config(save_folder)

        # write fmap
        opensbt_feature.plot_feature_map(res, 
                                         save_folder, 
                                         max_fitness=0, 
                                         min_fitness=-3,
                                         name_fmap=name_validation,
                                         title_suffix="validation_" + simulator_name.lower() + "_")




    # overview roads
    output.plot_roads_overview_generation(res,save_folder)   

    ########### create box plot ############
    # Import libraries
    
    # create box plot for each fitness dimension
    for j in range(problem.n_obj):
        fig = plt.figure(figsize =(10, 7))
        ax = fig.gca()
        
        def get_box_plot_data_road(fvalues, index_fitness):
            if problem.n_obj == 1:
                return np.concatenate(fvalues).tolist()
            else:
                return np.asarray([f[j] for f in fvalues])
        
        plt.boxplot([get_box_plot_data_road(fvalues, index_fitness = j) for fvalues in fvalues_roads])
        
        data["critical_ratio"] = [float(f"{x:.2f}") for x in data["critical_ratio"]]

        if simulator_name == "Beamng":
            beamng_step_size = -1
            plt.title(f"{simulator_name} | Simulator Validation and Flakiness Analysis {n_repeat} Repetitions | \nStep {beamng_step_size} | Fitness: {problem.objective_names[j]} | Critical Ratio: {data['critical_ratio']}")
        else:
            plt.title(f"{simulator_name} | Simulator Validation and Flakiness Analysis {n_repeat} Repetitions | \nFitness: {problem.objective_names[j]} | Critical Ratio: {data['critical_ratio']}")

        # hardcoded for now: first is max XTE and second is STEEERING Change
        if j==0:
            plt.ylim(-3.5, 0)
        if j==1:
            plt.ylim(-20, 0)

        n_turns_list = []
        curvature_list = []
        for i,fvalues in enumerate(fvalues_roads):
        
            # add road information
            road_p = pop[i*n_repeat].get("SO").otherParams["road"]
            road: Road = SimulatorRoad(control_points=[
                    Point(t[0], t[1], t[2]) for t in road_p
                    ],
                    road_points=[
                    Point(t[0], t[1], t[2]) for t in road_p
                    ],
                    road_width = ROAD_WIDTH
            )
            curvature_list.append(road.compute_curvature())
            n_turns_list.append(road.compute_num_turns()[0])
    
            if problem.n_obj == 1:
                y_values = np.asarray(np.concatenate(
                        fvalues
                        ).tolist())
            else:
                y_values = np.asarray([f[j] for f in fvalues])
            ax.plot([i + 1]*n_repeat, 
                    y_values,
                    'ko', 
                    alpha=0.7,
                    markerfacecolor='none', 
                    label=f'R{i + 1} | nt:{round(n_turns_list[i],0)} | c:{round(curvature_list[i],2)}') 

        plt.xticks([i + 1 for i in range(n_roads)], 
                    [f'R{i + 1}' for i in range(n_roads)])

        if j == 0:
            plt.axhline(y=-CRITICAL_XTE, color='r', linestyle='--')  # 'r' stands for red color, '--' stands for dashed line
        if j == 1:
            plt.axhline(y=-CRITICAL_STEERING, color='r', linestyle='--')  # 'r' stands for red color, '--' stands for dashed line

        # Add the legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        # show plot
        # plt.show()
        plt.savefig(save_folder + f"{problem.problem_name}_boxplot_fitness_{problem.objective_names[j]}.jpg")
        plt.clf()

    return res

def sel_sim_fnc(kwd: str):
    if kwd.lower() == "donkey":
        return  DonkeySimulator.simulate
    elif kwd.lower() == "udacity":
        return UdacitySimulator.simulate
    elif kwd.lower() == "udacity_remote":
        return UdacitySimulatorRemote.simulate
    elif kwd.lower() == "beamng":
        return BeamngSimulator.simulate
    elif kwd.lower() == "mock":
        return MockSimulator.simulate
    else:
        print("Simulator not known.")
        sys.exit(1)

if __name__ == "__main__":

    setup_logging(LOG_FILE)
    #############
    def parse_array_of_arrays(value):
        """Parses a string representation of an array of arrays."""
        try:
            # Split top-level arrays by semicolon
            return [[float(num) if '.' in num else int(num) for num in sublist.split(',')] for sublist in value.split(';')][0]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected semicolon-separated lists of comma-separated integers.")


    parser = argparse.ArgumentParser(description="Pass parameters for validation of simulator on different roads.")
    parser.add_argument('-s', dest='simulator', type=str,  default="Mock", action='store',
                        help='Specify the simulator to use.')
    parser.add_argument('-prefix', dest='prefix', type=str,  default="", action='store',
                        help='Specify the prefix for the output folder name.')

    parser.add_argument('-n', dest='n_repeat', type=int, default=5, action='store', help='Specify the number of repetitions.')
    parser.add_argument('-w', dest='write_res_extend', default=False, action='store_true', help='Write all information of the execution down or not.')
    parser.add_argument('-t', dest='beamng_step_size', type=int, default=None, action='store', help='The step size of beamng.')
    parser.add_argument('-tests', dest='tests', nargs="+", type=parse_array_of_arrays, action="store", required=False, help="Test to runs. If none provided internal file is used")
    parser.add_argument('-custom_lengths', dest='custom_lengths', action="store_true", default=False, required=False, help="Use custom angles.")

    args = parser.parse_args()
    #############
        
    simulate_function = sel_sim_fnc(args.simulator)
    n_repeat = args.n_repeat

    if args.beamng_step_size is None:
        beamng_step_size = config.BEAMNG_STEP_SIZE
    else:
        beamng_step_size = args.beamng_step_size
    #############
    prefix = args.prefix

    n_cp = NUM_CONTROL_NODES

    use_segment_lengths = args.custom_lengths

    if use_segment_lengths:
        problem = ADASProblem(
            problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
            scenario_path="",
            xl=[-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
            xu=[180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
            simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)] + 
                                    [f"seg_length{i}" for i in range(1,n_cp - 1)],
            fitness_function=fitness.MaxXTEFitness(diversify=True),
            critical_function=fitness.MaxXTECriticality(),
            simulate_function=simulate_function,
            simulation_time=30,
            sampling_time=0.25,
        )
    else:
        problem = ADASProblem(
            problem_name=f"Validation_Beamng_Matteo_A{n_cp-2}_XTE",
            scenario_path="",
            xl=[0]*(n_cp-2),
            xu=[360]*(n_cp-2),
            simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
            fitness_function=fitness.MaxXTEFitness(),
            critical_function=fitness.MaxXTECriticality(),
            simulate_function=simulate_function,
            simulation_time=30,
            sampling_time=0.25,
        )

    problem.problem_name = generate_problem_name(problem,
                                                category="Validity",
                                                fitness_name="XTE",
                                                suffix=f"step{beamng_step_size}",
                                                prefix=prefix)

    if args.tests != None:
        angles_list = args.tests  
        print(angles_list)  
    else:
        angles_list = sample_roads
    # angles_list = test_road_2
    # angles_list = sample_roads[7:]
    # angles_list = sample_roads[:2]
    # angles_list = critical_roads_2
    # angles_list = road_mix_1
    # angles_list = roads_msim_run0
    # angles_list = one_road
    # angles_list = disagree_candidate
    # angles_list = roads_ssim1_run0
    # angles_list = roads_polgyon_error[::-1]
    # angles_list = test_road[2:]

    n_roads = len(angles_list)

    #####################
    inds = []
    for angles in angles_list:
        inds = inds + [IndividualSimulated(X=angles) for _ in range(0,n_repeat)]


    pop = PopulationExtended(individuals=inds)

    evaluate_individuals(population = pop, problem=problem)
    analyse_and_output_results(pop, 
                        problem, 
                        n_repeat, 
                        n_roads = n_roads,
                        write_results_extended = args.write_res_extend,
                        simulator_name = args.simulator)