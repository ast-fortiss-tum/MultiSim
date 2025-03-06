
import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from sims.sim_names import sel_sim_fnc
import subprocess
import sys
import os
import matplotlib
from sims.beamng_simulation import BeamngSimulator

from mock.mock_simulation import MockSimulator
from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.algorithm.ps_grid import PureSamplingGrid
from opensbt.algorithm.ps_rand import PureSamplingRand
from algorithm.nsga2d_sim import NSGAIID_SIM
from scripts_analysis import validation_combined
from sims.donkey_simulation import DonkeySimulator
from sims.udacity.udacity_simulator import UdacitySimulator
import argparse
import logging as log
import os
import re
import sys

from opensbt.algorithm.nsga2_dt_sim import *
from opensbt.algorithm.nsga2_sim import * 

from single_experiments import experiment_switcher as s_experiment_switcher
from multi_experiments import experiment_switcher as m_experiment_switcher
from test_experiments import experiment_switcher as t_experiment_switcher
from extra_experiments import experiment_switcher as e_experiment_switcher
from predict_experiments import experiment_switcher as p_experiment_switcher

from opensbt.utils.log_utils import *
from config import LOG_FILE, EXPERIMENTAL_MODE, PERCENTAGE_VALIDATION_MSIM, PERCENTAGE_VALIDATION_SSIM
from pymoo.config import Config
from scripts.validate_results import validate_results
from opensbt.problem.adas_multi_sim_aggreement_problem_diverse import ADSMultiSimAgreementProblemDiverse
from scripts_analysis import validation

logger = log.getLogger(__name__)

log.getLogger('engineio').setLevel(log.ERROR)
log.getLogger('socketio').setLevel(log.ERROR)

matplotlib.use("Agg")

Config.warnings['not_compiled'] = False

results_folder = '/results/single/'
setup_logging(LOG_FILE)

algorithm = None
problem = None
experiment = None

########
parser = argparse.ArgumentParser(description="Pass parameters for search.")
parser.add_argument('-e', dest='exp_number', type=str, action='store',
                    help='Hardcoded example scenario to use [2 to 6].')
parser.add_argument('-r', dest='n_runs', default=1, type=int, action='store',
                    help='Number of times to repeat experiment.')
parser.add_argument('-i', dest='n_generations', type=int, default=None, action='store',
                    help='Number generations to perform.')
parser.add_argument('-n', dest='size_population', type=int, default=None, action='store',
                    help='The size of the initial population of scenario candidates.')
parser.add_argument('-a', dest='algorithm', type=int, default=None, action='store',
                    help='The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.')
parser.add_argument('-t', dest='maximal_execution_time', type=str, default=None, action='store',
                    help='The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might perform nsga2 iterations, when time limit is already reached.')
parser.add_argument('-f', dest='scenario_path', type=str, action='store',
                    help='The path to the scenario description file/experiment.')
parser.add_argument('-min', dest='var_min', nargs="+", type=float, action='store',
                    help='The lower bound of each parameter.')
parser.add_argument('-max', dest='var_max', nargs="+", type=float, action='store',
                    help='The upper bound of each parameter.')
parser.add_argument('-m', dest='design_names', nargs="+", type=str, action='store',
                    help='The names of the variables to modify.')
parser.add_argument('-dt', dest='max_tree_iterations', type=int, action='store',
                    help='The maximum number of total decision tree generations (when using NSGA2-DT algoritm).')
parser.add_argument('-o', dest='results_folder', type=str, action='store', default=os.sep + "results" + os.sep,
                    help='The name of the folder where the results of the search are stored (default: \\results\\single\\)')
parser.add_argument('-v', dest='do_validation', action='store_true', default=False,
                    help='Reruns tests in simulator to validate.')
parser.add_argument('-rv', dest='n_repeat_validation', default=5, type=int, action='store',
                    help='Number of times to reexecute tests for validation.')
parser.add_argument('-s', dest='seed', default=None, type=int, action='store',
                    help='Seed.')
parser.add_argument('-sf', dest='subfolder_name', type=str, action='store', default=None,
                    help='Subfolder name to store results')
parser.add_argument('-sims_validation', nargs="+", type=str, action="store", default="udacity donkey", 
                        help='Simulator names to use for validation")')
args = parser.parse_args()

#######

if args.exp_number and args.scenario_path:
    log.info("Flags set not correctly: Experiment file and example experiment cannot be set at the same time")
    sys.exit()
elif not (args.exp_number or args.scenario_path):
    log.info("Flags set not correctly: No file is provided or no example experiment selected.")
    sys.exit()

###### set experiment
####### have indiviualized imports
if args.exp_number:
    # exp_number provided
    selExpNumber = re.findall("[0-9]+", args.exp_number)[-1]
    log.info(f"Selected experiment number: {selExpNumber}")
    
    experiment_switcher = {
                           **s_experiment_switcher, 
                           **m_experiment_switcher, 
                           **t_experiment_switcher,
                           **e_experiment_switcher,
                           **p_experiment_switcher
    }
    # merge the switcher because we have multiple files
    experiment = experiment_switcher.get(int(selExpNumber))()

    config = experiment.search_configuration
    problem = experiment.problem
    algorithm = experiment.algorithm

elif (args.scenario_path):
    scenario_path = args.scenario_path
    var_min = []
    var_max = []

    #TODO create an experiment from user input
    #TODO create an ADASProblem from user input

    log.info("-- Experiment provided by file")

    if args.var_min is None:
        log.info("-- Minimal bounds for search are not set.")
        sys.exit()

    if args.var_max is None:
        log.info("-- Maximal bounds for search are not set.")
        sys.exit()

    log.info("Creating an experiment from user input not yet supported. Use default_experiments.py to create experiment")
    sys.exit()
else:
    log.info("-- No file provided and no experiment selected")
    sys.exit()

'''
override params if set by user
'''

if not args.size_population is None:
    config.population_size = args.size_population
    problem.update_name("pop", args.size_population)
if not args.n_generations is None:
    config.n_generations = args.n_generations
    config.inner_num_gen = args.n_generations #for NSGAII-DT
    problem.update_name("gen", args.n_generations)
if not args.algorithm is None:
    algorithm = AlgorithmType(args.algorithm)
if not args.maximal_execution_time is None:
    config.maximal_execution_time = args.maximal_execution_time
if not args.max_tree_iterations is None:
    config.max_tree_iterations = args.max_tree_iterations
if not args.results_folder is None:
    results_folder = args.results_folder
if not args.var_max is None:
    problem.var_max = args.var_max
if not args.var_min is None:
    problem.var_min = args.var_min
if not args.design_names is None:
    problem.design_names = args.design_names
if not args.seed is None:
    config.seed = args.seed
    problem.update_name("seed", args.seed)
if not args.subfolder_name is None:
    config.write_subfolder_name = args.subfolder_name

####### Run algorithm

if __name__ == "__main__":
    execTime = None
    algo = None
    for i in range(args.n_runs):
        if algorithm == AlgorithmType.NSGAII:
            log.info("pymoo NSGA-II algorithm is used.")
            algo = NSGAII_SIM(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.NSGAIIDT:
            log.info("NSGAII-DT algorithm is used.")
            algo = NSGAII_DT_SIM(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.NSGAIID:
            log.info("pymoo NLSGAII algorithm is used.")
            algo = NSGAIID_SIM(
                                    problem=problem,
                                    config=config)
        elif algorithm == AlgorithmType.PS_RAND:
            log.info("pymoo PureSampling algorithm is used.")
            algo = PureSamplingRand(
                                problem=problem,
                                config=config)
        elif algorithm == AlgorithmType.PS_GRID:
            log.info("pymoo PureSampling algorithm is used.")
            algo = PureSamplingGrid(
                                problem=problem,
                                config=config)
        else:
            raise ValueError("Error: No algorithm with the given code: " + str(algorithm))
        res = algo.run()

        if results_folder is not None:
            # backup will be written in different folder if results folder set
            results_folder = results_folder + os.sep
        else:
            results_folder = algo.save_folder if hasattr(algo, 'save_folder') \
                        else RESULTS_FOLDER
        
        # print(f"results_folder: {results_folder}")
        
        # print("algo save_folder", algo.save_folder)

        save_folder = algo.write_results(results_folder)  # temporar hack
        
        if problem.is_simulation() and args.do_validation:
            if EXPERIMENTAL_MODE:
                simulators = [
                    MockSimulator.simulate,
                    MockSimulator.simulate
                ]
            else:      
                simulators = [sel_sim_fnc(sim) for sim in args.sims_validation]
     
            if isinstance(problem, ADSMultiSimAgreementProblemDiverse):
                percentage_validation = PERCENTAGE_VALIDATION_MSIM
            else:
                percentage_validation = PERCENTAGE_VALIDATION_SSIM
            
            log.info(f"Using percentage validation: {percentage_validation}")

            validation.run_validation( 
                                               simulators, 
                                               args.n_repeat_validation,
                                               write_results_extended = False,
                                               save_folder = save_folder,
                                               percentage_validation=percentage_validation,
                                               folder_name_combined_prefix=f"validation_count-{percentage_validation}_combined_",
                                               folder_validation_prefix=f"validation_count-{percentage_validation}_",
                                               do_combined_validation=True)

    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")