from math import ceil
import os
from sims.donkey_simulation import DonkeySimulator
from mock.mock_simulation import MockSimulator
from opensbt.evaluation.fitness import *
from fitness.migration import CriticalAgreementDualMigration
from opensbt.problem.adas_multi_sim_problem import ADSMultiSimProblem

from pymoo.core.population import Population
from pymoo.core.individual import Individual
from opensbt.problem.adas_problem import ADASProblem
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.simulation.dummy_simulation_od import DummySimulator3
from fitness.fitness import CRITICAL_XTE, MaxAvgXTEFitness, MaxXTECriticality, MaxXTESteeringChangeFitness
from problem_utils.naming import generate_problem_name
from sims.udacity.udacity_simulator import UdacitySimulator
from fitness import fitness
from operators import operators, ambiegen_crossover
from config import NUM_CONTROL_NODES, ROAD_MUT_PROB, \
                    MAX_DISPLACEMENT, NUM_TRIALS_MUT,CROSS_RATE, SEED, \
                    MIN_SEG_LENGTH, MAX_SEG_LENGTH, EXPERIMENTAL_MODE

from scripts.validate_results import validate_results_pop
from opensbt.utils.files import find_folders_with_name
from sims.sim_names import sel_sim_fnc
import argparse

n_cp = NUM_CONTROL_NODES

''' perform validation with another sim '''

# we should store test case part of validation in extra file


# TODO load tests from validation_results.json
# validation_folders = find_folders_with_name(
#     r"..\results\analysis\analysis_5-runs_26-08-2024_19-17-26_20-gen_20-pop\run_0\msim\VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed702\NSGAII-D\myfolder\\",
#                         "validation_0")
# simulator_name = "mock"

# n_repeat = 5
# write_results_extended = False
# simulator_name = "beamng"
# folder_validation = "validation_beamng2"
# input_validation_results_file = validation_folders[0] + os.sep + "validation_results.json"

def validate_results_after(input_validation_results_file,
                   simulator_name,
                   n_repeat=5,  
                   write_results_extended=False, 
                   folder_validation="validation_X"
                 ):
    simulate_function = sel_sim_fnc(simulator_name)
    save_folder_parent = os.path.dirname(os.path.dirname(input_validation_results_file))
    # save_folder_validation = save_folder_parent + os.sep + folder_validation + os.sep
    # name_validation = save_folder_validation + "fmap_valid_" + simulator_name

    with open(input_validation_results_file, 'r') as file:
        data = json.load(file)

    angles = data["roads_angles"]
    print("loaded angles", angles)

    individuals = []
    for angle in angles:
        ind = Individual()
        ind.set("X", angle)
        ind.set("F", None)
        individuals.append(ind)
    pop = Population(individuals=individuals)

    # TODO instantiate problem; load if exists

    problem = ADASProblem(
        problem_name = f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path = "",
        xl = [-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
        xu = [180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
        simulation_variables = [f"angle{i}" for i in range(1,n_cp - 1)] + 
                                [f"seg_length{i}" for i in range(1,n_cp - 1)],
        fitness_function = fitness.MaxXTEFitness(diversify=True),
        critical_function = fitness.MaxXTECriticality(),
        simulate_function = simulate_function,
        simulation_time=30,
        sampling_time=0.25,
    )
    # TODO execute scripts

    validate_results_pop(pop, 
                        problem, 
                        n_repeat, 
                        save_folder_parent,
                        folder_validation,
                        write_results_extended,
                        simulator_name,
                        simulate_function)

if __name__ == "__main__":     
    ###############################

    parser = argparse.ArgumentParser(description="Run simulation with specific parameters")

    # Add arguments with default values
    parser.add_argument('--n_repeat', type=int, default=5, 
                        help='Number of times to repeat the simulation (default: 5)')
    parser.add_argument('--write_results_extended', action = "store_true", default=False, 
                        help='Whether to write extended results (default: False)')
    parser.add_argument('--simulator_name', type=str, default="beamng", 
                        help='Name of the simulator to use (default: beamng)', required = True)
    parser.add_argument('--folder_validation', type=str, default="validation_beamng2", 
                        help='Folder where validation data is stored (default: validation_beamng2)')
    parser.add_argument('--input_validation_results_file', type=str, default="validation_beamng2/validation_results.json",
                        help='Path to the validation results file (default: validation_beamng2/validation_results.json)')

    args = parser.parse_args()
    
    validate_results_after(args.input_validation_results_file,
                   args.simulator_name,
                   args.n_repeat,  
                   args.write_results_extended, 
                   args.folder_validation 
                 )