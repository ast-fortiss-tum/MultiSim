import os
from opensbt.evaluation.fitness import *
from fitness.migration import CriticalAgreementDualMigration
from opensbt.problem.adas_multi_sim_aggreement_problem_diverse_predict import ADSMultiSimAgreementProblemDiversePredict
from opensbt.problem.adas_multi_sim_problem import ADSMultiSimProblem
from opensbt.problem.adas_problem import ADASProblem
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.simulation.dummy_simulation_od import DummySimulator3
from fitness.fitness import CRITICAL_XTE, MaxAvgXTEFitness, MaxXTECriticality, MaxXTESteeringChangeFitness
from opensbt.problem.adas_multi_sim_aggreement_problem import ADSMultiSimAgreementProblem
from opensbt.problem.adas_multi_sim_aggreement_problem_diverse import ADSMultiSimAgreementProblemDiverse
from opensbt.problem.adas_multi_sim_disaggreement_problem_diverse import ADSMultiSimDisagreementProblemDiverse

from sims.udacity.udacity_simulator import UdacitySimulator
from fitness.migration import ConsensusMigration
from problem_utils.naming import generate_problem_name
from operators import operators, ambiegen_crossover
from config import MAX_SEG_LENGTH, MIN_SEG_LENGTH, NUM_CONTROL_NODES, ROAD_MUT_PROB, \
                   MAX_DISPLACEMENT,NUM_TRIALS_MUT, CROSS_RATE, SEED, EXPERIMENTAL_MODE, \
                    N_GENERATIONS, POPULATION_SIZE, MAXIMAL_EXECUTION_TIME, ARCHIVE_THRESHOLD, \
                    CLASSIFIER_DISAGREE_1, CLASSIFIER_DISAGREE_2
from math import ceil
from sims.beamng_simulation import BeamngSimulator
from sims.donkey_simulation import DonkeySimulator
from mock.mock_simulation import MockSimulator
from sims.udacity.udacity_simulator_remote import UdacitySimulatorRemote
from fitness import fitness
from fitness import migration

n_cp = NUM_CONTROL_NODES
n_generations = N_GENERATIONS
population_size = POPULATION_SIZE
maximal_execution_time = MAXIMAL_EXECUTION_TIME

def getExp66():
    problem = ADSMultiSimAgreementProblemDiversePredict(
        problem_name=f"Donkey_A{n_cp-2}_-180-180_XTE",
        scenario_path="",
        xl=[-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
        xu=[180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)] + 
                                [f"seg_length{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_functions=[
            BeamngSimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate,
            DonkeySimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate
        ],
        migrate_function=migration.CriticalAgreementDualMigrationDiverse(),
        simulation_time=30,
        sampling_time=0.25
    )
    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.archive_threshold = ARCHIVE_THRESHOLD
    config.seed = SEED
    config.maximal_execution_time = maximal_execution_time
    
    problem.problem_name = generate_problem_name(problem,
                                                fitness_name= "XTE_DIST_BOTH",
                                                prefix = "PREDICT_SAVE_MUTATION_MOO",
                                                suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")
    
    config.sampling = operators.RoadSampling(variable_length=True)

    config.mutation = operators.RoadMutation(mut_prob = ROAD_MUT_PROB, 
                                             max_displacement = MAX_DISPLACEMENT, 
                                             num_trials = NUM_TRIALS_MUT,
                                             variable_length = True,
                                             disagree_predictors= [CLASSIFIER_DISAGREE_1,
                                                                   CLASSIFIER_DISAGREE_2]
                                        )
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(
                                            cross_rate=CROSS_RATE,
                                            variable_length = True) 
    
    config.ideal = np.asarray([-3,-3,-1000, 0])
    config.nadir = np.asarray([0,0,0, 100])
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)


def getExp67():
    problem = ADSMultiSimAgreementProblemDiversePredict(
        problem_name=f"Donkey_A{n_cp-2}_-180-180_XTE",
        scenario_path="",
        xl=[-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
        xu=[180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)] + 
                                [f"seg_length{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_functions=[
            BeamngSimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate,
            DonkeySimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate
        ],
        migrate_function=migration.CriticalAgreementDualMigrationDiverse(),
        simulation_time=30,
        sampling_time=0.25,
        fitness_function_predict=fitness.MaxXTEFitnessPredict(diversify=True),
        disagree_classifiers= [CLASSIFIER_DISAGREE_1, 
                               CLASSIFIER_DISAGREE_2]
    )
    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.archive_threshold = ARCHIVE_THRESHOLD
    config.seed = SEED
    config.maximal_execution_time = maximal_execution_time
    
    problem.problem_name = generate_problem_name(problem,
                                                fitness_name= "XTE_DIST_BOTH",
                                                prefix = "PREDICT_SUR_MOO",
                                                suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")
    
    config.sampling = operators.RoadSampling(variable_length=True)

    config.mutation = operators.RoadMutation(mut_prob = ROAD_MUT_PROB, 
                                             max_displacement = MAX_DISPLACEMENT, 
                                             num_trials = NUM_TRIALS_MUT,
                                             variable_length = True
                                        )
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(
                                            cross_rate=CROSS_RATE,
                                            variable_length = True) 
    
    config.ideal = np.asarray([-3,-3,-1000, 0])
    config.nadir = np.asarray([0,0,0, 100])
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)
experiment_switcher = {
    66: getExp66,
    67: getExp67
}