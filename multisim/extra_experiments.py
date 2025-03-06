from math import ceil
import os
from sims.donkey_simulation import DonkeySimulator
from mock.mock_simulation import MockSimulator
from opensbt.evaluation.fitness import *
from fitness.migration import CriticalAgreementDualMigration
from opensbt.problem.adas_multi_sim_problem import ADSMultiSimProblem


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
                    MAX_DISPLACEMENT, NUM_TRIALS_MUT,CROSS_RATE, SEED

n_cp = NUM_CONTROL_NODES

problem = ADASProblem(
    problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
    scenario_path="",
    xl=[-180]*(n_cp-2),
    xu=[180]*(n_cp-2),
    simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
    fitness_function=fitness.MaxXTEFitness(),
    critical_function=fitness.MaxXTECriticality(),
    simulate_function=UdacitySimulator.simulate,
    simulation_time=30,
    sampling_time=0.25,
)

NUM_SAMPLES = 1

# RANDOM SAMPLING 

'''
RS
'''
def getExp0():
    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = 1
    config.population_size = NUM_SAMPLES
    config.sampling = operators.RoadSampling()
    config.seed = SEED
    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values
    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}",
                                                 prefix = "RS_")

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
                    
'''
NSGAII 
'''
def getExp1():
    config = DefaultSearchConfiguration()
    config.n_generations = 5
    config.population_size = 5
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED

    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_config{config.seed}")

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
'''
 NSGA-II-D
'''
def getExp2():
    diversify = True

    config = DefaultSearchConfiguration()
    config.n_generations = 10
    config.population_size = 10
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED
    config.archive_threshold = 5
    config.n_repopulate_max = 0.3

    if diversify:
        config.ideal = np.asarray([-3,-1000])#,-20])    # worst (=most critical) fitness values
        config.nadir = np.asarray([0,0])#,0])    # w
    else:
        config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
        config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_DIST",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")
    
    problem.set_fitness_function(
        fitness.MaxXTEFitness(diversify=diversify)
    )
    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)
experiment_switcher = { 
    0:getExp0, 
    1:getExp1,
    2:getExp2 
}
