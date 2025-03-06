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
                    MAX_DISPLACEMENT, NUM_TRIALS_MUT,CROSS_RATE, SEED, \
                    MIN_SEG_LENGTH, MAX_SEG_LENGTH, EXPERIMENTAL_MODE, \
                    N_GENERATIONS, POPULATION_SIZE, MAXIMAL_EXECUTION_TIME, \
                    ARCHIVE_THRESHOLD

n_cp = NUM_CONTROL_NODES
n_generations = N_GENERATIONS
population_size = POPULATION_SIZE
maximal_execution_time = MAXIMAL_EXECUTION_TIME

# Donkey Single Simulation

def getExp13():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTESteeringChangeFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_STEER")

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=0.8, max_displacement=6, num_trials=2)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=0.6) 
    config.prob_crossover  = 0
    config.eta_crossover = 30

    config.ideal = np.asarray([-3,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.3 * config.population_size)
    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

def getExp14():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[-180]*(n_cp-2),
        xu=[180]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTESteeringChangeFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30

    config.ideal = np.asarray([-3,-20])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0, 0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_STEER",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

def getExp15():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTESteeringChangeFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30

    config.ideal = np.asarray([-3,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_STEER",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

def getExp16():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30

    config.ideal = np.asarray([-3, -1000])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0,0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_DIST",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

def getExp17():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=MockSimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, max_displacement=MAX_DISPLACEMENT, num_trials=NUM_TRIALS_MUT)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30

    config.ideal = np.asarray([-3, -1000])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0, 0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE_DIST",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

###############
# validation of simulators on given roads
def getExp20():
    from sims.donkey_simulation import DonkeySimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Validation_Donkey_Samples_A{n_cp-2}_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.prob_crossover  = 0
    config.eta_crossover = 30

    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

def getExp21():
    from sims.donkey_simulation import DonkeySimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Validation_Beamng_Matteo_A{n_cp-2}_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=DonkeySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size

    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

''' Experiment where road segment lengths are also varied. '''
def getExp1000():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
        xu=[180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)] + 
                                [f"seg_length{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate,

        simulation_time=30,
        sampling_time=0.25,
    )
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.maximal_execution_time = maximal_execution_time
    
    config.sampling = operators.RoadSampling(variable_length=True)
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, 
                                             max_displacement=MAX_DISPLACEMENT, 
                                             num_trials=NUM_TRIALS_MUT,
                                             variable_length=True)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE,variable_length=True) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30
    config.archive_threshold = ARCHIVE_THRESHOLD

    config.ideal = np.asarray([-3, -1000])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0,0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 prefix = "VARSEG",
                                                 fitness_name= "XTE_DIST",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

''' Experiment where road segment lengths are also varied. '''
def getExp1001():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[-180]*(n_cp-2) + [MIN_SEG_LENGTH]*(n_cp-2),
        xu=[180]*(n_cp-2) + [MAX_SEG_LENGTH]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)] + 
                                [f"seg_length{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(diversify=True),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=DonkeySimulator.simulate if not EXPERIMENTAL_MODE else MockSimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    config = DefaultSearchConfiguration()
    config.n_generations = n_generations
    config.population_size = population_size
    config.maximal_execution_time = maximal_execution_time

    config.sampling = operators.RoadSampling(variable_length=True)
    config.mutation = operators.RoadMutation(mut_prob=ROAD_MUT_PROB, 
                                             max_displacement=MAX_DISPLACEMENT, 
                                             num_trials=NUM_TRIALS_MUT,
                                             variable_length=True)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=CROSS_RATE,variable_length=True) 
    config.seed = SEED
    # config.prob_crossover  = 0
    # config.eta_crossover = 30
    config.archive_threshold = ARCHIVE_THRESHOLD

    config.ideal = np.asarray([-3, -1000])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0,0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.35 * config.population_size)

    problem.problem_name = generate_problem_name(problem,
                                                 prefix = "VARSEG",
                                                 fitness_name= "XTE_DIST",
                                                 suffix= f"gen{config.n_generations}_pop{config.population_size}_seed{config.seed}")

    
    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)
###########

experiment_switcher = { 
    13: getExp13,
    13: getExp13,
    14: getExp14,
    15: getExp15,
    16: getExp16,
    17: getExp17,
    20: getExp20,
    21: getExp21,
    1000: getExp1000,
    1001: getExp1001
}
