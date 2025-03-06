import os
from opensbt.evaluation.fitness import *
from fitness.migration import ConsensusMigrationWeak, CriticalAgreementDualMigration
from opensbt.problem.adas_multi_sim_problem import ADSMultiSimProblem
from opensbt.problem.adas_problem import ADASProblem
from opensbt.experiment.experiment import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.simulation.dummy_simulation_od import DummySimulator3

# for the mock experiment
from fitness.fitness import CRITICAL_XTE, MaxAvgXTEFitness, MaxXTECriticality, MaxXTESteeringChangeFitness
from problem_utils.naming import generate_problem_name
from sims.udacity.udacity_simulator import UdacitySimulator
from fitness import fitness
from operators import operators, ambiegen_crossover
from config import NUM_CONTROL_NODES, MUT_PROB, MAX_DISPLACEMENT
from mock.mock_simulation import MockSimulator
from math import ceil

def getExp4() -> Experiment:
    problem = ADASProblem(
                          problem_name="Dummy_DIST_V",
                          scenario_path="",
                          xl=[0, 1, 0, 1],
                          xu=[360, 4,360, 2],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessMinDistanceVelocity(),
                          critical_function=CriticalAdasDistanceVelocity(),
                          simulate_function=DummySimulator3.simulate,
                          simulation_time=5,
                          sampling_time=0.25
                          )
    config = DefaultSearchConfiguration()
    config.population_size = 10
    config.n_generations = 10

    config.ideal = np.asarray([0,-5])    # worst (=most critical) fitness values
    config.nadir = np.asarray([2.5,0])    # worst (=most critical) fitness values

    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

def getExp5() -> Experiment:
    from opensbt.simulation.dummy_simulation import DummySimulator2, DummySimulator1

    problem = ADSMultiSimProblem(
                          problem_name="Multi_Dummy_DualMigration_DIST_V",
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0, 1, 0, 1],
                          xu=[360, 10,360, 5],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessAdaptedDistanceSpeed(),
                          critical_function=CriticalAdasAdaptedDistanceVelocity(),
                          simulate_functions = 
                           [
                            DummySimulator3.simulate,
                            DummySimulator1.simulate],
                          migrate_function=CriticalAgreementDualMigration(),
                          simulation_time=10,
                          sampling_time=0.25
                          )
    config = DefaultSearchConfiguration()
    config.population_size = 20
    config.n_generations = 20

    config.ideal = np.asarray([0,-5])    # worst (=most critical) fitness values
    config.nadir = np.asarray([2.5,0])    # worst (=most critical) fitness values

    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

def getExp6() -> Experiment:
    from opensbt.simulation.dummy_simulation import DummySimulator2, DummySimulator1
    from opensbt.problem.adas_multi_sim_aggreement_problem import ADSMultiSimAgreementProblem
    from fitness.migration import ConsensusMigration
    from opensbt.evaluation.critical import Critical

    class CriticalAdasDistance(Critical):
        def eval(self, vector_fitness, simout: SimulationOutput = None):
            if (vector_fitness[0] < 1):
                return True
            else:
                return False
        
    problem = ADSMultiSimAgreementProblem(
                          problem_name="Multi_Dummy_Dual_MOO_DIST_V",
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0, 1, 0, 1],
                          xu=[360, 10,360, 5],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessMinDistance(),
                          critical_function=CriticalAdasDistance(),
                          simulate_functions = 
                           [ 
                            DummySimulator3.simulate,
                            DummySimulator1.simulate],
                          migrate_function=CriticalAgreementDualMigration(),
                          simulation_time=10,
                          sampling_time=0.25
                          ) 
    config = DefaultSearchConfiguration()
    config.population_size = 10
    config.n_generations = 10

    # note that combined fitness function is three dimensional
    config.ideal = np.asarray([0,0])  
    config.nadir = np.asarray([10,10]) 

    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

def getExp7() -> Experiment:
    from opensbt.simulation.dummy_simulation import DummySimulator2, DummySimulator1
    from opensbt.problem.adas_multi_sim_aggreement_problem import ADSMultiSimAgreementProblem
    from fitness.migration import ConsensusMigration
    from opensbt.evaluation.critical import Critical

    class CriticalAdasDistance(Critical):
        def eval(self, vector_fitness, simout: SimulationOutput = None):
            if (vector_fitness[0] < 1):
                return True
            else:
                return False
        
    problem = ADSMultiSimProblem(
                          problem_name="Multi_Consens_Weak_DIST_V",
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0, 1, 0, 1],
                          xu=[360, 10,360, 5],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessMinDistance(),
                          critical_function=CriticalAdasDistance(),
                          simulate_functions = 
                           [ 
                            DummySimulator3.simulate,
                            DummySimulator1.simulate],
                          migrate_function=ConsensusMigrationWeak(),
                          simulation_time=10,
                          sampling_time=0.25
                          ) 
    config = DefaultSearchConfiguration()
    config.population_size = 4
    config.n_generations = 4

    # note that combined fitness function is three dimensional
    config.ideal = np.asarray([0,0])  
    config.nadir = np.asarray([10,10]) 

    experiment = Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment

def getExp8():
    from sims.beamng_simulation import BeamngSimulator
    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_function=MockSimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )
    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE")

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = 5
    config.population_size = 5
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=0.5, max_displacement=8)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=0.7) 
    config.prob_crossover  = 0
    config.eta_crossover = 30

    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)


def getExp9():
    from fitness import fitness
    from fitness import migration
    
    n_cp = NUM_CONTROL_NODES

    problem = ADSMultiSimProblem(
        problem_name=f"MM_A{n_cp-2}_0-360_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=fitness.MaxXTEFitness(),
        critical_function=fitness.MaxXTECriticality(),
        simulate_functions=[
            MockSimulator.simulate,
            MockSimulator.simulate
        ],
        migrate_function=migration.CriticalAgreementDualMigration(),
        simulation_time=30,
        sampling_time=0.25
    )
    problem.problem_name = generate_problem_name(problem,
                                                 fitness_name= "XTE")

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = 2
    config.population_size = 2
    
    config.sampling = operators.RoadSampling()
    config.mutation = operators.RoadMutation(mut_prob=0.8, max_displacement=6, num_trials=2)
    config.crossover = ambiegen_crossover.OnePointRoadCrossover(cross_rate=0.6)
    
    
    config.ideal = np.asarray([-3])#,-20])    # worst (=most critical) fitness values
    config.nadir = np.asarray([0])#,0])    # worst (=most critical) fitness values
    config.n_repopulate_max = ceil(0.1 * config.population_size)

    return Experiment(problem=problem,
                            algorithm=AlgorithmType.NSGAIID,
                            search_configuration=config)

experiment_switcher = {
    4: getExp4,
    5: getExp5,
    6: getExp6,
    7: getExp7,
    8: getExp8,
    9: getExp9
}
