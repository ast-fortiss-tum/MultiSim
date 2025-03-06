from pathlib import Path
import sys
import matplotlib
import pymoo
import numpy as np
import os

from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.model_ga.individual import IndividualSimulated
from problem_utils.naming import generate_problem_name
from sims.udacity.udacity_simulator import UdacitySimulator
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from mock.mock_simulation import MockSimulator
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.utils.evaluation import evaluate_individuals

from sims.donkey_simulation import DonkeySimulator
from sims.beamng_simulation import BeamngSimulator

from fitness.fitness import MaxXTEFitness, MaxXTECriticality
from config import NUM_CONTROL_NODES
from opensbt.problem.adas_problem import ADASProblem

from datetime import datetime
#############

n_cp = NUM_CONTROL_NODES

problem = ADASProblem(
    problem_name=f"Beamng_Flakiness_A{n_cp-2}_XTE",
    scenario_path="",
    xl=[0]*(n_cp-2),
    xu=[360]*(n_cp-2),
    simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
    fitness_function=MaxXTEFitness(),
    critical_function=MaxXTECriticality(),
    simulate_function=DonkeySimulator.simulate,
    simulation_time=30,
    sampling_time=0.25
)
problem.problem_name = generate_problem_name(problem,
                                             category="Flakiness",
                                             fitness_name="XTE")

angles = [22.391157,
          276.080206,
          205.239574,
          212.991184,
          276.774623,
          265.036182,
          279.134826]
n_repeat = 10

#####################

inds = [IndividualSimulated(X=angles) for _ in range(0,n_repeat)]
pop = PopulationExtended(individuals=inds)

evaluate_individuals(population = pop, problem=problem)

fvalues = pop.get("F")

for ind in pop:
    print(f"X: {ind.get('X')}")
    print(f"F: {ind.get('F')}")


print(f"fitness values: {fvalues}")

data = {
        "avg_fitness" : np.sum(fvalues)/len(fvalues),
        "max_fitness": np.max(fvalues),
        "min_fitness": np.min(fvalues),
        "std_fitness" : np.std(fvalues),
        "critical_ratio": len(pop.divide_critical_non_critical()[0])/len(pop),
        "road_angles" : angles,
        "all_fitness" : fvalues.tolist()
        }

#############
# write

import json
flakiness_folder = f"./scripts/out/{problem.problem_name}/"
Path(flakiness_folder).mkdir(parents=True, exist_ok=True)
time  = datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") 
with open(flakiness_folder + os.sep + f"flaky_results_{time}.json", 'w') as f:
    json.dump(data, f, indent=4)