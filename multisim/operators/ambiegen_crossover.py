
import copy
import numpy as np
import random
# For Python 3.6 we use the base keras
import random
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population

import keras

from fitness.fitness import MaxXTECriticality, MaxXTEFitnessDiverse
from opensbt.evaluation import fitness
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.problem.adas_problem import ADASProblem
from operators.djanus_roadgen import JanusTestGenerator
from road_generator.custom_road_generator import CustomRoadGenerator
from road_generator.roads.road import Road
from sims.udacity.udacity_simulator import UdacitySimulator
from self_driving.utils.visualization import RoadTestVisualizer
from operators.utils.validate_angle import max_angle_preserved
import logging as log
import config
from operators.operators import RoadSampling

NUM_CONTROL_NODES  = config.NUM_CONTROL_NODES
MAP_SIZE = config.MAP_SIZE
MAX_ANGLE = config.MAX_ANGLE


class OnePointRoadCrossover(Crossover):

    def __init__(self, cross_rate, variable_length = False):
        super().__init__(2, 2)
        self.cross_rate = cross_rate
        self.variable_length = variable_length

    def angles_preserved_single_road(self, angles, max_angle):
        for i in range(1,len(angles)):
            preserved, dif = max_angle_preserved(angles[i-1], angles[i], max_angle = max_angle)
            if not preserved:
                return False, dif
        return True, dif

    # identifies all crossover point where angle difference after crossover is preserved
    def indeces_angle_preserved(self, angles_a, angles_b, max_angle):
        indeces = []
        for i in range(2,len(angles_a)):
            if max_angle_preserved(angles_a[i-1], angles_b[i], max_angle = max_angle)[0] and \
                        max_angle_preserved(angles_b[i-1], angles_a[i], max_angle = max_angle)[0]:
                indeces.append(i)
        return indeces
    
    def _do(self, problem, X, **kwargs):
         # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape
        # print(f"input: {X}")
        # The output owith the shape (n_offsprings, n_matings, n_var)
        Y = np.full_like(X, None, dtype=object)
        
        # for each mating provided
        for k in range(n_matings):
            if self.variable_length:
                m = len(X[0, k])

                angles_a, angles_b = X[0, k][0:m//2], X[1, k][:m//2]
                lengths_a, lengths_b = X[0, k][m//2:], X[1, k][m//2:]
                
                r = np.random.random()
                if r < self.cross_rate:
                    # n_var = problem.n_var
                    # n = np.random.randint(1, n_var)
                    n_options = self.indeces_angle_preserved(angles_a, angles_b, max_angle = MAX_ANGLE)
                    if len(n_options) != 0:
                        n = random.choice(n_options)
                        # check if angle between segments preserved after crossover
                        off_a = np.concatenate([angles_a[:n], angles_b[n:]])
                        off_b = np.concatenate([angles_b[:n], angles_a[n:]])

                        # exchange also segment lengths
                        off_length_a = np.concatenate([lengths_a[:n], lengths_b[n:]])
                        off_length_b = np.concatenate([lengths_b[:n], lengths_a[n:]])
                        # print(f"off_a: {off_a}")
                        # print(f"off_b: {off_b}")
                        Y[0, k], Y[1, k] = off_a + off_length_a, off_b + off_length_b

                        print(f"[OnePointRoadCrossOver] Cx done at index {n}.")
                        print(f"[OnePointRoadCrossOver] parent1: {angles_a + lengths_a}")
                        print(f"[OnePointRoadCrossOver] parent2: {angles_b + lengths_b}")
                        print(f"[OnePointRoadCrossOver] off1: {off_a + off_length_a}")
                        print(f"[OnePointRoadCrossOver] off2: {off_b + off_length_b}\n")
                    else:
                        log.info("[OnePointRoadCrossOver] angle not preserved, not doing crossover. Trying next mating pair.\n")
                        Y[0, k], Y[1, k] =  X[0, k], X[1, k] 
                else:
                    Y[0, k], Y[1, k] = np.concatenate((angles_a,lengths_a)), np.concatenate((angles_b,lengths_b))
                    
                off1_preserved, diff_1 = self.angles_preserved_single_road(angles_a, MAX_ANGLE)
                off2_preserved, diff_2 = self.angles_preserved_single_road(angles_b, MAX_ANGLE)

                try:
                    assert  off1_preserved and \
                        log.info("[OnePointRoadCrossOver] angle not preserved, not doing crossover. Trying next mating pair.\n")
                except Exception as e:
                    print(e)
                
                from config import REPLACE_ROAD_CX

                if not off1_preserved:
                    # generated new roads
                    road = RoadSampling.generate_road(NUM_CONTROL_NODES)
                    angles1 = RoadSampling.get_angles_road(road)
                    off1_preserved = angles1

                    if REPLACE_ROAD_CX:
                        Y[0, k] =  np.concatenate((angles1, [20]*5))
                        print("[OnePointRoadCrossOver] First partner replaced")

                if not off2_preserved:
                    road = RoadSampling.generate_road(NUM_CONTROL_NODES)
                    angles2 = RoadSampling.get_angles_road(road)
                    off2_preserved = angles2

                    if REPLACE_ROAD_CX:
                        Y[1, k] = np.concatenate((angles2, [20]*5))
                        print("[OnePointRoadCrossOver] Second partner replaced")
            else:
                angles_a, angles_b = X[0, k], X[1, k]
                r = np.random.random()
                if r < self.cross_rate:
                    # n_var = problem.n_var
                    # n = np.random.randint(1, n_var)
                    n_options = self.indeces_angle_preserved(angles_a, angles_b, max_angle = MAX_ANGLE)
                    if len(n_options) != 0:
                        n = random.choice(n_options)
                        # check if angle between segments preserved after crossover
                        off_a = np.concatenate([angles_a[:n], angles_b[n:]])
                        off_b = np.concatenate([angles_b[:n], angles_a[n:]])
                        # print(f"off_a: {off_a}")
                        # print(f"off_b: {off_b}")
                        Y[0, k], Y[1, k] = off_a, off_b
                        print(f"[OnePointRoadCrossOver] Cx done at index {n}.")
                        print(f"[OnePointRoadCrossOver] parent1: {angles_a}")
                        print(f"[OnePointRoadCrossOver] parent2: {angles_b}")
                        print(f"[OnePointRoadCrossOver] off1: {off_a}")
                        print(f"[OnePointRoadCrossOver] off2: {off_b}\n")
                    else:
                        log.info("[OnePointRoadCrossOver] angle not preserved, not doing crossover. Trying next mating pair.\n")
                        Y[0, k], Y[1, k] =  X[0, k], X[1, k] 
                else:
                    Y[0, k], Y[1, k] = angles_a, angles_b
                    
                off1_preserved, diff_1 = self.angles_preserved_single_road(angles_a, MAX_ANGLE)
                off2_preserved, diff_2 = self.angles_preserved_single_road(angles_b, MAX_ANGLE)

                try:
                    assert  off1_preserved and \
                        off2_preserved, f"[OnePointRoadCrossOver] Max angle is not preserved after crossover. Differences are: {diff_1} and {diff_2}. Angles are: {angles_a} {angles_b}"
                except Exception as e:
                    print(e)
                
                from config import REPLACE_ROAD_CX

                if not off1_preserved:
                    # generated new roads
                    road = RoadSampling.generate_road(NUM_CONTROL_NODES)
                    angles1 = RoadSampling.get_angles_road(road)
                    off1_preserved = angles1

                    if REPLACE_ROAD_CX:
                        Y[0, k] = angles1
                        print("[OnePointRoadCrossOver] First partner replaced")

                if not off2_preserved:
                    road = RoadSampling.generate_road(NUM_CONTROL_NODES)
                    angles2 = RoadSampling.get_angles_road(road)
                    off2_preserved = angles2

                    if REPLACE_ROAD_CX:
                        Y[1, k] = angles2
                        print("[OnePointRoadCrossOver] Second partner replaced")

        # print(f"output: {Y}")
        return Y

if __name__ == "__main__":
    from pymoo.core.individual import Individual
    # RoadSampling().do(problem,
    #             n_samples=10)
    # n_cp = NUM_CONTROL_NODES

    # problem = ADASProblem(
    #     problem_name=f"Donkey_A{n_cp-2}_0-360_XTE",
    #     scenario_path="",
    #     xl=[0]*(n_cp-2),
    #     xu=[360]*(n_cp-2),
    #     simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
    #     fitness_function=fitness.MaxXTEFitness(diversify=True),
    #     critical_function=fitness.MaxXTECriticality(),
    #     simulate_function=MockSimulator.simulate,
    #     simulation_time=30,
    #     sampling_time=0.25,
    # )

    # cand = [
    #     np.asarray([1,2,3,4]),
    #     np.asarray([10,20,30,40]),
    #     np.asarray([11,22,34,44]),
    #     np.asarray([5,55,55,55]),
    # ]
    # individuals = [IndividualSimulated(X=ind) for ind in cand]
    # pop = Population(individuals=individuals)
    # pop =  np.asarray([[1,2,3,4],
    #         [10,20,30,40],
    #         [11,22,34,44],
    #         [5,55,55,55]])
    # opc = OnePointRoadCrossover(cross_rate=0.3)
    
    # cx_pop = opc._do(problem=problem,X=pop)

    # for ind in cx_pop:
    #     print(ind.get("X"))
