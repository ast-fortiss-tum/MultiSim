
import copy
import csv
import json
import os
import pickle
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
from opensbt.problem.adas_problem import ADASProblem
from operators.djanus_roadgen import JanusTestGenerator
from road_generator.custom_road_generator import CustomRoadGenerator
from self_driving.road import Road
from sims.udacity.udacity_simulator import UdacitySimulator
from self_driving.utils.visualization import RoadTestVisualizer
from operators.utils.validate_angle import max_angle_preserved
import logging as log
import config
import random
import math
from datetime import datetime

NUM_CONTROL_NODES  = config.NUM_CONTROL_NODES
MAP_SIZE = config.MAP_SIZE
MAX_ANGLE = config.MAX_ANGLE
SEG_LENGTH = config.SEG_LENGTH

class RoadSampling(Sampling):
    def __init__(self, variable_length = False):
        super().__init__()
        self.variable_length = variable_length

    def _do(self, problem, n_samples, **kwargs):
        X = np.full(n_samples,None, dtype=object)
        for i in range(0,n_samples):
            road = RoadSampling.generate_road(NUM_CONTROL_NODES)
        
            # road_test_visualizer = RoadTestVisualizer(map_size=MAP_SIZE)
            # road_test_visualizer.visualize_road_test(
            #     road=road, folder_path="operators/output/", 
            #     filename="road_{}".format(i), 
            #     plot_control_points=True
            # )
            # get angle representation for transformations
            angles = RoadSampling.get_angles_road(road)
            if self.variable_length:
                angles = angles + [SEG_LENGTH] * len(angles)

            # print(f"angle encoding is: {angles}")
            X[i] = np.asarray(angles)
        return X
    
    def get_angles_road(road):
        angles = []
        for j in range(2,NUM_CONTROL_NODES):
            angles.append(road.get_angle_points(first_p=road.control_points[j],
                                        second_p=road.control_points[j-1],
                                        shift=0)
                                        )
        return angles
    
    def generate_road(num_control_nodes):
        tg = JanusTestGenerator(map_size=MAP_SIZE,
                                simulator_name="donkey", # the sim name is not further used
                                num_control_nodes=num_control_nodes, 
                                max_angle=MAX_ANGLE)
        
        road : Road = tg.generate()
        return road


class RoadMutation(Mutation):
    def __init__(self, max_displacement = 9, 
                       mut_prob = 0.5, 
                       num_trials = 10,
                       variable_length = False,
                       disagree_predictors = None): # the last length/2 values are seg length values
        super().__init__()
        self.max_displacement = max_displacement
        self.mut_prob = mut_prob
        self.num_trials = num_trials
        self.time = datetime.now().strftime(
                            "%d-%m-%Y_%H-%M-%S")
        self.variable_length = variable_length
        self.disagree_predictors = disagree_predictors

    # change the angle by max +/- degree
    def _do(self, problem, X, **kwargs):
        Y= np.full(len(X), None, dtype=object)
        # mutate road here
        # for each individual
        for i in range(len(X)):
            # print(f"[RoadMutation] Mutating: {X[i]}")
            is_valid = False
            trial = 0

            while not is_valid and trial < self.num_trials:
                print(f"[RoadMutation] Executing mutation for {i}.th individual.")
                if self.variable_length:
                    n = len(X[i])

                    angles = copy.deepcopy(X[i])[0:n//2]
                    seg_lengths = copy.deepcopy(X[i])[n//2:]

                    index = random.randint(1, problem.n_var/2 - 1) # dont mutate first angle bc of fairness
                    # first we decide whether we do mutation of angles
                    if random.random() < self.mut_prob:
                        displacement_min, displacement_max = self.get_max_allowed_displacement(angles, 
                                                                                            index, 
                                                                                            threshold=self.max_displacement, 
                                                                                            max_angle=MAX_ANGLE)

                        # print(f"[RoadMutation] Trying min, max displacement: { displacement_min, displacement_max}")
                        # then we decide the direction
                        angles[index]  = angles[index] + random.randint(- displacement_min, displacement_max)
                        # Y[i] = angles
                    else:
                        is_valid = True
                        print(f"[RoadMutation] No mutation of angles.\n")
                        # Y[i] = angles

                    # we decide whether we mutate segment lengths
                    if random.random() < self.mut_prob:
                        offset = 2 # mutate only the beginning segments
                        index_seg = random.randint(0,problem.n_var//2 - offset)
                        length = random.randint(config.MIN_SEG_LENGTH, config.MAX_SEG_LENGTH)
                
                        seg_lengths[index_seg] = length
                
                    Y[i] = np.concatenate((angles, seg_lengths))

                    # print("updated road:", Y[i])
                else:
                    angles = copy.deepcopy(X[i])
                    seg_lengths = [config.SEG_LENGTH for _ in range(len(angles))]
                    index = random.randint(1, problem.n_var - 1) # dont mutate first angle bc of fairness
                    
                    # first we decide whether we do mutation
                    if random.random() < self.mut_prob:
                        displacement_min, displacement_max = self.get_max_allowed_displacement(angles, 
                                                                                            index, 
                                                                                            threshold=self.max_displacement, 
                                                                                            max_angle=MAX_ANGLE)

                        # print(f"[RoadMutation] Trying min, max displacement: { displacement_min, displacement_max}")
                        # then we decide the direction
                        angles[index]  = angles[index] + random.randint(- displacement_min, displacement_max)
                        Y[i] = angles
                    else:
                        # is_valid = True
                        print(f"[RoadMutation] No mutation.\n")
                        Y[i] = angles
                        continue

                # check road
                road_generator = CustomRoadGenerator(map_size=config.MAP_SIZE,
                                                num_control_nodes=config.NUM_CONTROL_NODES - 1,
                                                seg_length=config.SEG_LENGTH)
                road: Road = road_generator.generate(starting_pos=UdacitySimulator.initial_pos,
                                            angles=angles,
                                           simulator_name=config.UDACITY_SIM_NAME,
                                           seg_lengths = seg_lengths)

                no_overlap_and_in_box = road.is_valid()
                print(f"[RoadMutation] no overlap and in box: {no_overlap_and_in_box}")
                from config import MUTATION_CHECK_VALID_ALL
                if MUTATION_CHECK_VALID_ALL:
                    max_angles_ok = RoadMutation.angles_preserved_single_road(Y[i], MAX_ANGLE)
                else:
                    max_angles_ok = max_angle_preserved(Y[i][index-1], 
                                                    Y[i][index], 
                                                    MAX_ANGLE)[0] if index -1 >= 0 else True \
                    and \
                        max_angle_preserved(Y[i][index], 
                                            Y[i][index+1], 
                                            MAX_ANGLE)[0] \
                            if index +1 < len(angles) else True
                
                # if Road.is_too_sharp(road.get_concrete_representation(to_plot=True)):
                #     print("Road is too sharp")
                #     input()
                
                is_valid =  no_overlap_and_in_box and max_angles_ok
                
                if not is_valid:
                    if not no_overlap_and_in_box:
                        pass
                        print(f"[RoadMutation] Overlap.")
                    if not max_angles_ok:
                        pass
                        print(f"[RoadMutation] Max angles not ok.")
                    # print(f"[RoadMutation] Repeating mutation.\n")
                    trial += 1
                else:
                    pass
                    print(f"[RoadMutation] Road is valid. No repetition. ")
            
            is_disagreement = False

            if self.disagree_predictors is not None:
                from operators import disagree_predictor

                is_disagreement, _ = disagree_predictor.predict(Y[i],
                                           predictor_paths=self.disagree_predictors,
                                           problem_name=problem.problem_name,
                                           threshold=problem.threshold_uncertainty)
                        
            if not is_valid or is_disagreement:
                # if disagreement we also generate new road

                # replace not valid road by randomly generated road
                road = RoadSampling.generate_road(NUM_CONTROL_NODES)
                angles = RoadSampling.get_angles_road(road)

                if self.variable_length:
                    seg_lengths = [SEG_LENGTH for _ in range(len(angles))]
                    Y[i] = np.concatenate((angles, seg_lengths))
                else:
                    Y[i] = angles
                print("[RoadMutation] Mutation not possible. Random road generated.")
        print("MUTATION finished")
        return Y
    
    def get_max_allowed_displacement(self, angles, index, threshold, max_angle):
        max_feasable_pre = min(abs(max_angle - angles[index - 1]), threshold)
        max_feasable_post = min(abs(max_angle - angles[index - 1]), threshold)

        min_feasable_pre = max_feasable_pre
        min_feasable_post = max_feasable_post
        return math.floor(max(min_feasable_pre, min_feasable_post)), math.floor(min(max_feasable_post, max_feasable_pre))
    
    @staticmethod
    def angles_preserved_single_road(angles, max_angle):
        for i in range(1,len(angles)):
            preserved, dif = max_angle_preserved(angles[i-1], angles[i], max_angle = max_angle)
            if not preserved:
                return False, dif
        return True, dif
    
class MyNoCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    # def do(self, problem, X, **kwargs):
        
    #     print(X.shape)

    #     # print(f"n_matings, n_var: {n_matings}, {n_var}")
    #     # Because there the number of parents and offsprings are equal it keeps the shape of X
    #     Y = np.full_like(X, None, dtype=object)
    #     for i,ind in enumerate(X):
    #         Y[i]  = X[i]
    #     print(Y)
    #     # for each mating provided
    #     res = Y
    #     # res = Population.create(*[np.random.choice(parents) for parents in X])

    #     return res
    
    def _do(self, problem, X, **kwargs):
         # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape
        print(f"X.shape: {X.shape}")
        print(f"input: {X}")
        # The output owith the shape (n_offsprings, n_matings, n_var)
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            angles_a, angles_b = X[0, k], X[1, k]
            print(f"selected parent 1: {angles_a}")
            print(f"selected parent 2: {angles_b}")
            angles_a[0] = 0
            angles_a[1] = 0

            # do nothing
            Y[0, k], Y[1, k] = angles_a, angles_b
        
        print(f"output: {Y}")
        return Y
        
problem = ADASProblem(
        problem_name="Udacity_4A_0-90_XTE_DIVERSE",
        scenario_path="",
             xl=[0, 0, 0, 0],
             xu=[90, 90, 90,90],
        simulation_variables=[
            "angle1",
            "angle2",
            "angle3",
            "angle4"],
        fitness_function=MaxXTEFitnessDiverse(),
        critical_function=MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )

if __name__ == "__main__":
    from pymoo.core.individual import Individual
    # RoadSampling().do(problem,
    #             n_samples=10)

    cand = [
        np.asarray([1,2,3,4]),
        np.asarray([10,20,30,40]),
        np.asarray([11,22,34,44]),
        np.asarray([5,55,55,55]),
    ]
    individuals = [Individual(X=ind) for ind in cand]
    pop = Population(individuals=individuals)

    cx_pop = MyNoCrossover().do(problem, pop)

    for ind in cx_pop:
        print(ind.get("X"))
