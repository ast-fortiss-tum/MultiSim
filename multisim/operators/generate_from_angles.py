
        
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
from opensbt.problem.adas_problem import ADASProblem
from operators.djanus_roadgen import JanusTestGenerator
from road_generator.custom_road_generator import CustomRoadGenerator
from road_generator.roads.road import Road
from sims.udacity.udacity_simulator import UdacitySimulator
from self_driving.utils.visualization import RoadTestVisualizer

import config

NUM_CONTROL_NODES  = config.NUM_CONTROL_NODES
MAP_SIZE = config.MAP_SIZE
MAX_ANGLE = config.MAX_ANGLE
SEG_LENGTH = config.SEG_LENGTH

import random


from road_generator.custom_road_generator import CustomRoadGenerator

angles =[90.0, 12.0, 51.0, 133.0, 204.01, 213.01]
road_generator = CustomRoadGenerator(map_size=250,
                            num_control_nodes=NUM_CONTROL_NODES,
                            seg_length=config.SEG_LENGTH)

road = road_generator.generate(starting_pos=UdacitySimulator.initial_pos,
                            angles=angles, 
                            seg_lengths=NUM_CONTROL_NODES*[SEG_LENGTH],
                            simulator_name=config.UDACITY_SIM_NAME)

road_test_visualizer = RoadTestVisualizer(map_size=MAP_SIZE)
road_test_visualizer.visualize_road_test(
    road=road, 
    folder_path="./operators", 
    filename="road_angles_{}".format(0), 
    plot_control_points=True
)
print(f"n control points: {len(road.control_points)}")
