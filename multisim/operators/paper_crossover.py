
# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import copy
from random import randint
import time
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
from shapely.geometry import Point

from road_generator.custom_road_generator import CustomRoadGenerator
from road_generator.roads.simulator_road import SimulatorRoad

import math
import numpy as np

from road_generator.road_generator import RoadGenerator
from road_generator.roads.road import Road
from road_generator.roads.road_polygon import RoadPolygon
from road_generator.roads.bbox import RoadBoundingBox
from road_generator.roads.catmull_rom import catmull_rom

from shapely.errors import ShapelyDeprecationWarning
import warnings

from config import ROAD_WIDTH
warnings.simplefilter("ignore", ShapelyDeprecationWarning)

from self_driving import road_utils
import config

from self_driving.utils.visualization import RoadTestVisualizer
##############

angles_all = [
    [90, 80, 100, 45, 90],
    [90,90,70,100,70],
    [90, 80, 100, 100, 70],
    [90,90,70,45, 90],
    [90,90,70,45, 100]
]

for angles in angles_all:
    seg_lengths = [15] *5

    gen = CustomRoadGenerator(map_size=250,
                                        num_control_nodes=len(angles) + 2,
                                        seg_length=config.SEG_LENGTH,
                                        max_angle=config.MAX_ANGLE)

    start_time = time.perf_counter()

    road = gen.generate(simulator_name=config.UDACITY_SIM_NAME,
                        angles = angles,
                        starting_pos=(0,0,0,0),
                        seg_lengths=seg_lengths)

    concrete_representation = road.get_concrete_representation(to_plot = True)
    print(time.perf_counter() - start_time)

    print(f"curvature: {road.compute_curvature()}")
    print(f"num turns: {road.compute_num_turns()}")
    f = plt.figure()
    ax = plt.gca()
    plt.axis('equal')

    road_test_visualizer = RoadTestVisualizer(map_size=250, ax=ax)
    road_test_visualizer.visualize_road_basic(road_points=concrete_representation, folder_path="./operators/", 
                                            filename=f"road_cx_paper_{str(angles)}",
                                            simulator="mock",
                                            plot_control_points=True,
                                            road=road,
                                            format="pdf")

    angles_computed = []
    print([p.x for p in road.control_points])
    print([p.y for p in road.control_points])
    print([p.z for p in road.control_points])

    for i in range(1,len(angles)):
        angles_computed.append(road.get_angle_points(first_p=road.control_points[i+1],
                                    second_p=road.control_points[i])
                                    )
    print(f"computed angles: {angles_computed}")
    plt.close(f)
    plt.clf()