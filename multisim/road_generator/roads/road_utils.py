import random
from typing import Tuple, List

import numpy as np
from shapely.geometry import Point
import warnings
from random import randint
from typing import List, Tuple

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point

from driving.udacity_road import UdacityRoad
from utils.randomness import set_random_seed
from utils.visualization import RoadTestVisualizer
from config import ROAD_WIDTH, NUM_SAMPLED_POINTS, NUM_CONTROL_NODES, MAX_ANGLE, SEG_LENGTH, DISPLACEMENT
from driving.catmull_rom import catmull_rom
from self_driving.global_log import GlobalLog
from driving.bbox import RoadBoundingBox

import math
import numpy as np

from driving.road import Road
from driving.road_polygon import RoadPolygon
from test_generators.test_generator import TestGenerator

warnings.simplefilter("ignore", ShapelyDeprecationWarning)

from driving.bbox import RoadBoundingBox
from driving.catmull_rom import catmull_rom
from driving.road_polygon import RoadPolygon
from utils.visualization import RoadTestVisualizer
from driving.udacity_road import UdacityRoad
from config import ROAD_WIDTH, NUM_SAMPLED_POINTS, NUM_CONTROL_NODES, MAX_ANGLE, SEG_LENGTH


def get_closest_control_point(point: Tuple[float, float], cp: List[Tuple[float, float]]) -> int:
    nodes = np.asarray(cp)
    dist_2 = np.sum((nodes - point) ** 2, axis=1)
    return int(np.argmin(dist_2))


def mutate_road(index: int, cp: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    temp = list(cp[index])
    print("before mutation: ", temp)

    if np.random.rand() < 0.5:
        temp[0] = temp[0] + DISPLACEMENT
    else:
        temp[1] = temp[1] + DISPLACEMENT

    print("after mutation: ", temp)
    cp[index] = tuple(temp)

    return cp


def is_valid(control_nodes, sample_nodes):
    return (RoadPolygon.from_nodes(sample_nodes).is_valid() and
            RoadBoundingBox(bbox_size=(0, 0, 250, 250)).contains(RoadPolygon.from_nodes(control_nodes[1:-1])))


if __name__ == '__main__':
    control_points_road = [(125.0, 0.0, -28.0, 8.0), (125.0, 25.0, -28.0, 8.0), (125.0, 50.0, -28.0, 8.0),
                           (128.91086162600578, 74.69220851487844, -28.0, 8.0),
                           (104.15415990746652, 78.17153603888008, -28.0, 8.0),
                           (79.29111252325968, 80.78474762057142, -28.0, 8.0),
                           (60.712491886324834, 97.51301277954289, -28.0, 8.0),
                           (36.2588018679797, 92.3152205090989, -28.0, 8.0),
                           (11.805111849634557, 87.11742823865492, -28.0, 8.0)]

    point_start = (60, 88, -28.0, 8.0)

    road_test_visualizer = RoadTestVisualizer(map_size=250)

    print(control_points_road)

    condition = True
    control_nodes = control_points_road
    while condition:
        control_nodes = control_nodes[1:]
        sample_nodes = catmull_rom(control_nodes, NUM_SAMPLED_POINTS)
        if is_valid(control_nodes, sample_nodes):
            condition = False

    road_points = [Point(node[0], node[1]) for node in sample_nodes]
    control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]

    road_test_visualizer.visualize_road_test(
        road=UdacityRoad(road_width=ROAD_WIDTH,
                         road_points=road_points,
                         control_points=control_points),
        folder_path='../',
        filename='road',
        plot_control_points=False
    )

    closest = get_closest_control_point(tuple([point_start[0], point_start[1]]),
                                        [tuple([node[0], node[1]]) for node in control_nodes])
    print(closest)

    mutated = mutate_road(closest, control_points_road)

    print(mutated)

    condition = True
    control_nodes = mutated
    while condition:
        control_nodes = control_nodes[1:]
        sample_nodes = catmull_rom(control_nodes, NUM_SAMPLED_POINTS)
        if is_valid(control_nodes, sample_nodes):
            condition = False

    road_points = [Point(node[0], node[1]) for node in sample_nodes]
    control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]

    road_test_visualizer.visualize_road_test(
        road=UdacityRoad(road_width=ROAD_WIDTH, road_points=road_points, control_points=control_points),
        folder_path='../',
        filename='road',
        plot_control_points=False
    )