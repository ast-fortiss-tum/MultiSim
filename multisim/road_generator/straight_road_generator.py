# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import copy
from random import randint
from typing import List, Tuple, Union
from shapely.geometry import Point
from config import ROAD_WIDTH

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

from self_driving import road_utils

import config
import time
from self_driving.utils.visualization import RoadTestVisualizer

warnings.simplefilter("ignore", ShapelyDeprecationWarning)

class StraightRoadGenerator(RoadGenerator):
    """Generate random roads given the configuration parameters. The"""

    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20

    def __init__(
        self,
        map_size: int,
        num_control_nodes=8,
        max_angle=90,
        seg_length=25,
        num_spline_nodes=20,
        initial_node=(0.0, 0.0, 0.0, 4),
        bbox_size=(0, 0, 250, 250),
    ):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4 and len(bbox_size) == 4
        self.map_size = map_size
        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.road_bbox = RoadBoundingBox(bbox_size=bbox_size)
        self.road_to_generate = None

        self.previous_road: Road = None

    def set_max_angle(self, max_angle: int) -> None:
        assert max_angle > 0, "Max angle must be > 0. Found: {}".format(max_angle)
        self.max_angle = max_angle

    def is_valid(self, control_nodes, sample_nodes):
        return RoadPolygon.from_nodes(
            sample_nodes
        ).is_valid() and self.road_bbox.contains(
            RoadPolygon.from_nodes(control_nodes[1:-1])
        )

    def generate(self, *args, **kwargs) -> Road:
        """
        Needs a list of integer angles in the kwargs param `angles`.
        Optionally takes another list of segment lengths in `seg_lengths` key of kwargs.
        """

        if self.road_to_generate is not None:
            road_to_generate = copy.deepcopy(self.road_to_generate)
            self.road_to_generate = None
            return road_to_generate

        sample_nodes = None

        if "starting_pos" in kwargs:
            starting_pos = kwargs["starting_pos"]
        else:
            starting_pos = self.initial_node

        spos = starting_pos
        control_nodes = [spos] + [(spos[0], spos[0] + i*10 ,spos[2], 4) for i in range(0,20,1)]
        control_nodes = control_nodes[0:]
        sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)

        road_points = [Point(node[0], node[1], node[2]) for node in sample_nodes]
        control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]
        _, _, _, width = self.initial_node

        self.previous_road = road_utils.get_road(road_width=width, 
                                                 road_points=road_points, 
                                                 control_points=control_points, 
                                                 simulator_name=kwargs["simulator_name"])
        
        return self.previous_road

    def _get_initial_point(self) -> Point:
        return Point(self.initial_node[0], self.initial_node[1])

if __name__ == "__main__":

    map_size = 250

    # set_random_seed(seed=0)
    angles = [60,-10,0,90,90]
    seg_lengths = [40 for _ in angles]
    
    gen = StraightRoadGenerator(map_size=250,
                                        num_control_nodes=len(angles),
                                        seg_length=config.SEG_LENGTH,
                                        max_angle=config.MAX_ANGLE)

    start_time = time.perf_counter()

    road = gen.generate(simulator_name=config.UDACITY_SIM_NAME,
                        angles = angles,
                        starting_pos=(0,0,0,4),
                        seg_lengths=seg_lengths)

    concrete_representation = road.get_concrete_representation()
    print(time.perf_counter() - start_time)

    road_test_visualizer = RoadTestVisualizer(map_size=map_size)
    road_test_visualizer.visualize_road_test(road=road, folder_path="./road_generator/", filename="road_straight")
