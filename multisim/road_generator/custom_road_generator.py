# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import copy
from random import randint
import time
from typing import List, Tuple, Union
from shapely.geometry import Point

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


class CustomRoadGenerator(RoadGenerator):
    """Generate random roads given the configuration parameters."""

    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20

    def __init__(
        self,
        map_size: int,
        num_control_nodes=8,
        max_angle=90,
        seg_length=25,
        num_spline_nodes=20,
        initial_node=(0.0, 0.0, 0.0, config.ROAD_WIDTH),
        bbox_size=(0, 0, 250, 250),
        max_angles = None
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

        if max_angles is None:
            self.max_angles = [max_angle for i in range(num_control_nodes)]
        else:
            self.max_angles = max_angles
        
        # print(f"[CustomRoadGenerator] num_control_nodes:", self.num_control_nodes)

    def set_max_angle(self, max_angle: int) -> None:
        assert max_angle > 0, "Max angle must be > 0. Found: {}".format(max_angle)
        self.max_angle = max_angle

    def generate_control_nodes(
        self,
        starting_pos: Tuple[float, float, float, float],
        angles: List[int],
        seg_lengths: Union[List[int], None],
    ) -> List[Tuple[float]]:
        
        # print("[CUSTOM ROAD GENERATOR] angles", angles)
        # print("[CUSTOM ROAD GENERATOR] seg_lengths", seg_lengths)

        # print(f"[CustomRoadGenerator] input for road generator: {angles}")
        # if not seg_lengths is None:
        #     assert len(angles) == len(
        #         seg_lengths
        #     ), f"Angles {angles} and lengths {seg_lengths} must have the same length"
        # assert (
        #     len(angles) == self.num_control_nodes
        # ), f"We need {self.num_control_nodes} angles {angles}"

        # set the initial node
        self.initial_node = starting_pos
        nodes = [self._get_initial_control_node(), self.initial_node]
        # i_valid is the number of valid generated control nodes.
        i_valid = 0

        while i_valid < self.num_control_nodes - 1:
            seg_length = self.seg_length
            if seg_lengths is not None and i_valid < len(seg_lengths):
                seg_length = seg_lengths[i_valid] if i_valid < self.num_control_nodes - 2 \
                             else seg_lengths[i_valid -1]
            
            # print("in road generation: i_valid", i_valid)
            # print("in road generation: angles", angles)
            # print("in road generation: seg_lengths", seg_lengths)

            nodes.append(
                self._get_next_node(
                    nodes[-2],
                    nodes[-1],
                    angles[i_valid] if i_valid < self.num_control_nodes - 2 else \
                        angles[i_valid -1],
                    self._get_next_max_angle(i_valid),
                    seg_length,
                )
            )
            # print(
            #     f"Road Instance {i_valid}, angle: {angles[i_valid]}, {seg_length}: {nodes}"
            # )
            i_valid += 1
        return nodes

    def is_valid(self, control_nodes, sample_nodes):
        return RoadPolygon.from_nodes(
            sample_nodes
        ).is_valid() and self.road_bbox.contains(
            RoadPolygon.from_nodes(control_nodes[1:-1])
        )

    def generate(self, *args, **kwargs) -> str:
        """
        Needs a list of integer angles in the kwargs param `angles`.
        Optionally takes another list of segment lengths in `seg_lengths` key of kwargs.
        """

        # if self.road_to_generate is not None:
        #     road_to_generate = copy.deepcopy(self.road_to_generate)
        #     self.road_to_generate = None
        #     return road_to_generate

        sample_nodes = None

        seg_lengths = None
        if "seg_lengths" in kwargs:
            seg_lengths = kwargs["seg_lengths"]

        control_nodes = self.generate_control_nodes(
            starting_pos=kwargs["starting_pos"],
            angles=kwargs["angles"],
            seg_lengths=seg_lengths,
        )
        control_nodes = control_nodes[0:]
        sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)

        road_points = [Point(node[0], node[1], node[2]) for node in sample_nodes]
        control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]
        _, _, _, width = self.initial_node

        self.previous_road = road_utils.get_road(road_width=width, 
                                                 road_points=road_points, 
                                                 control_points=control_points, 
                                                 simulator_name=kwargs["simulator_name"])
        
        # print(f"[CustomRoadGenerator] generated road: {self.previous_road.get_string_repr()}")
        return self.previous_road

    def _get_initial_point(self) -> Point:
        return Point(self.initial_node[0], self.initial_node[1])

    def _get_initial_control_node(self) -> Tuple[float, float, float, float]:
        x0, y0, z, width = self.initial_node
        x, y = self._get_next_xy(x0, y0, 270, self.seg_length)

        return x, y, z, width

    def _get_next_node(
        self,
        first_node,
        second_node: Tuple[float, float, float, float],
        angle: int,
        max_angle,
        seg_length: Union[float, None] = None,
    ) -> Tuple[float, float, float, float]:
        # v = np.subtract(second_node, first_node)
        # start_angle = int(np.degrees(np.arctan2(v[1], v[0])))
        # if angle > start_angle + max_angle or angle < start_angle - max_angle:
        #     print(
        #         f"{5 * '+'} Warning {angle} is not in range of {start_angle - max_angle} and {start_angle + max_angle}. Selecting random angle now {5 * '+'}"
        #     )
        #     angle = randint(start_angle - max_angle, start_angle + max_angle)
        x0, y0, z0, width0 = second_node
        if seg_length is None:
            seg_length = self.seg_length
        x1, y1 = self._get_next_xy(x0, y0, angle, seg_length)
        return x1, y1, z0, width0

    def _get_next_xy(
        self, x0: float, y0: float, angle: float, seg_length: int
    ) -> Tuple[float, float]:
        angle_rad = math.radians(angle)
        return x0 + seg_length * math.cos(angle_rad), y0 + seg_length * math.sin(
            angle_rad
        )

    def _get_next_max_angle(
        self, i: int, threshold=NUM_INITIAL_SEGMENTS_THRESHOLD,
    ) -> float:
        if i < threshold or i == self.num_control_nodes - 1:
            return 0
        else:
            # return self.max_angle
            return self.max_angles[i]



if __name__ == "__main__":
    import random

    map_size = 250

    # set_random_seed(seed=0)

    angle_complex = [30,
                    88.279769,
                    0.997705,
                    85.822416, 
                    80,
                    -80
                    ]
    angle_easy = [42.188653,11.720887,28.704612,55.379076]
    angle_custom1 = [0,90,0,0]
    angle_custom2 = [0,90,0,90]
    angle_custom3 = [0,0,0,0]

    angles_roads = [
        angle_complex#, 
        # angle_easy, 
        # angle_custom1,
        # angle_custom2,
        # angle_custom3
    ]

    # angles = [60,-10,0,90]
    for angles in angles_roads:
        seg_lengths = [ random.randint(40,40) for _ in angles]
    
        gen = CustomRoadGenerator(map_size=250,
                                            num_control_nodes=len(angles) + 1,
                                            seg_length=config.SEG_LENGTH,
                                            max_angle=config.MAX_ANGLE)

        start_time = time.perf_counter()

        road = gen.generate(simulator_name=config.UDACITY_SIM_NAME,
                            angles = angles,
                            starting_pos=(0,0,0,0),
                            seg_lengths=seg_lengths)

        concrete_representation = road.get_concrete_representation()
        print(time.perf_counter() - start_time)

        print(f"curvature: {road.compute_curvature()}")
        print(f"num turns: {road.compute_num_turns()}")

        road_test_visualizer = RoadTestVisualizer(map_size=map_size)
        road_test_visualizer.visualize_road_test(road=road, folder_path="./road_generator/", filename=f"road_custom_{str(angles)}")

        angles_computed = []
        print([p.x for p in road.control_points])
        print([p.y for p in road.control_points])
        print([p.z for p in road.control_points])

        for i in range(1,len(angles)):
            angles_computed.append(road.get_angle_points(first_p=road.control_points[i+1],
                                        second_p=road.control_points[i])
                                        )
        print(f"computed angles: {angles_computed}")