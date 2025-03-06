# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import copy
import math
import time
from random import randint
from typing import List

import numpy as np
from shapely.geometry import Point

from self_driving.utils.visualization import RoadTestVisualizer
from config import (
    AGENT_TYPES,
    BEAMNG_SIM_NAME,
    DONKEY_SIM_NAME,
    MAX_ANGLE,
    NUM_CONTROL_NODES,
    NUM_SAMPLED_POINTS,
    ROAD_WIDTH,
    SEG_LENGTH,
    UDACITY_SIM_NAME,
)
from self_driving.custom_types import Tuple2F, Tuple4F
from self_driving.global_log import GlobalLog
from self_driving.bbox import RoadBoundingBox
from self_driving.catmull_rom import catmull_rom
from self_driving.road import Road
from self_driving.road_polygon import RoadPolygon
from self_driving.road_utils import get_road
from road_generator.road_generator import RoadGenerator
from self_driving.utils.randomness import set_random_seed

# def angle_between_vectors(p1: Point, p2: Point):
#     v1 = np.asarray([p1.x, p1.y])
#     v2 = np.asarray([p2.x, p2.y])

#     dot_prod = sum((a*b) for a, b in zip(v1, v2))
#     mag_v1 = math.sqrt(sum(a**2 for a in v1))
#     mag_v2 = math.sqrt(sum(a**2 for a in v2))
#     if mag_v1 == 0 or mag_v2 == 0:
#         raise ValueError("One of the vectors has zero magnitude.")
#     return np.degrees(np.arcsin(abs(dot_prod) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

class JanusTestGenerator(object):
    """Generate random roads given the configuration parameters"""

    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20

    def __init__(
        self,
        map_size: int,
        simulator_name: str,
        agent_type: str = "random",
        num_control_nodes=NUM_CONTROL_NODES,
        max_angle=MAX_ANGLE,
        seg_length=SEG_LENGTH,
        num_spline_nodes=NUM_SAMPLED_POINTS,
        initial_node=(125.0, 0.0, -28.0, ROAD_WIDTH),  # z = -28.0 (BeamNG), width = 8.0 (BeamNG)
        bbox_size=(0, 0, 250, 250),
    ):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4 and len(bbox_size) == 4
        assert agent_type in AGENT_TYPES, "Unknown agent type: {}".format(agent_type)

        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.agent_type = agent_type
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.road_bbox = RoadBoundingBox(bbox_size)
        self.simulator_name = simulator_name

        self.previous_road: Road = None
        self.logg = GlobalLog("JanusTestGenerator")
        self.road_to_generate = None
        assert not self.road_bbox.intersects_vertices(self._get_initial_point())

    def set_max_angle(self, max_angle: int) -> None:
        assert max_angle > 0, "Max angle must be > 0. Found: {}".format(max_angle)
        self.max_angle = max_angle

    def generate_control_nodes(self, attempts=NUM_UNDO_ATTEMPTS) -> List[Tuple4F]:
        condition = True
        while condition:
            nodes = [self._get_initial_control_node(), self.initial_node]
            # i_valid is the number of valid generated control nodes.
            i_valid = 0
            # When attempt >= attempts and the skeleton of the road is still invalid,
            # the construction of the skeleton starts again from the beginning.
            # attempt is incremented every time the skeleton is invalid.
            attempt = 0

            while i_valid < self.num_control_nodes and attempt <= attempts:
                nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(i_valid)))
                road_polygon = RoadPolygon.from_nodes(nodes)

                # budget is the number of iterations used to attempt to add a valid next control node
                # before also removing the previous control node.
                budget = self.num_control_nodes - i_valid
                assert budget >= 1

                intersect_boundary = self.road_bbox.intersects_boundary(road_polygon.polygons[-1])
                is_valid = road_polygon.is_valid() and (
                    ((i_valid == 0) and intersect_boundary) or ((i_valid > 0) and not intersect_boundary)
                )
                while not is_valid and budget > 0:
                    nodes.pop()
                    budget -= 1
                    attempt += 1

                    nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(i_valid)))
                    road_polygon = RoadPolygon.from_nodes(nodes)

                    intersect_boundary = self.road_bbox.intersects_boundary(road_polygon.polygons[-1])
                    is_valid = road_polygon.is_valid() and (
                        ((i_valid == 0) and intersect_boundary) or ((i_valid > 0) and not intersect_boundary)
                    )

                if is_valid:
                    i_valid += 1
                else:
                    assert budget == 0
                    nodes.pop()
                    if len(nodes) > 2:
                        nodes.pop()
                        i_valid -= 1

                assert RoadPolygon.from_nodes(nodes).is_valid()
                assert 0 <= i_valid <= self.num_control_nodes

            # The road generation ends when there are the control nodes plus the two extra nodes needed by
            # the current Catmull-Rom model
            if len(nodes) - 2 == self.num_control_nodes:
                condition = False

        return nodes

    def is_valid(self, control_nodes, sample_nodes):
        return RoadPolygon.from_nodes(sample_nodes).is_valid() and self.road_bbox.contains(
            RoadPolygon.from_nodes(control_nodes[1:-1])
        )

    def generate(self) -> Road:

        if self.road_to_generate is not None:
            road_to_generate = copy.deepcopy(self.road_to_generate)
            self.road_to_generate = None
            return road_to_generate

        if self.agent_type == "autopilot":
            # When collecting the training set invert the road such that the left and right curves are balanced
            if self.previous_road is not None:
                self.logg.debug("Autopilot mode. Inverting previous road to get a more balanced training set")
                road_to_return = get_road(
                    road_points=list(reversed(self.previous_road.road_points)),
                    control_points=list(reversed(self.previous_road.control_points)),
                    road_width=ROAD_WIDTH,
                    simulator_name=self.simulator_name,
                )
                self.previous_road = None
                return road_to_return

        sample_nodes = None
        condition = True
        while condition:
            control_nodes = self.generate_control_nodes()
            control_nodes = control_nodes[1:]
            sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
            if self.is_valid(control_nodes, sample_nodes):
                condition = False

        road_points = [Point(node[0], node[1]) for node in sample_nodes]
        control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]

        self.previous_road = get_road(
            road_points=road_points, control_points=control_points, road_width=ROAD_WIDTH, simulator_name=self.simulator_name
        )

        return self.previous_road

    def _get_initial_point(self) -> Point:
        return Point(self.initial_node[0], self.initial_node[1])

    def _get_initial_control_node(self) -> Tuple4F:
        x0, y0, z, width = self.initial_node
        x, y = self._get_next_xy(x0, y0, 270)
        assert not (self.road_bbox.bbox.contains(Point(x, y)))

        return x, y, z, width

    def _get_next_node(self, first_node, second_node: Tuple4F, max_angle) -> Tuple4F:
        v = np.subtract(second_node, first_node)
        start_angle = int(np.degrees(np.arctan2(v[1], v[0])))
        # print("+++++++++++++++++++++++")
        # print(f"start_angle is: {start_angle}")
        angle = randint(start_angle - max_angle, start_angle + max_angle)
        # print(f"selected angle is: {angle}")
        # print(f"angle difference: {angle - start_angle}")

        x0, y0, z0, width0 = second_node
        x1, y1 = self._get_next_xy(x0, y0, angle)
        return x1, y1, z0, width0

    def _get_next_xy(self, x0: float, y0: float, angle: float) -> Tuple2F:
        angle_rad = math.radians(angle)
        return x0 + self.seg_length * math.cos(angle_rad), y0 + self.seg_length * math.sin(angle_rad)

    def _get_next_max_angle(self, i: int, threshold=NUM_INITIAL_SEGMENTS_THRESHOLD) -> float:
        if i < threshold or i == self.num_control_nodes - 1:
            return 0
        else:
            return self.max_angle

if __name__ == "__main__":
    import shapely
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
    
    map_size = 250

    sim_name = DONKEY_SIM_NAME

    # set_random_seed(seed=0)

    for i in range(5):

        print("************ {} ************".format(i))

        roadgen = JanusTestGenerator(map_size=map_size, simulator_name=sim_name, num_control_nodes=NUM_CONTROL_NODES)

        start_time = time.perf_counter()
        road = roadgen.generate()
        concrete_representation = road.get_concrete_representation()
        # print([(cp.x, cp.y, cp.z) for cp in road.control_points])
        print("Time:", time.perf_counter() - start_time)
        print("Curvature:", road.compute_curvature())
        print("Num turns:", road.compute_num_turns())

        road_test_visualizer = RoadTestVisualizer(map_size=map_size)
        road_test_visualizer.visualize_road_test(
            road=road, 
            folder_path="./operators", 
            filename="road_{}".format(i), 
            plot_control_points=True
        )

        print(f"n control points: {len(road.control_points)}")

        # get angle representation
        angles = []
        for j in range(2,NUM_CONTROL_NODES):
            angles.append(road.get_angle_points(first_p=road.control_points[j-1],
                                        second_p=road.control_points[j],
                                        shift=0)
                                        )
        print(f"angles: {angles}")

        # angle_distance = road.compute_angle_distance_pairs_for_each_point()
        
        # print(f"angle distance: {[d for d,_,_ in angle_distance]}")
        # get angles vectors
    
        for i in range(1, len(angles)):
            a1 = angles[i-1]
            a2 = angles[i]
            from operators.utils.validate_angle import max_angle_preserved
            preserved, dif =  max_angle_preserved(a1,a2,MAX_ANGLE)
            assert preserved, f"dif is {dif}"
            #assert abs(angles[i] - angles[i-1]) <= MAX_ANGLE, f"Angle difference violated for {angles[i]} and {angles[i-1]}"