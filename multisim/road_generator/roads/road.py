import copy
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Iterator

import numpy as np
from shapely.geometry import Point

from self_driving.custom_types import Tuple4F

from road_generator.roads.bbox import RoadBoundingBox
from road_generator.roads.catmull_rom import catmull_rom
from road_generator.roads.road_polygon import RoadPolygon


class Road(ABC):
    def __init__(
        self,
        road_width: int,
        road_points: List[Point],
        control_points: List[Point],
        bbox_size=(0, 0, 250, 250),
        num_sampled_points=20,
    ):
        self.road_width = road_width
        self.road_points = road_points
        self.control_points = control_points
        self.max_angle = None
        self.num_sampled_points = num_sampled_points
        self.road_bbox = RoadBoundingBox(bbox_size)

    @abstractmethod
    def get_concrete_representation(self, to_plot: bool = False) -> List[Tuple4F]:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def get_inverse_concrete_representation(
        self, to_plot: bool = False
    ) -> List[Tuple4F]:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def serialize_concrete_representation(self, cr: List[Tuple4F]) -> str:
        raise NotImplemented("Not implemented")

    @staticmethod
    def get_road_points_from_concrete(cr: List[Tuple4F]) -> List[Point]:
        return [Point(cr_item[0], cr_item[1]) for cr_item in cr]

    @staticmethod
    def get_road_width_from_concrete(cr: List[Tuple4F]) -> int:
        assert len(cr) > 0, "Concrete representation must not be empty"
        return cr[0][-1]

    @staticmethod
    def get_radius(p1: Point, p2: Point, p3: Point) -> float:
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2.x * p2.x + p2.y * p2.y
        bc = (p1.x * p1.x + p1.y * p1.y - temp) / 2
        cd = (temp - p3.x * p3.x - p3.y * p3.y) / 2
        det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)

        if abs(det) < 1.0e-6:
            return np.inf

        # Center of circle
        cx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) / det
        cy = ((p1.x - p2.x) * cd - (p2.x - p3.x) * bc) / det

        radius = np.sqrt((cx - p1.x) ** 2 + (cy - p1.y) ** 2)

        return radius

    def compute_curvature(self, w: int = 5) -> float:
        assert len(self.road_points) > 0, "There must be road points to compute angles"
        min_radius = np.inf
        nodes = self.road_points
        # print("computing curvature:")
        for i in range(len(nodes) - w):
            p1 = nodes[i]
            p2 = nodes[i + int((w - 1) / 2)]
            p3 = nodes[i + (w - 1)]
            radius = self.get_radius(p1=p1, p2=p2, p3=p3)
            # print(f"radius single curve:", 1/radius)
            if radius < min_radius:
                min_radius = radius
                # print(f"points: {p1} {p2} {p3}")
                # print(f"radius updated to: {min_radius}")
        curvature = 1 / min_radius
        # print("end computing curvature:")
        return curvature

    # FIXME: a bit dirty, copied from deepjanus_test_generator
    def is_valid(self) -> bool:
        return RoadPolygon.from_nodes(
            self.get_concrete_representation(to_plot=True)
        ).is_valid() and self.road_bbox.contains(
            RoadPolygon.from_nodes(
                [
                    (cp.x, cp.y, cp.z, self.road_width)
                    for cp in self.control_points[1:-1]
                ]
            )
        )
    def is_too_sharp(the_test, TSHD_RADIUS=47):
        """
        TAKEN FROM AMBIEGEN
        
        If the minimum radius of the test is greater than the TSHD_RADIUS, then the test is too sharp

        Args:
        the_test: the input road topology
        TSHD_RADIUS: The radius of the circle that is used to check if the test is too sharp. Defaults to
        47

        Returns:
        the boolean value of the check variable.
        """
        if TSHD_RADIUS > Road.min_radius(the_test) > 0.0:
            check = True
        else:
            check = False
        return check
    
    def mutate_gene(
        self, index: int, lower_bound: int, upper_bound: int, xy_prob: float = 0.5
    ) -> Tuple[int, int]:
        mutated_point = copy.deepcopy(self.control_points[index])
        # Choose the mutation extent
        mut_value = random.randint(a=lower_bound, b=upper_bound)
        # Avoid choosing 0
        if mut_value == 0:
            mut_value += 1

        if random.random() < xy_prob:
            index_mutated = 0
            mutated_point = Point(
                mutated_point.x + mut_value, mutated_point.y, mutated_point.z
            )
        else:
            index_mutated = 1
            mutated_point = Point(
                mutated_point.x, mutated_point.y + mut_value, mutated_point.z
            )

        self.control_points[index] = mutated_point
        road_points = catmull_rom(
            [(cp.x, cp.y, cp.z, self.road_width) for cp in self.control_points[1:]],
            num_spline_points=self.num_sampled_points,
        )
        self.road_points = [Point(rp[0], rp[1]) for rp in road_points]
        return index_mutated, mut_value

    def undo_mutation(self, index: int, index_mutated: int, mut_value: int) -> None:
        mutated_point = copy.deepcopy(self.control_points[index])
        if index_mutated == 0:
            mutated_point = Point(
                mutated_point.x - mut_value, mutated_point.y, mutated_point.z
            )
        else:
            mutated_point = Point(
                mutated_point.x, mutated_point.y - mut_value, mutated_point.z
            )
        self.control_points[index] = mutated_point
        road_points = catmull_rom(
            [(cp.x, cp.y, cp.z, self.road_width) for cp in self.control_points[1:]],
            num_spline_points=self.num_sampled_points,
        )
        self.road_points = [Point(rp[0], rp[1], rp[2]) for rp in road_points]

    def export(self) -> Dict:
        result = dict()
        result["control_points"] = [(cp.x, cp.y, cp.z) for cp in self.control_points]
        result["road_points"] = self.get_concrete_representation(to_plot=True)
        result["road_width"] = self.road_width
        return result

    def are_control_points_different(self, other_control_points: List[Point]) -> bool:
        for i, cp in enumerate(self.control_points):
            other_cp = other_control_points[i]
            if cp.x != other_cp.x or cp.y != other_cp.y:
                return True
        return False

    @staticmethod
    def import_keys() -> List[str]:
        return ["control_points", "road_points", "road_width"]

    def __eq__(self, other: "Road") -> bool:
        if isinstance(other, Road):
            return not self.are_control_points_different(
                other_control_points=other.control_points
            )
        raise RuntimeError("other {} is not an road".format(type(other)))

    def __hash__(self) -> int:
        return hash("_".join(["@".join([cp.x, cp.y]) for cp in self.control_points]))

    @staticmethod
    def compute_angle_distance(v0: np.ndarray, v1: np.ndarray) -> float:
        at_0 = np.arctan2(v0[1], v0[0])
        at_1 = np.arctan2(v1[1], v1[0])
        return at_1 - at_0
    
    @staticmethod
    def get_angle_points(first_p: Point, second_p: Point, shift = 180):
        
        v = np.subtract(np.asarray([second_p.x, second_p.y]),
                            np.asarray([first_p.x, first_p.y]))
        angle = shift + np.ceil((np.degrees(np.arctan2(v[1], v[0])))*100)/100
        return angle
        
    def compute_angle_distance_pairs_for_each_point(
        self,
    ) -> List[Tuple[float, float, List[Point]]]:
        result = []
        v1 = np.subtract(self.road_points[1], self.road_points[0])
        for i in range(len(self.road_points) - 1):
            v0 = v1
            v1 = np.subtract(self.road_points[i + 1], self.road_points[i])
            angle = self.compute_angle_distance(v0=v0, v1=v1)
            distance = np.linalg.norm(v1)
            result.append(
                (angle, distance, [self.road_points[i + 1], self.road_points[i]])
            )
        return result

    @staticmethod
    def grouper(lst: List[str]) -> Iterator[List[str]]:
        prev = None
        group = []
        for item in lst:
            if not prev or item == prev:
                group.append(item)
            else:
                yield group
                group = [item]
            prev = item
        if len(group) > 0:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    @staticmethod
    def first_super_grouper(
        it: Iterator[List[str]],
        first_segment_threshold: int,
        second_segment_threshold: int,
    ) -> Iterator[List[str]]:
        prev = None
        group = []
        for item in it:
            if not prev:
                group.extend(item)
            elif len(item) < second_segment_threshold and item[0] == "s":
                item = [prev[-1]] * len(item)
                group.extend(item)
            elif (
                len(item) < first_segment_threshold
                and item[0] != "s"
                and prev[-1] == item[0]
            ):
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if len(group) > 0:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    @staticmethod
    def second_super_grouper(
        it: Iterator[List[str]], first_segment_threshold: int
    ) -> Iterator[List[str]]:
        prev = None
        group = []
        for item in it:
            if not prev:
                group.extend(item)
            elif len(item) < first_segment_threshold:
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if len(group) > 0:
            yield group

    def find_circle(p1, p2, p3):
        """
        TAKEN FROM AMBIEGEN

        The function takes three points and returns the radius of the circle that passes through them

        Args:
        p1: the first point
        p2: the point that is the center of the circle
        p3: the point that is the furthest away from the line

        Returns:
        The radius of the circle.
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return np.inf

        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
        # print(radius)
        return radius

    def min_radius(x, w=5):
        """
        AMBIEGEN

        It takes a list of points (x) and a window size (w) and returns the minimum radius of curvature of
        the line segment defined by the points in the window

        Args:
        x: the x,y coordinates of the points
        w: window size. Defaults to 5

        Returns:
        The minimum radius of curvature of the road.
        """
        mr = np.inf
        nodes = x
        for i in range(len(nodes) - w):
            p1 = nodes[i]
            p2 = nodes[i + int((w - 1) / 2)]
            p3 = nodes[i + (w - 1)]
            radius = Road.find_circle(p1, p2, p3)
            if radius < mr:
                mr = radius
        if mr == np.inf:
            mr = 0

        return mr * 3.280839895  # , mincurv
    
    # FIXME: not so clear what it does
    def compute_num_turns(self) -> Tuple[int, float]:
        angle_distance_pairs = self.compute_angle_distance_pairs_for_each_point()

        first_segment_threshold = 15
        second_segment_threshold = 10
        angle_threshold = 0.005

        # iterate over the nodes to get the turns bigger than the threshold
        # a turn category is assigned to each node
        # l is a left turn
        # r is a right turn
        # s is a straight segment
        # first node is always an s
        turns = []
        for i in range(len(angle_distance_pairs)):
            angle = (angle_distance_pairs[i][0] + 180) % 360 - 180
            if np.abs(angle) > angle_threshold:
                if angle > 0:
                    turns.append("l")
                else:
                    turns.append("r")
            else:
                turns.append("s")

        groups = self.grouper(lst=turns)
        first_super_group = self.first_super_grouper(
            it=groups,
            first_segment_threshold=first_segment_threshold,
            second_segment_threshold=second_segment_threshold,
        )
        second_super_group = self.second_super_grouper(
            it=first_super_group, first_segment_threshold=first_segment_threshold
        )

        num_turns = 0
        w = 5
        sum_lengths = 0
        weighted_num_turns = 0
        for g in second_super_group:
            min_radius = np.inf
            for i in range(sum_lengths, len(g) + sum_lengths - w):
                p1 = self.road_points[i]
                p2 = self.road_points[i + int((w - 1) / 2)]
                p3 = self.road_points[i + (w - 1)]
                radius = self.get_radius(p1=p1, p2=p2, p3=p3)
                min_radius = min(min_radius, radius)

            sum_lengths += len(g)

            if g[-1] != "s":
                num_turns += 1
                weighted_num_turns += 1 / min_radius

        return num_turns, weighted_num_turns
