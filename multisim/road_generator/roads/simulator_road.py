from typing import List, Tuple, Union
from shapely.geometry import Point
from road_generator.roads.road import Road


class SimulatorRoad(Road):
    def __init__(
        self, road_width: int, road_points: List[Point], control_points: List[Point]
    ):
        super().__init__(
            road_width=road_width,
            road_points=road_points,
            control_points=control_points,
        )

    def get_control_points(self) -> List[Tuple[float, float]]:
        return [(point.x, point.y) for point in self.control_points]

    def get_concrete_representation(
        self, to_plot: bool = False
    ) -> List[Tuple[float, float, float, float]]:
        if to_plot:
            return [
                (point.x, point.y, 1.90000000, self.road_width)
                for point in self.road_points
            ]
        return [
            (point.x, point.z, point.y, self.road_width)
            for point in self.road_points
        ]

    def get_inverse_concrete_representation(
        self, to_plot: bool = False
    ) -> List[Tuple[float, float, float, float]]:
        if to_plot:
            return [
                (point.x, point.y, 1.90000000, self.road_width)
                for point in reversed(self.road_points)
            ]
        return [
            (point.x, 1.90000000, point.y, self.road_width)
            for point in reversed(self.road_points)
        ]

    def serialize_concrete_representation(self, cr: List[Tuple[float]]) -> str:
        # for Udacity we only need x, y, z
        to_string = []
        for t in cr:
            s = "{},{},{}".format(t[0], t[1], t[2])
            to_string.append(s)
        return "@".join(to_string)
      # self.previous_road.serialize_concrete_representation(
        #     cr=self.previous_road.get_concrete_representation()
        # )

            
    def get_string_repr(road: Road):
        return road.serialize_concrete_representation(road.get_concrete_representation())
