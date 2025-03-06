# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import time
from typing import List

import numpy as np
from shapely.geometry import Point

# from self_driving.utils.visualization import RoadTestVisualizer
from config import DONKEY_SIM_NAME, ROAD_WIDTH
from self_driving.global_log import GlobalLog
from self_driving.road_utils import get_road
from road_generator.road_generator import RoadGenerator
from self_driving.utils.randomness import set_random_seed


class SinRoadGenerator(RoadGenerator):
    def __init__(self, 
        simulator_name: str,
        map_size: int,
        initial_node=(0.0, 0.0, 0.0, 0.0),
        bbox_size=(0, 0, 250, 250)):
    

        assert len(initial_node) == 4 and len(bbox_size) == 4
        self.map_size = map_size
        self.initial_node = initial_node
        self.road_to_generate = None
        self.bbox_size = bbox_size
        self.simulator_name = simulator_name
        self.logg = GlobalLog("SinTestGenerator")

    def generate(self, amplitude = 1, frequency = 1/10, **kwargs):
        road_points: List[Point] = [Point(x, np.sin(x * frequency) * amplitude) 
                                      for x in np.arange(0.0, 200.0, 2.0)]

        # for p in road_points:
        #     p.x = p.x + self.initial_node[0]
        #     p.y = p.y + self.initial_node[2]
        #     p.z = p.z + self.initial_node[1]

        return get_road(
            road_points=road_points,
            control_points=[Point(p.x, p.y, 0.0) for p in road_points],
            road_width=ROAD_WIDTH,
            simulator_name=self.simulator_name,
        )

    def set_max_angle(self, max_angle: int) -> None:
        raise NotImplemented("Not implemented")


# if __name__ == "__main__":

#     map_size = 250

#     set_random_seed(seed=0)

#     roadgen = SinTestGenerator(simulator_name=DONKEY_SIM_NAME)
#     start_time = time.perf_counter()
#     road = roadgen.generate()
#     concrete_representation = road.get_concrete_representation()
#     print(time.perf_counter() - start_time)

#     road_test_visualizer = RoadTestVisualizer(map_size=map_size)
#     road_test_visualizer.visualize_road_test(road=road, folder_path="../", filename="road_2")
