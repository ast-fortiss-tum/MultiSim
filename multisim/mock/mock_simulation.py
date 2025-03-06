
import random
from typing import List
import config
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.simulation.simulator import SimulationOutput, Simulator
from road_generator.custom_road_generator import CustomRoadGenerator
from sims.udacity.udacity_simulator import UdacitySimulator
import logging as log
from road_generator.roads.road import Road
from self_driving.agent_utils import calc_yaw_ego

class MockSimulator(Simulator):
    
    @staticmethod
    def simulate(
        list_individuals: List[IndividualSimulated],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        results = []
        test_generator = CustomRoadGenerator(map_size=250,
                                            num_control_nodes=config.NUM_CONTROL_NODES,
                                            seg_length=config.SEG_LENGTH) 
    
        for ind in list_individuals:
            speed = 0
            try:
                def get_rand_bounds(a,b):
                    return a + (b-a) * random.random()
                instance_values = [v for v in zip(variable_names, ind)]

                # TODO improve how we distinguish whether segment lenghts are part or not of the representation
                if len(ind) > test_generator.num_control_nodes:
                    angles = [ int(instance_values[i][1]) for i in range(0, int(len(instance_values)/2))]
                    seg_lengths = [ int(instance_values[i][1]) for i in range(int(len(instance_values)/2), len(instance_values))]
                else:
                    angles = [ int(instance_values[i][1]) for i in range(0, int(len(instance_values)))]
                    seg_lengths = None

                road: Road = test_generator.generate(
                            starting_pos=UdacitySimulator.initial_pos,
                            angles=angles,
                            simulator_name=config.UDACITY_SIM_NAME,
                            seg_lengths=seg_lengths
                            )
                # road = test_generator.generate()
                waypoints = road.get_string_repr()
                waypoints_raw = road.get_concrete_representation()

                # we just create a trajectory close to the waypoints
                N_STEPS = len(waypoints_raw)
                speeds = [20 for i in range(0,N_STEPS)]
                pos = [(waypoints_raw[i][0] - 1,waypoints_raw[i][2] - 1,waypoints_raw[i][1]) for i in range(0,N_STEPS)]
                
                random_xte = get_rand_bounds(-3,-1) 
                xte = [random_xte for i in range(0, N_STEPS)]
                
                steerings = [get_rand_bounds(-1,+1) for i in range(0, N_STEPS)]
                throttles =  [get_rand_bounds(-1,+1) for i in range(0, N_STEPS)]
                yaw =  calc_yaw_ego(pos)
          
                fps_rate = 20
                log.info(f"FPS rate: {fps_rate}")

                # morph values into SimulationOutput Object
                result = SimulationOutput(
                    simTime=-1,
                    times=[x for x in range(len(speeds))],
                    location={
                        "ego": [(x[0], x[1]) for x in pos],  # cut out z value
                    },
                    velocity={
                        "ego": [],
                    },
                    speed={
                        "ego": speeds,
                    },
                    acceleration={"ego": 
                                  []},
                    yaw={
                        "ego" : yaw
                    },                   
                    collisions=[],
                    actors={
                        "ego" : "ego",
                        "pedestrians" : [],
                        "vehicles": ["ego"]
                    },
                    otherParams={"xte": xte,
                                "simulator" : "Udacity",
                                "road": road.get_concrete_representation(to_plot=True),
                                "steerings" : steerings,
                                "throttles" : throttles,
                                "fps_rate": fps_rate}
                )
                results.append(result)
            except Exception as e:
                raise e
        return results
