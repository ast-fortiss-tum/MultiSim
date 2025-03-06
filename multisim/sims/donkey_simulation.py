# related to open_sbt
from dataclasses import dataclass
from recording.write_image import write_image
from sims.donkey.donkey_gym_env import DonkeyGymEnv
from sims.donkey.scenes.simulator_scenes import GeneratedTrack
from opensbt.problem.adas_problem import ADASProblem
from opensbt.evaluation.fitness import *
from opensbt.evaluation.critical import *
from road_generator.file_road_generator import FileRoadGenerator
from road_generator.road_generator import RoadGenerator
from road_generator.roads.road import Road
from opensbt.simulation.simulator import Simulator, SimulationOutput
from opensbt.model_ga.individual import Individual
from opensbt.algorithm.nsga2_sim import NSGAII_SIM
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
import os
import argparse
from typing import List, Dict, Any, Tuple, Union
import traceback

from road_generator.custom_road_generator import CustomRoadGenerator

from numpy import ndarray, dtype, uint8
import numpy as np
from self_driving.supervised_agent import SupervisedAgent

import time
import config
from self_driving.agent_utils import calc_yaw_ego
import logging as log

@dataclass
class Scenario:
    """
    Models a scenario in terms of a road
    """
    road: Union[Road, None]

@dataclass
class ScenarioOutcome:
    """
    Models the outcome of a scenario
    """
    frames: List[int]
    pos: List[Tuple[float, float, float]]
    xte: List[float]
    speeds: List[float]
    actions: List[List[float]]
    scenario: Union[Scenario, None]
    isSuccess: bool
    duration: float
    fps_rate: float
  
class DonkeySimulator(Simulator):
    scenario_counter = 0
    
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = True
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        road_generator = CustomRoadGenerator(map_size=250,
                                            num_control_nodes=config.NUM_CONTROL_NODES,
                                            seg_length=config.SEG_LENGTH)

        import logging as log
        log.info("Loading existing road.")

        #file_path = "./road_generator/predefined/roads_matteo_maxibon.txt"
        # file_path = "./road_generator/predefined/roads_test.json"

        # road_generator = FileRoadGenerator(map_size=250,
        #                 file_path=file_path)
        env = None
        
        agent = SupervisedAgent(
                        #senv=env,
                        env_name=config.UDACITY_SIM_NAME,
                        model_path=config.DNN_MODEL_PATH,
                        min_speed=config.MIN_SPEED,
                        max_speed=config.MAX_SPEED_DONKEY,
                        input_shape=config.INPUT_SHAPE,
                        predict_throttle=False,
                        fake_images=False
                        )
                
        # create all scenarios
        scenarios: List[Scenario] = [
            DonkeySimulator.individualToScenario(
                starting_pos=road_generator.initial_node,
                individual=ind,
                variable_names=variable_names,
                road_generator=road_generator,
            )
            for ind in list_individuals
        ]

        # run the individualts
        outcomes = []
        steerings_all = []
        throttles_all = []
        import logging as log

        MAX_REPEAT = 10
        TIME_WAIT = 3
        
        # iterate over all scenarios
        for j, scenario in enumerate(scenarios):
            do_repeat = True
            repeat_counter = 0
            DonkeySimulator.scenario_counter += 1
            log.info(f"[DonkeySimulator] Simulating {DonkeySimulator.scenario_counter}. scenario.")
            while do_repeat and repeat_counter <= MAX_REPEAT:
                try:
                    if env is None:
                        log.info("Env is None. Creating env.")
                        env = DonkeyGymEnv(
                                seed=1,
                                add_to_port=-1,
                                test_generator=road_generator,
                                simulator_scene=GeneratedTrack(),
                                headless= False,
                                exe_path=config.DONKEY_EXE_PATH)
                                    
                    outcome = DonkeySimulator.simulate_scenario(env, agent, scenario=scenario)
                    outcomes.append(outcome)
                    do_repeat = False
                    
                    steerings_all.append([s[0][0] for s in outcome.actions])
                    throttles_all.append([s[0][1] for s in outcome.actions])
                except Exception as e:
                    log.info("[DonkeySimulator] Exception during simulation ocurred: ")
                    traceback.print_exc()
                    time.sleep(TIME_WAIT)                
                    log.error(f"\n---- Repeating run for {repeat_counter}.time due to exception: ---- \n {e} \n")
                    repeat_counter += 1
                finally:
                    try:
                        if env is not None:
                            env.close()
                            env = None
                        # kill_donkey_simulator()
                    except Exception as e:
                        print(e)

        log.info(f"[DonkeySimulator] DonkeyGymEnv close...")
        # convert the outcomes to sbt format
        simouts = []
        for i,scenario in enumerate(scenarios):
            outcome = outcomes[i]
            control_points = scenario.road.control_points
            control_points_serializable =[(point.x, point.y) for point in control_points[1:]] if control_points else None

            simouts.append(
                SimulationOutput(
                simTime=outcome.duration,
                times=outcome.frames,
                location={"ego": [(x[0], x[1]) for x in outcome.pos]},
                velocity={
                    "ego": DonkeySimulator._calculate_velocities(
                        outcome.pos, outcome.speeds
                    )
                },
                speed={"ego": outcome.speeds},
                acceleration={"ego": DonkeySimulator.calc_acceleration(outcome.speeds, fps = 20)},
                yaw={
                        "ego" : calc_yaw_ego(outcome.pos)
                    },
                collisions=[],
                actors={
                    "ego" : "ego",
                    "vehicles" : ["ego"],
                    "pedestrians" : []
                },
                otherParams={"xte": outcome.xte,
                              "simulator": "Donkey",
                              "road":  scenario.
                                        road.
                                        get_concrete_representation(to_plot=True),
                                "road_control_points": control_points_serializable,
                               "steerings" : steerings_all[i],
                               "throttles" : throttles_all[i],
                               "fps_rate": outcome.fps_rate                          
                              }
            )
            )

        # close if simulator is automatically restarted
        if env is not None and env.exe_path is not None:
            env.close()
        return simouts


    @staticmethod
    def individualToScenario(
        individual: Individual,
        variable_names: List[str],
        road_generator: RoadGenerator,
        starting_pos: Tuple[int]
    ) -> Scenario:
        instance_values = [v for v in zip(variable_names, individual)]
        angles: List[str] = []
        seg_lengths: List[str] = []

        for i in range(0, len(instance_values)):
            if instance_values[i][0].startswith("angle"):
                new_angle = int(instance_values[i][1])
                angles.append(new_angle)
            elif instance_values[i][0].startswith("seg_length"):
                seg_length = int(instance_values[i][1])
                seg_lengths.append(seg_length)

        # generate the road string from the configuration
        seg_lengths  = seg_lengths if len(seg_lengths) > 0 else None
        road = road_generator.generate(starting_pos=starting_pos,
                                            angles=angles, 
                                           seg_lengths=seg_lengths,
                                           simulator_name=config.DONKEY_SIM_NAME)
        # road = road_generator.generate()
        
        return Scenario(
            road=road
        )

    @staticmethod
    def _calculate_velocities(
        positions: List[Tuple[float, float, float]], 
        speeds: List[float]
    ) -> Tuple[float, float, float]:
        """
        Calculate velocities given a list of positions and corresponding speeds.
        """
        velocities = []

        for i in range(len(positions) - 1):
            displacement = np.array(positions[i + 1]) - np.array(positions[i])
            direction = displacement / np.linalg.norm(displacement)
            velocity = direction * speeds[i]
            velocities.append(velocity)
        return velocities
   
    @staticmethod
    def calc_acceleration(speeds: List, fps: int):
        acc=[0]
        for i in range(1,len(speeds)):
            a = (speeds[i] - speeds[i-1])*fps / 3.6 # convert to m/s
            acc.append(a)
        return acc

    @staticmethod
    def simulate_scenario(env, agent: SupervisedAgent, scenario: Scenario
        ) -> ScenarioOutcome:
            road = scenario.road
    
            # set all params for init loop
            actions = [[0.0, 0.0]]

            # set up params for saving data
            pos_list = []
            xte_list = []
            actions_list = []
            speed_list = []
            isSuccess = False
            state = {}
            done = False

            # reset the scene to match the scenario
            obs = env.reset(skip_generation=False, road=road)

            time.sleep(1)

            print("Scenario constructed.")

            start = time.time()

            fps_time_start = time.time()
            counter = 0
            counter_all = []

            img_time_start = time.time()

            fps_desired_timer = start
            # run the scenario
            while not done:
                try:
                    record_image = False
                    if record_image and time.time() - img_time_start > 0.5:
                        img_time_start = time.time()        
                        # write image
                        write_image(DonkeySimulator.scenario_counter, 
                            image=obs,
                            simname="donkey")

                    # process only images with specific frequency
                    if(time.time() - fps_desired_timer) < 1/config.FPS_DESIRED_DONKEY:
                        continue
                    else:
                        fps_desired_timer = time.time()

                    # calculate fps
                    if time.time() - fps_time_start > 1:
                        #reset 
                        # log.info(f"Frames in 1s: {counter}")
                        # log.info(f"Time passed: {time.time() - fps_time_start}")
                        
                        counter_all.append(counter)
                        counter = 0
                        fps_time_start = time.time()
                    else:
                        counter += 1

                    # TODO: Play around with this value
                    state["simulator_name"] = config.DONKEY_SIM_NAME
                    actions = agent.predict(obs,
                             state = state)

                    obs, done, info = env.step(actions)
                
                    state["xte"] = info.get("cte", None)
                    state["xte_pid"] = info.get("xte_pid", None)
                    state["speed"] = info.get("speed", None)
                    state["image"] = obs

                    pos = info.get("pos", None)

                    # print(f"Position is: {pos}")
                    # print(f"XTE is: {state['xte']}")
                    # print(f"Speed is: {state['speed']}")

                    # throw exception if last 10 XTE values were 0; BUG with reseting road
                    if (len(xte_list) > 9 and (np.asarray(xte_list[0:10]) == 0).all()):
                        raise Exception("Last 10 XTE values are zero. Exception rised.")
                    
                    # save data for output
                    pos_list.append([
                                     pos[0], 
                                     pos[1], 
                                     0
                                    ])
                    if config.CAP_XTE:
                        xte_list.append(state["xte"] 
                                        if abs(state["xte"]) <= config.MAX_XTE \
                                        else config.MAX_XTE)
                        assert np.all(abs(np.asarray(xte_list)) <= config.MAX_XTE), f"At least one element is not smaller than {config.MAX_XTE}"
                    else:
                        xte_list.append(state["xte"])
                        
                    speed_list.append(state["speed"])
                    actions_list.append(actions)

                    end = time.time()

                    time_elapsed = int(end - start)

                    if time_elapsed > config.TIME_LIMIT:  
                        print(f"Over time limit, terminating.")    
                        done = True       

                    # check fps
                    env.viewer.handler.observation_timer.on_frame()
                        
                    if done:
                        isSuccess = True
                        break
                    elif abs(state["xte"]) > config.MAX_XTE:
                        print(f"Is above MAXIMAL_XTE (={config.MAX_XTE}), terminating.")
                        done = True
                        isSuccess = False
                        break

                except KeyboardInterrupt:
                    print(f"{5 * '+'} SDSandBox Simulator Got Interrupted {5 * '+'}")
                    # self.client.stop()
                    raise KeyboardInterrupt
            end = time.time()
           
            fps_rate = np.sum(counter_all)/time_elapsed
            log.info(f"FPS rate: {fps_rate}")

            duration = end - start

            return ScenarioOutcome(
                frames=[x for x in range(len(pos_list))],
                pos=pos_list,
                xte=xte_list,
                speeds=speed_list,
                actions=actions_list,
                scenario=scenario,
                isSuccess=isSuccess,
                duration = duration,
                fps_rate=fps_rate
            )
    
if __name__ == "__main__":
    road = [-90, -110, 175, -120, -135] + [10, 20, 17, 15,15]
    variable_names = [f"angle{i}" for i in range(5)] + [f"seg_length{i}" for i in range(5)] 
    DonkeySimulator.simulate(list_individuals= [road],
                                   variable_names=variable_names,
                                   do_visualize=True,
                                   scenario_path="",
                                   sim_time=10,
                                   time_step=10)