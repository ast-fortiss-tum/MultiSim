# related to open_sbt
from dataclasses import dataclass
from logging import log
import shutil
from recording.write_image import write_image
from self_driving.agent import Agent
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

from sims.beamng.beamng_gym_env import BeamngGymEnv
import logging as log

import time
import config
from self_driving.agent_utils import calc_yaw_ego
from sims.os_utils import kill_beamng_simulator

def clean_user_folder(directory=config.BEAMNG_USER):
    # List all files in the directory
    try:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if not f.endswith("tech.key"):
                    os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    except Exception as e:
        print(f"An error occurred: {e}")
    
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
    fpsRate: float
    simTime: float = -1



CENTER_OFFSET = 1.54
MAX_REPEAT = 5
TIME_WAIT = 5

class BeamngSimulator(Simulator):
    scenario_counter = 0
    kill_thread_created = False

    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        # run beamng periodic killer if not yet done
        # if config.KILL_BMNG_PERIODICALLY:
        #     file = os.path.join(".", 'sims', 'beamng', 'kill_bmng.bat')

        #     killer = BeamngPeriodicKiller(file = file,
        #                             interval_seconds=30,
        #                             max_time = 60)
        #     killer.start()
        if config.KILL_BMNG_PERIODICALLY and not BeamngSimulator.kill_thread_created:
            import os
            from sims.beamng.kill_utils import kill_periodically_beamng
            from config import BEAMNG_KILL_INTERVAL, BEAMNG_KILL_MAX

            file = os.path.join(".", 'sims', 'beamng', 'kill_bmng.bat')
            kill_periodically_beamng(interval_seconds = BEAMNG_KILL_INTERVAL,
                                        file = file,
                                        max_time = BEAMNG_KILL_MAX)
            BeamngSimulator.kill_thread_created = True
            
        road_generator = CustomRoadGenerator(map_size = config.MAP_SIZE,
                                            num_control_nodes=config.NUM_CONTROL_NODES,
                                                seg_length=config.SEG_LENGTH)
    
        # file_path = os.getcwd() + os.sep + "road_generator/roads_matteo_maxibon.txt"
        # road_generator = FileRoadGenerator(map_size=250,
        #                 file_path=file_path,
        #                 current_index=1)
        
        # file_path = "./road_generator/predefined/roads_matteo_maxibon.txt"
        #file_path = "./road_generator/predefined/roads_test.json"

        # road_generator = FileRoadGenerator(map_size=250,
        #                 file_path=file_path)

        agent = SupervisedAgent(
                        #env=env,
                        env_name=config.BEAMNG_SIM_NAME,
                        model_path=config.DNN_MODEL_PATH,
                        min_speed=config.MIN_SPEED,
                        max_speed=config.MAX_SPEED,
                        input_shape=config.INPUT_SHAPE,
                        predict_throttle=False,
                        fake_images=False
                        )
        # create all scenarios
        scenarios: List[Scenario] = [
            BeamngSimulator.individualToScenario(
                starting_pos=road_generator.initial_node,
                individual=ind,
                variable_names=variable_names,
                road_generator=road_generator)
            for ind in list_individuals
        ]
        env = None
       
        outcomes = []
        steerings_all = []
        throttles_all = []

        # iterate over all scenarios
        for j, scenario in enumerate(scenarios):
            do_repeat = True
            repeat_counter = 0
            BeamngSimulator.scenario_counter += 1
            log.info(f"[BeamngSimulator] Simulating {BeamngSimulator.scenario_counter}. scenario.")            
            while do_repeat and repeat_counter <= MAX_REPEAT:
                # clean_user_folder()
                # print("[BeamngSimulator] Waiting 2s.")
                time.sleep(2)
                try:
                    if env is None:
                        env = BeamngGymEnv(
                            seed=1,
                            add_to_port=0,
                            test_generator=road_generator,
                            #simulator_scene=GeneratedTrack(),
                            autopilot=False,
                            beamng_home=config.BEAMNG_HOME,
                            beamng_user=config.BEAMNG_USER,
                            oobtolerance = 2
                        )
                        log.info(f"[BeamngSimulation] BeamngGymEnv instance created")

                    outcome = BeamngSimulator.simulate_scanario(env, agent, scenario=scenario)
                    if outcome.simTime < config.MIN_SIM_TIME:
                        print("[BeamngSimulator] Repeating simulation because sim time indicates failure.")
                        raise Exception()

                    outcomes.append(outcome)
                    do_repeat = False
                    
                    steerings_all.append([s[0][0] for s in outcome.actions])
                    throttles_all.append([s[0][1] for s in outcome.actions])
                
                except Exception as e:
                    log.error("[BeamngSimulator] Exception during simulation ocurred: ")
                    traceback.print_exc()
                    time.sleep(TIME_WAIT)                
                    log.error(f"\n---- [BeamngSimulator] Repeating run for {repeat_counter}.time due to exception: ---- \n {e} \n")
                    repeat_counter += 1

                    # remove outcome if added
                    if len(outcomes) == j + 1:
                        outcomes = outcomes[:-1]
                finally:
                    if env is not None:
                        log.info(f"[BeamngSimulation] BeamngGymEnv close...")
                        env.close()
                        env = None
                    try:
                        kill_beamng_simulator()
                    except Exception as e:
                        print("[BeamngSimulator] Exception ocurred when killing simulator.")
                        print(e)

        assert len(outcomes) == len(scenarios), "Number scenarios does not match number outcomes."
        
        # convert the outcomes to sbt format
        simouts = []
        for i, scenario in enumerate(scenarios):
            log.info(f"[BeamngSimulator] Accessing {i}th outcome")
            outcome = outcomes[i]
            control_points = scenario.road.control_points
            control_points_serializable = [(point.x, point.y) for point in control_points[1:]] if control_points else None

            simouts.append(
                SimulationOutput(
                simTime=outcome.simTime,
                times=outcome.frames,
                location={"ego": [(x[0], x[1]) for x in outcome.pos]},
                velocity={
                    "ego": BeamngSimulator._calculate_velocities(
                        outcome.pos, outcome.speeds
                    )
                },
                speed={"ego": outcome.speeds},
                acceleration={"ego": []},
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
                            "simulator": "BeamNG",
                            "road":  scenario.
                                        road.
                                        get_concrete_representation(to_plot=True),
                            "road_control_points": control_points_serializable,
                            "steerings" : steerings_all[i],
                            "throttles" : throttles_all[i],
                            "fps_rate" : outcome.fpsRate 
                            }
            )
            )

            log.info("[BeamngSimulator] All scenarios simulated.")
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
                
        log.info(f"[BeamngSimulator] angles: {angles}")
        log.info(f"[BeamngSimulator] seg_lengths: {seg_lengths}")

        # generate the road string from the configuration
        seg_lengths  = seg_lengths if len(seg_lengths) > 0 else None
        road = road_generator.generate(starting_pos=starting_pos,
                                            angles=angles, 
                                           seg_lengths=seg_lengths,
                                           simulator_name=config.BEAMNG_SIM_NAME)
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
    def simulate_scanario(env, agent: Agent, scenario: Scenario
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

            try:
                # reset the scene to match the scenario
                obs = env.reset(skip_generation=False, road=road)
            except Exception as e:
                # HACK temporary to catch polygon invalid exception
                log.info(f"Catched exception {e}")
                
            time.sleep(2)

            log.info("[BeamngSimulator] Scenario constructed.")
            start = time.time()

            fps_time_start = time.time()
            counter = 0
            counter_all = []
            img_time_start = fps_time_start
            # run the scenario
            while not done:
                try:
                    record_image = False
                    if record_image and time.time() - img_time_start > 0.5:
                        img_time_start = time.time()        
                        # write image
                        write_image(BeamngSimulator.scenario_counter, 
                            image=obs,
                            simname="beamng")
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
                    # time.sleep(20)
                    state["simulator_name"] = config.DONKEY_SIM_NAME
                    actions = agent.predict(
                                    obs,
                                    state = state)
                    
                    # log.info(f"[BeamngSimulation] actions: {actions}")
                    obs, done, info = env.step(actions[0])
                
                    state["xte"] =  info.get("lateral_position") - CENTER_OFFSET #temporar hack
                    state["xte_pid"] = info.get("xte_pid", None)
                    state["speed"] = info.get("speed", None)
                    state["image"] = obs
                    state["is_success"] = info.get("is_success", None)
                    
                    # print("XTE", state["xte"])

                    # steering_angles.append(action[0])
                    pos = info.get("pos", None)
                    # log.info(f"[BeamngSimulator] position is: {pos}")
                    # # check if we are done

                    # # log.info(f"[BeamngSimulator] XTE is: {state['xte']}")
                    # log.info(f"[BeamngSimulator] s_success: {state['is_success']}")
                    # log.info(f"[BeamngSimulator] speed: {state['speed']}")
                    # log.info(f"[BeamngSimulator] XTE: {state['xte']}")
                    
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
                    # HACK: we cap values if they are above limit
                    speed_list.append(state["speed"])
                    actions_list.append(actions)

                    # log.info(f"Time between images: {timer() - end}")

                    # if timer() - end > 2:
                    #     msg = "Only 1 s between images. Indicating delay. Restart."
                    #     log.info(msg)
                    #     raise Exception(msg)

                    end = time.time()
                    time_elapsed = int(end - start)
                    
                    if time_elapsed % 2 == 0:
                        pass
                        # log.info(f"[BeamngSimulator] time_elapsed: {time_elapsed}")
                    elif time_elapsed > config.TIME_LIMIT:  
                        log.info(f"[BeamngSimulator] Over time limit, terminating.")    
                        done = True       
                    elif abs(state["xte"]) > config.MAX_XTE:
                        log.info("[BeamngSimulator] Is above MAXIMAL_XTE. Terminating.")
                        done = True
                    else:
                        pass
                except KeyboardInterrupt:
                    # self.client.stop()
                    raise KeyboardInterrupt
                
            end = time.time()
           
            fps_rate = np.sum(counter_all)/time_elapsed
            log.info(f"FPS rate: {fps_rate}")
        
            return ScenarioOutcome(
                frames=[x for x in range(len(pos_list))],
                pos=pos_list,
                xte=xte_list,
                speeds=speed_list,
                actions=actions_list,
                scenario=scenario,
                isSuccess=isSuccess,
                simTime=time_elapsed,
                fpsRate=fps_rate
            )



if __name__ == "__main__":
    road = [90.01, 105, 29.01, 69.02, 69.02] + [20, 20, 10, 10, 20]
    variable_names = [f"angle{i}" for i in range(5)] + [f"seg_length{i}" for i in range(5)] 
    BeamngSimulator.simulate(list_individuals= [road],
                                   variable_names=variable_names,
                                   do_visualize=True,
                                   scenario_path="",
                                   sim_time=10,
                                   time_step=10)