# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import time
from typing import NamedTuple

import gym
import numpy as np
from gym import spaces
from PIL import Image

from config import DONKEY_SIM_NAME
from self_driving.custom_types import ObserveData
from sims.donkey.config import BASE_PORT, BASE_SOCKET_LOCAL_ADDRESS, INPUT_DIM, MAX_STEERING
from sims.donkey.core.donkey_sim import DonkeyUnitySimController
from sims.donkey.scenes.simulator_scenes import SimulatorScene
from sims.unity_proc import UnityProcess
from self_driving.global_log import GlobalLog
from road_generator.road_generator import RoadGenerator
from road_generator.roads.road import Road

SIM_WAIT_START = 3

class DonkeyGymEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        add_to_port: int,
        simulator_scene: SimulatorScene,
        headless: bool = False,
        exe_path: str = None,
        test_generator: RoadGenerator = None):
        
        self.env_name=DONKEY_SIM_NAME
        self.exe_path = exe_path
        self.logger = GlobalLog("DonkeyGymEnv")
        self.test_generator = test_generator

        # TCP port for communicating with simulation
        if add_to_port == -1:
            port = int(os.environ.get("DONKEY_SIM_PORT", 9091))
            socket_local_address = int(os.environ.get("BASE_SOCKET_LOCAL_ADDRESS", 52804))
        else:
            port = BASE_PORT + add_to_port
            socket_local_address = BASE_SOCKET_LOCAL_ADDRESS + port

        self.logger.debug("Simulator port: {}".format(port))

        self.unity_process = None
        if self.exe_path is not None:
            self.logger.info("Starting DonkeyGym env")
            assert os.path.exists(self.exe_path), "Path {} does not exist".format(self.exe_path)
            # Start Unity simulation subprocess if needed
            self.unity_process = UnityProcess(sim_name=DONKEY_SIM_NAME)
            self.unity_process.start(sim_path=self.exe_path, headless=headless, port=port)

        time.sleep(SIM_WAIT_START)
        # start simulation com
        self.viewer = DonkeyUnitySimController(
            socket_local_address=socket_local_address,
            port=port,
            seed=seed,
            test_generator=test_generator,
            simulator_scene=simulator_scene,
        )

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]), high=np.array([MAX_STEERING, 1]), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)
        self.seed(seed)
        # wait until loaded
        self.viewer.wait_until_loaded()

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def stop_simulation(self):
        self.viewer.handler.send_pause_simulation()

    def restart_simulation(self):
        self.viewer.handler.send_restart_simulation()

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle
        self.viewer.take_action(action)
        observation, done, info = self.observe()

        return observation, done, info

    def reset(self, skip_generation: bool = False, road: Road = None) -> np.ndarray:
        print("[DonkeyGymEnv] in reset")
        self.viewer.reset(skip_generation=skip_generation, road=road)
        observation, done, info = self.observe()

        print("[DonkeyGymEnv] Observation")

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.viewer.handler.original_image
        return None

    def observe(self) -> ObserveData:
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        observation, done, info = self.viewer.observe()

        return observation, done, info

    def close(self):
        if self.unity_process is not None:
            self.unity_process.quit()
        self.viewer.quit()

    def pause_simulation(self):
        self.viewer.handler.send_pause_simulation()

    def restart_simulation(self):
        self.viewer.handler.send_restart_simulation()

    def seed(self, seed=None):
        self.viewer.seed(seed)
