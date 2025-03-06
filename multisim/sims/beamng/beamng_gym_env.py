# Original author: Roma Sokolkov
# Edited by Antonin Raffin
from typing import NamedTuple

import gym
import numpy as np
from gym import spaces
from PIL import Image

from config import BEAMNG_SIM_NAME
from self_driving.custom_types import ObserveData
from sims.beamng.beamng_executor import BeamngExecutor
from sims.beamng.config import INPUT_DIM, MAP_SIZE, MAX_STEERING, MAX_THROTTLE, MIN_THROTTLE
from self_driving.global_log import GlobalLog
from road_generator.roads.road import Road
from road_generator.road_generator import RoadGenerator


class BeamngGymEnv(gym.Env):
    """
    Gym interface for BeamNG with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        add_to_port: int,
        test_generator: RoadGenerator,
        beamng_home: str,
        beamng_user: str,
        logs_folder_name: str = None,
        autopilot: bool = False,
        cyclegan_model = None,
        cyclegan_options =  None,
        oobtolerance = 0.95
    ):
        self.env_name=BEAMNG_SIM_NAME
    

        self.min_throttle = MIN_THROTTLE
        self.max_throttle = MAX_THROTTLE
        self.test_generator = test_generator
        self.logger = GlobalLog("BeamngGymEnv")

        self.executor = BeamngExecutor(
            beamng_home=beamng_home,
            oob_tolerance=oobtolerance,
            beamng_user=beamng_user,
            add_to_port=add_to_port,
            logs_folder_name=logs_folder_name,
            test_generator=self.test_generator,
            autopilot=autopilot,
        )

        self.action_space = spaces.Box(low=np.array([-MAX_STEERING]), high=np.array([MAX_STEERING]), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)

        self.seed(seed)
        self.count = 0

    def close_connection(self):
        return self.executor.close()

    def stop_simulation(self):
        raise NotImplementedError("Not implemented")

    def restart_simulation(self):
        raise NotImplementedError("Not implemented")

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, bool, dict)
        """
        # action[0] is the steering angle

        self.executor.take_action(steering=action[0], throttle=action[1] if len(action) > 1 else None)
        observe_data = self.observe()

        return observe_data


    def reset(self, skip_generation: bool = False, road: Road = None) -> np.ndarray:
        print("[BeamngGymEnv] in reset")
        self.executor.reset(skip_generation=skip_generation, road=road)
        observation, done, info = self.observe()
        
        print("[BeamngGymEnv] Observation")

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.executor.original_image
        return None

    def observe(self) -> ObserveData:
        """
        :return: (np.ndarray, bool, dict)
        """
        observation, done, info = self.executor.observe()

        # if self.cyclegan_model is not None:
        #     # im = self.get_fake_image(obs=observation)
        #     # fake = Image.fromarray(im)
        #     # original = Image.fromarray(observation)
        #     # original.show()
        #     # fake.show()
        #     return self.get_fake_image(obs=observation), done, info

        return observation, done, info

    def close(self):
        self.executor.close()

    def seed(self, seed=None):
        pass
