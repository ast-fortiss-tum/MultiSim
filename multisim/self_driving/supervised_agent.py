from typing import Dict, Tuple

import numpy as np

from road_generator.custom_road_generator import CustomRoadGenerator
from self_driving.autopilot_model import AutopilotModel
from self_driving.custom_types import GymEnv
from self_driving.global_log import GlobalLog
from self_driving.agent import Agent
from self_driving.utils.dataset_utils import preprocess

import config
from sims.udacity.udacity_utils.envs.udacity.udacity_gym_env import UdacityGymEnv_RoadGen

class SupervisedAgent(Agent):
    def __init__(
        self,
        # env: GymEnv,
        env_name: str,
        model_path: str,
        max_speed: int,
        min_speed: int,
        input_shape: Tuple[int],
        predict_throttle: bool = False,
        fake_images: bool = False,
    ):
        super().__init__(
                         #env=env, 
                         env_name=env_name)

        self.logger = GlobalLog("supevised_agent")

        self.agent = AutopilotModel(env_name=env_name, input_shape=input_shape, predict_throttle=predict_throttle)
        self.agent.load(model_path=model_path)
        self.agent.model.compile(loss="sgd", metrics=["mse"])

        self.predict_throttle = predict_throttle
        self.model_path = model_path
        self.fake_images = fake_images

        self.max_speed = max_speed
        self.min_speed = min_speed

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        obs = preprocess(image=obs, env_name=self.env_name, fake_images=self.fake_images)

        # the model expects 4D array
        obs = np.array([obs])

        if self.predict_throttle:
            action = self.agent.model.predict(obs, batch_size=1)
            steering, throttle = action[0], action[1]
        else:
            multiplier = 1
            steering = float(self.agent.model.predict(obs, batch_size=1, verbose=0))
            if state["simulator_name"] == config.UDACITY_SIM_NAME:
                steering = config.STEERING_CORRECTION * steering
            speed = 0.0 if state.get("speed", None) is None else state["speed"]

            if speed > self.max_speed:
                print("slowing down")
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            # steering = self.change_steering(steering=steering)
            throttle = multiplier * np.clip(a=1.0 - steering**2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

            # if the track begins with a curve the model steers at the maximum and the throttle will be 0 since the
            # speed is 0. To counteract this give a non-zero throttle such that the car can start going
            if abs(steering) >= 1.0 and throttle == 0.0 and speed == 0.0 and len(state) > 0:
                self.logger.warn("Road starts with a curve! Giving the car an extra throttle")
                throttle = 0.5

        return np.asarray([[steering, throttle]], dtype = np.float32)
    


# test on some images
if __name__ == "__main__":
    test_generator = CustomRoadGenerator(map_size=250,
                                        num_control_nodes=3,
                                        seg_length=config.SEG_LENGTH)

    env = UdacityGymEnv_RoadGen(
        seed=1,
        test_generator=test_generator,
        exe_path=config.UDACITY_EXE_PATH
    )
    agent = SupervisedAgent(env=env,
                    env_name=config.UDACITY_SIM_NAME,
                    model_path=config.DNN_MODEL_PATH,
                    min_speed=config.MIN_SPEED,
                    max_speed=config.MAX_SPEED,
                    input_shape=config.INPUT_SHAPE,
                    predict_throttle=False,
                    fake_images=False
                    )
    from PIL import Image
    import os
    n_files = 10
    folder = os.getcwd() + os.sep + "testdata_3" + os.sep 
    #folder = os.getcwd() + os.sep 

    for i in range(n_files):
        path = folder +  f"udacity_image_{i}.png"
        image = Image.open(path)
        obs = np.asarray(image)
        predicted = agent.predict(obs=obs,state = dict(speed=config.MAX_SPEED, 
                                        simulator_name=config.UDACITY_SIM_NAME))[0][0]   
        f  = open(folder + f"udacity_steering_{i}.txt", "r")
        expected = float(f.read())
        f.close()

        difference = expected - predicted

        print(f"exp | pred | dif: {expected} {predicted} {difference}")
    env.close()
