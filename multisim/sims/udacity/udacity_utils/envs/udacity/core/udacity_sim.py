"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Original author: Tawn Kramer

import base64
import time
from io import BytesIO
from threading import Thread
from typing import Tuple, Dict, List

import numpy as np
import socketio
from PIL import Image
from flask import Flask

from road_generator.road_generator import RoadGenerator
from sims.udacity.udacity_utils.envs.udacity.config import INPUT_DIM

from config import MAX_XTE as MAX_CTE_ERROR
from sims.udacity.udacity_utils.envs.udacity.core.client import start_app
from sims.udacity.udacity_utils.global_log import GlobalLog
import time as time

sio = socketio.Server()
flask_app = Flask(__name__)

last_obs = None
is_connect = False
deployed_track_string = None
generated_track_string = None
steering = 0.0
throttle = 0.0
speed = 0.0
cte = 0.0
cte_pid = 0.0
hit = 0.0
done = False
image_array = None
track_sent = False
pos_x = 0.0
pos_y = 0.0
pos_z = 0.0

import os

def write_to_logfile(msg, file_path=os.getcwd() + os.sep + "logs" + os.sep + f"times_{os.name}.log"):
    with open(file_path, 'a' if os.path.exists(file_path) else 'w') as file:
        file.write(str(msg) + '\n')


@sio.on("connect")
def connect(sid, environ) -> None:
    global is_connect
    is_connect = True
    print("Connect to Udacity simulator: {}".format(sid))
    send_control(steering_angle=0, throttle=0)


def send_control(steering_angle: float, throttle: float) -> None:  
    #write_to_logfile("OUT: " + str(int(time.time()*1000)))
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
        skip_sid=True,
    )


def send_track(track_string: str) -> None:
    global track_sent
    if not track_sent:
        sio.emit("track", data={"track_string": track_string}, skip_sid=True)
        track_sent = True


def send_reset() -> None:
    sio.emit("reset", data={}, skip_sid=True)


@sio.on("telemetry")
def telemetry(sid, data) -> None:
    global steering
    global throttle
    global speed
    global cte
    global hit
    global image_array
    global deployed_track_string
    global generated_track_string
    global done
    global cte_pid
    global pos_x
    global pos_y
    global pos_z

    if data:
        #write_to_logfile("IN: " + str(int(time.time()*1000)))

        #################
        speed = float(data["speed"].replace(",",".")) * 3.6  # conversion m/s to km/h
        cte = float(data["cte"].replace(",","."))
        cte_pid = float(data["cte_pid"].replace(",","."))
        pos_x = float(data["pos_x"].replace(",","."))
        pos_y = float(data["pos_y"].replace(",","."))
        pos_z = float(data["pos_z"].replace(",","."))
        hit = data["hit"]
        deployed_track_string = data["track"]
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_array = np.copy(np.array(image))

        if done:
            send_reset()
        elif (
            generated_track_string is not None
            and deployed_track_string != generated_track_string
        ):
            # TODO: add timeout
            send_track(track_string=generated_track_string)
        else:
            send_control(steering_angle=steering, throttle=throttle)


class UdacitySimController:
    """
    Wrapper for communicating with unity simulation.
    """

    def __init__(
        self,
        port: int,
        test_generator: RoadGenerator = None,
    ):
        self.port = port
        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.test_generator = test_generator
        self.max_cte_error = MAX_CTE_ERROR

        self.is_success = 0
        self.current_track = None
        self.image_array = np.zeros(self.camera_img_size)

        self.logger = GlobalLog("UdacitySimController")
    
        print("[UdacitySimController] Before starting thread")

        self.client_thread = Thread(target=start_app, args=(flask_app, sio, self.port))
        self.client_thread.daemon = True
        self.client_thread.start()
        self.logger = GlobalLog("UdacitySimController")

        while not is_connect:
            time.sleep(0.3)
            print("sleep...and repeat to connect")

        print("[UdacitySimController] Thread started")

    def reset(
        self,
        skip_generation: bool = False,
        track_string: str = None,
    ) -> None:
        global last_obs
        global speed
        global throttle
        global steering
        global image_array
        global hit
        global cte
        global cte_pid
        global done
        global generated_track_string
        global track_sent
        global pos_x
        global pos_y
        global pos_z

        last_obs = None
        speed = 0.0
        throttle = 0.0
        steering = 0.0
        self.image_array = np.zeros(self.camera_img_size)
        hit = "none"
        cte = 0.0
        cte_pid = 0.0
        done = False
        generated_track_string = None
        track_sent = False
        pos_x = 0.0
        pos_y = 0.0
        pos_z = 0.0

        self.is_success = 0
        self.current_track = None

        if not skip_generation and track_string is not None:
            generated_track_string = track_string

        time.sleep(1.5)

        # if not skip_generation:
        #     assert self.test_generator is not None, "Test generator is not instantiated"
        #     # self.generate_track(mut_point=mut_point, angles=angles)
        #     self.generate(starting_pos=starting_pos,
        #                     angles=angles, 
        #                                    seg_lengths=seg_lengths)
        # time.sleep(1)

    def generate_track(self, track_string: str = None):
        global generated_track_string

        if track_string is not None:
            generated_track_string = track_string

    # def generate_track(
    #     self,
    #     mut_point: int = None,
    #     generated_track: Road = None,
    #     angles: List[int] = [],
    # ):
    #     global generated_track_string

    #     if generated_track is None:
    #         start_time = time.perf_counter()
    #         self.logger.debug("Start generating track")
    #         track = self.test_generator.generate(mut_point, angles=angles)
    #         self.current_track = track
    #         self.logger.debug(track.get_control_points())
    #         self.logger.debug(
    #             "Track generated: {:.2f}s".format(time.perf_counter() - start_time)
    #         )
    #     else:
    #         self.current_track = generated_track

    #     generated_track_string = self.current_track.serialize_concrete_representation(
    #         cr=self.current_track.get_concrete_representation()
    #     )

    @staticmethod
    def take_action(action: np.ndarray) -> None:
        global throttle
        global steering
        steering = action[0][0]
        throttle = action[0][1]

        # steering = action[0]
        # throttle = action[1]

    def observe(self) -> Tuple[np.ndarray, bool, Dict]:
        global last_obs
        global image_array
        global done
        global speed
        global cte_pid
        global pos_x
        global pos_y
        global pos_z
        global cte

        while last_obs is image_array:
            time.sleep(1.0 / 120.0)
        last_obs = image_array
        self.image_array = image_array

        done = self.is_game_over()
        # z and y coordinates appear to be switched
        info = {
            "is_success": self.is_success,
            "track": self.current_track,
            "speed": speed,
            "pos": (pos_x, pos_z, pos_y),
            "cte": cte,
        }

        return last_obs, done, info

    def quit(self):
        self.logger.info("Stopping client")

    def is_game_over(self) -> bool:
        global cte
        global hit
        global speed

        if abs(cte) > self.max_cte_error or hit != "none":
            if abs(cte) > self.max_cte_error:
                self.is_success = 0
            else:
                self.is_success = 1
            return True
        return False
