from collections import namedtuple
from typing import List, Tuple

import numpy as np
from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Electrics, Sensor, State, Timer

VehicleStateProperties = [
    "timer",
    "pos",
    "dir",
    "vel",
    "steering",
    "steering_input",
    "brake",
    "brake_input",
    "throttle",
    "throttle_input",
    "wheelspeed",
    "vel_kmh",
]

VehicleState = namedtuple("VehicleState", VehicleStateProperties)


class VehicleStateReader:
    def __init__(self, vehicle: Vehicle, beamng: BeamNGpy, additional_sensors: List[Tuple[str, Sensor]] = None):
        self.vehicle = vehicle

        self.beamng = beamng
        self.sensors = None
        self.state: VehicleState = None
        self.vehicle_state = {}

        # assert 'state' in self.vehicle.sensors.keys(), "Default state sensor is missing"
        # Starting from BeamNG.tech 0.23.5_1 once the scenario is over a vehicle's sensors get automatically detached
        # Including the defatul state sensor... so we need to ensure that is there somehow, or stop reusing the vehicle
        # object across simulations
        try:
            state = State()
            self.vehicle.attach_sensor("state", state)
        except:
            pass

        electrics = Electrics()
        timer = Timer()

        self.vehicle.attach_sensor("electrics", electrics)
        self.vehicle.attach_sensor("timer", timer)
        # print(f"[VehicleStateReader] to {self.vehicle.vid} sensor attached: electrics")
        # print(f"[VehicleStateReader] to {self.vehicle.vid} sensor attached: timer")


        if additional_sensors:
            for (name, sensor) in additional_sensors:
                self.vehicle.attach_sensor(name, sensor)
                print(f"[VehicleStateReader] sensor attached: {name}")

    def get_state(self) -> VehicleState:
        return self.state

    def get_vehicle_bbox(self) -> dict:
        return self.vehicle.get_bbox()

    def update_state(self):
        # print(f"current state: {self.get_state()}")
        # print(f"[VSR] vehicle vid: {self.vehicle.vid}")
        # print(f"[VSR] vehicle state: {self.vehicle.state}")
     
        #sensors = self.vehicle.poll_sensors()

        sensors = self.beamng.poll_sensors(self.vehicle)
        self.sensors = sensors

        st = sensors["state"]
        ele = sensors["electrics"]

        # print(f"[VehicleStateReader] {st}")
        # print(f"[VehicleStateReader] sensors: {sensors}")
        vel = tuple(st["vel"])

        self.state = VehicleState(
            timer=sensors["timer"]["time"],
            pos=tuple(st["pos"]),
            dir=tuple(st["dir"]),
            vel=vel,
            steering=ele.get("steering", None),
            steering_input=ele.get("steering_input", None),
            brake=ele.get("brake", None),
            brake_input=ele.get("brake_input", None),
            throttle=ele.get("throttle", None),
            throttle_input=ele.get("throttle_input", None),
            wheelspeed=ele.get("wheelspeed", None),
            vel_kmh=int(round(np.linalg.norm(vel) * 3.6)),
        )
