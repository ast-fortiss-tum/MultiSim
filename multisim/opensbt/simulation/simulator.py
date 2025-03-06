
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from pymoo.core.individual import Individual
from abc import ABC, abstractmethod, abstractstaticmethod
from opensbt.utils.encoder_utils import NumpyEncoder

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

'''
    Represent data output by a simulator

    Example json representation of a SimulationOutput instance:

    {
        "simTime" : 3,
        "times": [1.0,2.0,3.0],
        "location": { "ego" : [(1,1),(2,2),(3,3)],
                    "adversary" : [(4,1),(4,2),(4,3)},

        "velocity": { "ego" : [0.5,0.5,0.5],
                        "adversary" : [0.9,0.9,0.9],
                        },
        "collisions": [],
        "actors" : {1: "ego",
                        2: "adversary"
                    },
        "otherParams" : {
            "isCollision": False
        }
    }
        
'''
@dataclass
class SimulationOutput(object):
    simTime: float
    times: List
    location: Dict
    velocity: Dict
    speed: Dict
    acceleration: Dict
    yaw: Dict
    collisions: List
    actors: Dict
    otherParams: Dict

    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        return json.dumps(self.__dict__,
                        allow_nan=True, 
                        indent=4,
                        cls=NumpyEncoder)
    def to_dict(self):
        return {
            'simTime': self.simTime,
            'times': self.times,
            'location': self.location,
            'velocity': self.velocity,
            'speed': self.speed,
            'acceleration': self.acceleration,
            'yaw': self.yaw,
            'collisions': self.collisions,
            'actors': self.actors,
            'otherParams': self.otherParams
        }

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)


class Simulator(ABC):
    do_visualize = False
    sim_time = 10
    time_step = 0.01
    
    #TODO check whether to define sim time , time step as attribute here (instead in problem), show_visualization as well
    @abstractstaticmethod
    def simulate(list_individuals: List[Individual], 
                variable_names: List[str], 
                scenario_path: str, 
                sim_time: float, 
                time_step: float,
                do_visualize: bool = True) -> List[SimulationOutput]:
        raise NotImplementedError("Implement the simulation of a batch of scenario instances.")
    
    @abstractstaticmethod
    def kill():
        pass