from typing import List
from opensbt.simulation.simulator import Simulator, SimulationOutput
from math import sin, cos, pi, ceil
import numpy as np
from random import random
from opensbt.utils.geometric import *

from utils import geometric
import json

class DummySimulator1(Simulator):
    time_step = 0.05
    randomness_bias = 0.01
    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulate(list_individuals, variable_names, scenario_path: str, sim_time: float, time_step: float = time_step) -> List[SimulationOutput]:
        results = []
        for ind in list_individuals:
            simout = DummySimulator1.simulate_single(ind, 
                                                    variable_names, 
                                                    filepath=scenario_path, 
                                                    sim_time=sim_time,
                                                    time_step=time_step)
            results.append(simout)
        return results

    @staticmethod
    def simulate_single(vars, variable_names, filepath, sim_time: float, time_step: float) -> SimulationOutput:
        # print("*** INPUT SIMULATE ***")
        # print("vars: " + str(vars))
        # print("simTime: " + str(simTime))
        # print("samplingTime: " +  str(samplingTime))
        egoInitialVelocity = vars[1]
        pedInitialVelocity = vars[3]
        egoOrientation = vars[0]
        pedOrientation = vars[2]
        start_pos_ped = [2, 10] # should be also defined by input parameter
        start_pos_ego = [0, 0] # should be also defined by input parameter

        egoTrajectory = planMotion(start_pos_ego, egoOrientation, egoInitialVelocity, sim_time, time_step)
        objectTrajectory = planMotion(start_pos_ped, pedOrientation, pedInitialVelocity, sim_time, time_step)

        lineEgoPoints = (egoTrajectory[1:3, 0], egoTrajectory[1:3, 1])
        linePedPoints = (objectTrajectory[1:3, 0], objectTrajectory[1:3, 1])

        colpoint = geometric.intersection(list(lineEgoPoints), list(linePedPoints))
        # TODO add some dummy collision detection

        # add some noise to location vector
        ego_location = [pos for pos in zip(list(egoTrajectory[1, :]), list(egoTrajectory[2, :]))]
        obj_location = [pos for pos in zip(list(objectTrajectory[1, :]), list(objectTrajectory[2, :]))]

        ego_location = [(pos[0] +random()*DummySimulator1.randomness_bias,pos[1] +random()*DummySimulator1.randomness_bias) for pos in ego_location]

        result = {
            "simTime": 0,
            "times": list(egoTrajectory[0, :]),
            "location": {"ego": ego_location,
                         "adversary": obj_location},

            "velocity": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "speed": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "acceleration": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "yaw": {"ego": list(egoTrajectory[4, :]),
                           "adversary": list(objectTrajectory[4, :]),
                           },

            "collisions": [],
            "actors" : { 
                "ego": "ego",
                "adversary": "adversary",
                "vehicles" : [],
                "pedestrians" : []
            },
            "otherParams": {},
            "type" : {"ego" : "vehicle",
                        "adversary" : "pedestrian"
                        },
        }

        otherParams = {}

        if colpoint != []:
            dist_ego_colpoint = geometric.dist(colpoint, list(egoTrajectory[1:3, 0]))
            dist_ped_colpoint = geometric.dist(colpoint, list(objectTrajectory[1:3, 0]))

            t_col_ego = dist_ego_colpoint / egoInitialVelocity
            t_col_ped = dist_ped_colpoint / pedInitialVelocity

            # collision occurs when both objects reach line crossing at
            # the same time (with some tolerance)

            t_tolerance = 1;  # time tolerance for missed collision
            if (t_col_ego - t_col_ped) < t_tolerance:
                otherParams['isCollision'] = True
            else:
                otherParams['isCollision'] = False

            result["collisions"] = [[2]]
            result["otherParams"] = otherParams

        else:
            otherParams['isCollision'] = False

        otherParams['simulator'] = "Dummy1"
        result["otherParams"] = otherParams

        return SimulationOutput.from_json(json.dumps(result))

class DummySimulator2(Simulator):
    time_step = 0.05

    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulate(list_individuals, variable_names, scenario_path: str, sim_time: float, time_step: float = time_step) -> List[SimulationOutput]:
        results = []
        for ind in list_individuals:
            simout = DummySimulator2.simulate_single(ind, 
                                                    variable_names, 
                                                    filepath=scenario_path, 
                                                    sim_time=sim_time,
                                                    time_step=time_step)
            results.append(simout)
        return results

    @staticmethod
    def simulate_single(vars, variable_names, filepath, sim_time: float, time_step: float) -> SimulationOutput:
        # print("*** INPUT SIMULATE ***")
        # print("vars: " + str(vars))
        # print("simTime: " + str(simTime))
        # print("samplingTime: " +  str(samplingTime))
        egoInitialVelocity = vars[1]
        pedInitialVelocity = vars[3]
        egoOrientation = vars[0]
        pedOrientation = vars[2]
        start_pos_ped = [2, 10] # should be also defined by input parameter
        start_pos_ego = [0, 0] # should be also defined by input parameter

        egoTrajectory = planMotion(start_pos_ego, egoOrientation, egoInitialVelocity, sim_time, time_step)
        objectTrajectory = planMotion(start_pos_ped, pedOrientation, pedInitialVelocity, sim_time, time_step)

        lineEgoPoints = (egoTrajectory[1:3, 0], egoTrajectory[1:3, 1])
        linePedPoints = (objectTrajectory[1:3, 0], objectTrajectory[1:3, 1])

        colpoint = geometric.intersection(list(lineEgoPoints), list(linePedPoints))

        ego_location = [pos for pos in zip(list(egoTrajectory[1, :]), list(egoTrajectory[2, :]))]
        obj_location = [pos for pos in zip(list(objectTrajectory[1, :]), list(objectTrajectory[2, :]))]

        result = {
            "simTime": 0,
            "times": list(egoTrajectory[0, :]),
            "location": {"ego": ego_location,
                         "adversary": obj_location},

            "velocity": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "speed": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "acceleration": {"ego": list(egoTrajectory[3, :]),
                         "adversary": list(objectTrajectory[3, :]),
                         },

            "yaw": {"ego": list(egoTrajectory[4, :]),
                           "adversary": list(objectTrajectory[4, :]),
                           },

            "collisions": [],
            "actors" : { 
                "ego": "ego",
                "adversary": "adversary",
                "vehicles" : [],
                "pedestrians" : []
            },
            "otherParams": {},
            "type" : {"ego" : "vehicle",
                        "adversary" : "pedestrian"
                        },
        }

        otherParams = {}

        if colpoint != []:
            dist_ego_colpoint = geometric.dist(colpoint, list(egoTrajectory[1:3, 0]))
            dist_ped_colpoint = geometric.dist(colpoint, list(objectTrajectory[1:3, 0]))

            t_col_ego = dist_ego_colpoint / egoInitialVelocity
            t_col_ped = dist_ped_colpoint / pedInitialVelocity

            # collision occurs when both objects reach line crossing at
            # the same time (with some tolerance)

            t_tolerance = 1;  # time tolerance for missed collision
            if (t_col_ego - t_col_ped) < t_tolerance:
                otherParams['isCollision'] = True
            else:
                otherParams['isCollision'] = False
            result["collisions"] = [[2]]
            result["otherParams"] = otherParams

        else:
            otherParams['isCollision'] = False

        otherParams['simulator'] = "Dummy2"
        
        result["otherParams"] = otherParams

        return SimulationOutput.from_json(json.dumps(result))
    
def planMotion(startingPosition, orientation, velocity, simTime, samplingTime):
    theta = orientation
    T = simTime
    v = velocity
    t = samplingTime

    S = v * T
    distX = cos(theta * pi / 180) * S
    distY = sin(theta * pi / 180) * S
    nSteps = ceil(T / t)

    asize = nSteps + 1
    arrayTime = np.linspace(0, T, asize)
    arrayX = np.linspace(startingPosition[0], startingPosition[0] + distX, asize)
    arrayY = np.linspace(startingPosition[1], startingPosition[1] + distY, asize)
    arrayV = v * np.ones(asize, dtype=np.int64)
    # arrayV_x = arrayV * sin(theta * pi / 180)
    # arrayV_y = arrayV * cos(theta * pi / 180)
    arrayYaw = theta * np.ones(arrayTime.size)

    # print('theta = ', theta, 'v_x = ', arrayV_x, 'v_y = ', arrayV_y)
    return np.concatenate((arrayTime, arrayX, arrayY, arrayV, arrayYaw)).reshape(5, asize)
