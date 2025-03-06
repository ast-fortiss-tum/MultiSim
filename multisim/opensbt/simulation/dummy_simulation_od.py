from typing import List
from opensbt.simulation.simulator import Simulator, SimulationOutput
from math import sin, cos, pi, ceil
import numpy as np
from random import random
from opensbt.utils.geometric import *

from utils import geometric
import json

class DummySimulator3(Simulator):
    time_step = 1
    DETECTION_THRESH = 1
    RANDOMNESS_BIAS = 0.2
    
    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulate(list_individuals, variable_names, scenario_path: str, sim_time: float, time_step: float = time_step) -> List[SimulationOutput]:
        results = []
        for ind in list_individuals:
            simout = DummySimulator3.simulate_single(ind, 
                                                    variable_names, 
                                                    filepath=scenario_path, 
                                                    sim_time=sim_time,
                                                    time_step=time_step)
            results.append(simout)
        return results

    @staticmethod
    def simulate_single(vars, variable_names, filepath, sim_time: float, time_step: float, detection_dist = DETECTION_THRESH, randomness_bias = RANDOMNESS_BIAS) -> SimulationOutput:

        egoInitialVelocity = vars[1]
        pedInitialVelocity = vars[3]
        egoOrientation = vars[0]
        pedOrientation = vars[2]
        start_pos_ped = [2, 10] # should be also defined by input parameter
        start_pos_ego = [0, 0] # should be also defined by input parameter

        # first plan the motion of ego, get for each time step predicted position/velocity

        plan_ego_traj= planMotion(start_pos_ego, egoOrientation, egoInitialVelocity, sim_time, time_step)
        plan_adv_traj = planMotion(start_pos_ped, pedOrientation, pedInitialVelocity, sim_time, time_step)

        line_ego_points = (plan_ego_traj[1:3, 0], plan_ego_traj[1:3, 1])
        line_ped_points = (plan_adv_traj[1:3, 0], plan_adv_traj[1:3, 1])

        colpoint = geometric.intersection(list(line_ego_points), list(line_ped_points))

        n_steps = plan_ego_traj.shape[1]
        
        # iterate through each time step checking other actors nearby to avoid collision
        real_ego_traj = np.asarray([
                                    [plan_ego_traj[0,0]],
                                    [plan_ego_traj[1,0]],
                                    [plan_ego_traj[2,0]],
                                    [plan_ego_traj[3,0]],
                                    [plan_ego_traj[4,0]]
        ])

        # first position is real
        k = 1
        for i in range(1,n_steps):
            #print(f"current index: {i}")
            current_size = real_ego_traj[1,:]
            if len(current_size) < n_steps:
                pos_ego = [real_ego_traj[1,-1],real_ego_traj[2,-1]]
                pos_others = [[plan_adv_traj[1,i],plan_adv_traj[2,i]]]
                if are_actors_nearby(pos_ego, pos_others, detection_dist=detection_dist):
                    # plan make a step, ego vehicle trajectory in the form (Time, X, Y, V, Yaw)
                    # stop for some time steps
                    # t_stop = 3
                    # for t in range(0, n_steps - current_size - t_stop)):
                    new_t = real_ego_traj[0, i - 1] 
                    new_x =  real_ego_traj[1, i - 1] 
                    new_y =  real_ego_traj[2, i - 1] 
                    new_v = 0
                    new_yaw = real_ego_traj[4, i - 1] 
                    new_state =  np.asarray([[new_t],[new_x],[new_y], [new_v], [new_yaw]])
                    real_ego_traj = np.append(real_ego_traj, new_state, axis = 1)
                    # waiting_steps += t_stop - 1
                    # print(f"waiting_steps: {waiting_steps}")
                else:     
                    new_t =  plan_ego_traj[0, k] 
                    new_x =  plan_ego_traj[1, k] 
                    new_y =  plan_ego_traj[2, k] 
                    new_v = plan_ego_traj[3, k]
                    if real_ego_traj[3, k-1] == 0:
                        new_v = plan_ego_traj[3,k]/2
   
                    new_yaw = plan_ego_traj[4,k]     
                    new_state =  np.asarray([[new_t],[new_x],[new_y], [new_v], [new_yaw]])
                    real_ego_traj = np.append(real_ego_traj, new_state, axis = 1)
                    k += 1
                #print(f"len(real_ego_traj[3, :]): {len(real_ego_traj[3, :])}")

        # add some noise to location vector
        ego_location = [pos for pos in zip(list(real_ego_traj[1, :]), list(real_ego_traj[2, :]))]
        obj_location = [pos for pos in zip(list(plan_adv_traj[1, :]), list(plan_adv_traj[2, :]))]

        ego_location = [(pos[0] +random()*randomness_bias,pos[1] +random()*randomness_bias) for pos in ego_location]

        result = {
            "simTime": 0,
            "times": list(real_ego_traj[0, :]),
            "location": {"ego": ego_location,
                         "adversary": obj_location},

            "velocity": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "speed": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "acceleration": {"ego": list(real_ego_traj[3, :]),
                         "adversary": list(plan_adv_traj[3, :]),
                         },

            "yaw": {"ego": list(real_ego_traj[4, :]),
                           "adversary": list(plan_adv_traj[4, :]),
                           },

            "collisions": [],
            "actors" : { 
                "ego": "ego",
                "adversary": "adversary",
                "vehicles" : [],
                "pedestrians" : []
            },
            "type" : {"ego" : "vehicle",
                        "adversary" : "pedestrian"
                        },
            "otherParams": {}
        }

        otherParams = {}

        # Collision calculation (Only for fixed trajectories)
        if colpoint != []:
            dist_ego_colpoint = geometric.dist(colpoint, list(real_ego_traj[1:3, 0]))
            dist_ped_colpoint = geometric.dist(colpoint, list(plan_adv_traj[1:3, 0]))

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
        otherParams['simulator'] = "Dummy3"
        
        result["otherParams"] = otherParams

        

        return SimulationOutput.from_json(json.dumps(result))
    
def are_actors_nearby(pos_ego, pos_others, detection_dist=3):
    for i in range(0, len(pos_others)):
        if geometric.dist(pos_ego,pos_others[i]) < detection_dist:
            return True
    return False   

'''
    Linear motion planner
    
    Output: 
        Ego vehicle trajectory in the form (Time, X, Y, V, Yaw)
'''
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
