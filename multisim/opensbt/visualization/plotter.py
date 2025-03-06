from math import sqrt
from opensbt.simulation.simulator import SimulationOutput
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.colors import colorConverter

from matplotlib.collections import PatchCollection
import os
import numpy as np
from opensbt.visualization.configuration import *
import config
import logging as log

def plot_gif(parameter_values, 
            simout: SimulationOutput, 
            savePath=None, 
            fileName=None,
            skip=2,
            plot_animation=True,
            focus_on_road=False,
            ax=None,
            format = "png",
            no_title = False,
            road = None,
            simulator = None,
            no_center = False):
    
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(3.9)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(1.8)

    if "pedestrian_size" in simout.otherParams:
        pedestrian_size = float(simout.otherParams["pedestrian_size"])
    else:
        pedestrian_size = 0.4

    # log.info(f"[plot_gif] Skip set to: {skip}")
    
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    colors = { "ego" : colorConverter.to_rgba('yellow', alpha=0.6),
            "adversary" : colorConverter.to_rgba('red', alpha=0.2),
               "vehicles": colorConverter.to_rgba('green', alpha=0.6),
               "pedestrians": colorConverter.to_rgba('purple', alpha=0.2),
               "traces": colorConverter.to_rgba('green', alpha=0.3),
               "road" : colorConverter.to_rgba('grey', alpha=0.2),}


    ''' actors =
        { 
            "ego" : <ego_name>,   //ego vehicle
            "adversary": <adv_name>  //primary adversary actor (TODO: remove)
            "vehicles" : <list of actor name that are vehicles> 
            "pedestrians" : <list of actors names that are pedstrians>
        }
     '''

    actors = simout.actors
    ego_name = actors["ego"]
    adversary_name = actors["adversary"] if "adversary" in actors else None

    vehicles_names = actors["vehicles"]
    pedestrians_names = actors["pedestrians"]
    actors_names = [ego_name] + \
                    ([adversary_name] if adversary_name is not None else []) + \
                    vehicles_names + \
                    pedestrians_names

    "Traces and yaws for actors"
    trace_ego = np.array(simout.location[ego_name])  # time series of Ego position
    yaw_ego = np.array(simout.yaw[ego_name])  # time series of Ego yaw

    if adversary_name is not None:
        trace_adversary = np.array(simout.location[adversary_name])  # time series of adversary position
        yaw_adversary = np.array(simout.yaw[adversary_name])  # time series of adversary yaw

    traces_vehicles = [np.array(simout.location[vehicle_name]) for vehicle_name in vehicles_names]
    yaws_vehicles = [np.array(simout.yaw[vehicle_name]) for vehicle_name in vehicles_names]

    traces_pedestrians = [np.array(simout.location[pedestrian_name]) for pedestrian_name in pedestrians_names]

    "Cartesian coordinates for actors"
    x_ego = trace_ego[:, 0]
    y_ego = trace_ego[:, 1]

    if adversary_name is not None:
        x_adversary = trace_adversary[:, 0]
        y_adversary = trace_adversary[:, 1]

    x_vehicles = [traces_vehicles[i][:, 0] for i in range(len(vehicles_names))]
    y_vehicles = [traces_vehicles[i][:, 1] for i in range(len(vehicles_names))]

    x_pedestrians = [traces_pedestrians[i][:, 0] for i in range(len(pedestrians_names))]
    y_pedestrians = [traces_pedestrians[i][:, 1] for i in range(len(pedestrians_names))]

    "Lables and titles"
    # label_parameters = str(["{:.3f}".format(v) for v in parameter_values])
    label_parameters = "\n".join(
        [", ".join(["{:.3f}".format(v) for v in parameter_values[i:i+5]]) for i in range(0, len(parameter_values), 5)]
    )
    
    if not no_title:
        ax.set_title(f"{simout.otherParams.get('simulator','Simulation')}\n{label_parameters}",
                  fontsize = 12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    "Plot road"
    if "road" in simout.otherParams:
        road = simout.otherParams["road"] if road is None else road
        plot_road(road, 
                color=colors["road"], 
                ax = ax,
                simulator= simout.otherParams["simulator"] if simulator is None else simulator,
                no_center= no_center)

    '''Finding suitable axis limits for visualization'''
    borders_x = []
    borders_y = []

    if "road" in simout.otherParams and focus_on_road:
        x_road = [p[0] for p in road]
        y_road = [p[1] for p in road]
        borders_x.extend([min(x_road), max(x_road)])
        borders_y.extend([min(y_road), max(y_road)])
    else:
        for actor_name in actors_names:
            trace_actor = np.array(simout.location[actor_name])
            x_actor = trace_actor[:,0]
            y_actor = trace_actor[:,1]
            borders_x.extend([min(x_actor), max(x_actor)])
            borders_y.extend([min(y_actor), max(y_actor)])

        left_border_x = min(borders_x)
        right_border_x = max(borders_x)
        center_x = (left_border_x + right_border_x) / 2

        left_border_y = min(borders_y)
        right_border_y = max(borders_y)
        center_y = (left_border_y + right_border_y) / 2

        maximum_size_of_axis = max((right_border_y - left_border_y), (right_border_x - left_border_x))

        ax.set(xlim=(center_x - maximum_size_of_axis * 0.55, center_x + maximum_size_of_axis * 0.55),
            ylim=(center_y - maximum_size_of_axis * 0.55, center_y + maximum_size_of_axis * 0.55))

    "Rectangle objects for ego and vehicles"
    patch_ego = Rectangle((0, 0), width=car_width, height=car_length, color=colors["ego"])
    patch_ego.set_width(car_width)
    patch_ego.set_height(car_length)

    patches_vehicles = [Rectangle((0, 0), width=car_width, height=car_length, color=colors["vehicles"]) for _ in
                        range(len(vehicles_names))]
    for patch_vehicle in patches_vehicles:
        patch_vehicle.set_width(car_width)
        patch_vehicle.set_height(car_length)
        # TODO: introduce separate dimensions of vehicles


    "Cicle object or patch object for adversary"

    circle_adversary = None
    patch_adversary = None

    if adversary_name is not None:
        if simout.type[adversary_name] == "pedestrian":
            circle_adversary = Circle((0, 0), radius=pedestrian_size, color=colors["adversary"])

        if simout.type[adversary_name] == "vehicle":
            patch_adversary = Rectangle((0, 0), width=car_width, height=car_length, color=colors["ego"])
            patch_adversary.set_width(car_width)
            patch_adversary.set_height(car_length)

    "Cicle objects for pedestrians"
    circles_pedestrians = [Circle((0, 0), radius=pedestrian_size, color=colors["pedestrians"]) for _ in range(len(pedestrians_names))]

    "Plot traces of all objects"
    ax.plot(x_ego, y_ego, color=colors["traces"])

    if adversary_name is not None:
        ax.plot(x_adversary, y_adversary, color=colors["adversary"])

    for key, pedestrian in enumerate(pedestrians_names):
        x_pedestrian = x_pedestrians[key]
        y_pedestrian = y_pedestrians[key]
        ax.plot(x_pedestrian, y_pedestrian, color=colors["pedestrians"])

    for key, vehicle in enumerate(vehicles_names):
        x_vehicle = x_vehicles[key]
        y_vehicle = y_vehicles[key]
        ax.plot(x_vehicle, y_vehicle, color=colors["vehicles"])

    annotations = []

    def update(i):
        # updating ego
        rotation_angle_ego = (yaw_ego[0::skip][i] - 90) % 360
        set_angle_incompatible(patch_ego,rotation_angle_ego)
        shift_angle_ego = rotation_angle_ego - 180 / np.pi * np.arctan(car_width / car_length)
        shift_x_ego = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(shift_angle_ego * np.pi / 180)
        shift_y_ego = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(shift_angle_ego * np.pi / 180)
        patch_ego.set_xy([x_ego[0::skip][i] + shift_x_ego, y_ego[0::skip][i] - shift_y_ego])
        ax.add_patch(patch_ego)
        
        for annotation in annotations:
            annotation.remove()
        annotations.clear()

        annotations.append(ax.annotate(f'Frame {i}', xy=(x_ego[0::skip][i] + shift_x_ego, y_ego[0::skip][i] - shift_y_ego), 
                    xytext=(0, -20),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=12, color='black'))


        # updating adversary (if pedestrian)
        if circle_adversary is not None:
            circle_adversary.center = x_adversary[0::skip][i], y_adversary[0::skip][i]
            ax.add_patch(circle_adversary)

        # updating adversary (if vehicle)
        if patch_adversary is not None:
            rotation_angle_adversary = (yaw_adversary[0::skip][i] - 90) % 360
            set_angle_incompatible(patch_adversary,rotation_angle_adversary)
            shift_angle_adversary = rotation_angle_adversary - 180 / np.pi * np.arctan(car_width / car_length)
            shift_x_adversary = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(
                shift_angle_adversary * np.pi / 180)
            shift_y_adversary = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(
                shift_angle_adversary * np.pi / 180)
            patch_adversary.set_xy([x_adversary[0::skip][i] + shift_x_adversary, y_adversary[0::skip][i] - shift_y_adversary])
            ax.add_patch(patch_adversary)

        # updating pedestrians
        for key, pedestrian in enumerate(pedestrians_names):
            x_pedestrian = x_pedestrians[key]
            y_pedestrian = y_pedestrians[key]
            circle_pedestrian = circles_pedestrians[key]
            circle_pedestrian.center = x_pedestrian[0::skip][i], y_pedestrian[0::skip][i]
            ax.add_patch(circle_pedestrian)

        # updating vehicles
        for key, vehicle in enumerate(vehicles_names):
            x_vehicle = x_vehicles[key]
            y_vehicle = y_vehicles[key]
            yaw_vehicle = yaws_vehicles[key]
            patch_vehicle = patches_vehicles[key]

            rotation_angle_vehicle = (yaw_vehicle[0::skip][i] - 90) % 360
            set_angle_incompatible(patch_vehicle, rotation_angle_vehicle)

            shift_angle_vehicle = rotation_angle_vehicle - 180 / np.pi * np.arctan(car_width / car_length)
            shift_x_vehicle = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(
                shift_angle_vehicle * np.pi / 180)
            shift_y_vehicle = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(
                shift_angle_vehicle * np.pi / 180)
            patch_vehicle.set_xy([x_vehicle[0::skip][i] + shift_x_vehicle, y_vehicle[0::skip][i] - shift_y_vehicle])
            ax.add_patch(patch_vehicle)
        return
    
    ax.axes.set_aspect('equal')

    if fig is not None:
        if plot_animation:
            if "sampling_rate" in simout.otherParams:
                recorded_fps = simout.otherParams["sampling_rate"]
            else:
                recorded_fps = 100

            ani = FuncAnimation(fig, update, frames=len(simout.times[0::skip]))
            writer = PillowWriter(fps=recorded_fps)
            ani.save(str(savePath) + str(fileName) + ".gif", writer=writer)
        else:
            plt.savefig(format = format, fname = str(savePath) + str(fileName) + f".{format}")
    
    if fig is not None:
        plt.clf()
        plt.close(fig)
    
    return


def plot_road(road, color="#808080", ax = None, simulator = None, no_center = False):
    # plot centerline
    points=[(p[0],p[1]) for p in road ]
    width = road[0][3]
    # print(f"plotting road: {points}")
    # if ax is None:
    #     plt.plot([p[0] for p in points], 
    #          [p[1] for p in points], 
    #          linewidth=5, 
    #          c=color)
    # else:
    #     ax.plot([p[0] for p in points], 
    #          [p[1] for p in points], 
    #          linewidth=5, 
    #          c=color)
    # plot lanes, border
    from self_driving.utils.visualization import RoadTestVisualizer
    
    road_test_visualizer = RoadTestVisualizer(map_size=config.MAP_SIZE, ax = ax)
    road_test_visualizer.visualize_road_basic(road_points=points, 
                                    width=width/2,
                                    do_save=False,
                                    car_trajectory=list(),
                                    simulator=simulator,
                                    no_center=no_center)

    # print("Road plotted.") 

# cath matplotlib incompat with new api call
def set_angle_incompatible(patch,angle):
    try:
        # print("Setting angle was not successful.")
        patch.set_angle(angle)
    except Exception as e:
        #print(e)
        if isinstance(patch,Rectangle):
            #print(type(patch))
            return Rectangle(patch.get_xy(), 
                             width=patch.get_width(), 
                             height=patch.get_height(), 
                             color=patch.get_facecolor(), 
                             angle=angle)
        
