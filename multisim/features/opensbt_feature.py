import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

import os
from features.feature_map import FeatureMap
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.quality_indicators.quality import Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log
from opensbt.visualization.visualization3d import visualize_3d
import config
from road_generator.roads.simulator_road import SimulatorRoad

WRITE_ALL_INDIVIDUALS = True
BACKUP_FOLDER =  "backup" + os.sep
METRIC_PLOTS_FOLDER =  "metric_plots" + os.sep

def plot_feature_map_from_pop(pop: Population,
    save_folder: str,
    feature_names = ["curvature", "turn_count"],
    bound_up=[0.4,5],
    bound_low=[0,0],
    interval=[0.02,1],
    max_fitness=1,
    min_fitness=0,
    combine_strategy = "optimal",
    name_fmap = None,
    title_suffix = "",
    iterations = -1):

    map: FeatureMap = get_feature_map_from_pop(pop,
                                   feature_names,
                                   bound_up,
                                   bound_low,
                                   interval,
                                   max_fitness,
                                   min_fitness,
                                   combine_strategy)
    
    map.plot_map(filepath=save_folder,
                 filename=name_fmap,
                 title_suffix = title_suffix,
                 iterations=iterations)
    
    map.export_to_json(save_folder + title_suffix + "fmap.json")

    return map

def get_feature_map_from_pop(pop: Population,
    feature_names = ["curvature", "turn_count"],
    bound_up=[0.4,5],
    bound_low=[0,0],
    interval=[0.02,1],
    max_fitness=1,
    min_fitness=0,
    combine_strategy = "optimal"):

    pop = duplicate_free(pop)

    map: FeatureMap = FeatureMap(
                feature_names=feature_names,
                bound_up=bound_up,
                bound_low=bound_low,
                interval=interval,
                max_fitness=max_fitness,
                min_fitness=min_fitness,
                combine_strategy=combine_strategy)
        
    for ind in pop:
        # print(f"[in plot feature map] {ind.get('SO')}")
        # print(f"[in plot feature map] {type(ind.get('SO'))}")
        # print(f"[in plot feature map] is instance check: {type(ind.get('SO')) == SimulationOutput}")
        # print(f"[in plot feature map] is instance check: {isinstance(ind.get('SO'), SimulationOutput)}")

        if isinstance(ind.get("SO"), np.ndarray):
            # multi sim problem
            simout = ind.get("SO")[0]
            is_multisim = True
        else:
            simout = ind.get("SO")
            is_multisim = False
        if "road" in simout.otherParams:
            from road_generator.roads.road import Road
            road_points = simout.otherParams["road"]
            road = SimulatorRoad(road_width=config.ROAD_WIDTH,
                        road_points = Road.get_road_points_from_concrete(road_points),
                        control_points=Road.get_road_points_from_concrete(road_points))
            curvature = road.compute_curvature()
            num_turns = road.compute_num_turns()[0]

            # we assume the max of the fitness values is relevant (for parallel optimization for different sims)
            if is_multisim:
                # we take the "best" fitness value; if the agree on criticality cell is still critical, if not cell is not critical
                fitness = max(ind.get("F_all")[0][0],
                              ind.get("F_all")[1][0])
            else:
                fitness = ind.get("F")[0]
            map.update_map(feature_values=[curvature, num_turns], 
                           fitness=fitness,
                           input_values=ind.get("X"))
        else:
            print("No road im simout available, feature computation not possible.")
            return None
    return map
    
def plot_feature_map(res: Result,
    save_folder: str,
    feature_names = ["curvature", "turn_count"],
    bound_up=[0.4,5],
    bound_low=[0,0],
    interval=[0.02,1],
    max_fitness=1,
    min_fitness=0,
    combine_strategy = "optimal",
    name_fmap = None,
    title_suffix = "",
    iterations = -1):

    pop = res.obtain_archive()

    # print("len before dup free", len(pop))
    pop = duplicate_free(pop)

    # print("len after", len(pop))
 
    map = plot_feature_map_from_pop(pop,
                          save_folder,
                          feature_names,
                          bound_up,
                          bound_low,
                          interval,
                          max_fitness,
                          min_fitness,
                          combine_strategy,
                          name_fmap,
                          title_suffix,
                          iterations)
    return map