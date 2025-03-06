from typing import List, Tuple
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.model_ga.population import PopulationExtended
from opensbt.evaluation.fitness import Fitness
from opensbt.evaluation.critical import Critical
from opensbt.simulation.simulator import SimulationOutput
import numpy as np

from config import CRITICAL_XTE, CRITICAL_AVG_XTE, MAX_ACC, CRITICAL_STEERING

class MaxXTEFitness(Fitness):
    def __init__(self, diversify=False) -> None:
        super().__init__()
        self.diversify = diversify

    @property
    def min_or_max(self):
        if self.diversify:
            return "max","max"
        else:
            return "max",

    @property
    def name(self):
        if self.diversify:
            return "Max XTE (neg)","Distance Archive"
        else:
            return "Max XTE (neg)",

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
  
        distance_archive = 0
        if self.diversify:
            if "algorithm" in kwargs:
                algorithm = kwargs["algorithm"]
                
                if not hasattr(algorithm, 'archive_novelty'):
                    distance_archive = 0
                    print("no archive novelty")
                else:
                    _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                    kwargs["individual"])
                    # print(f"archive size: {len(algorithm.archive_novelty)}")
            # print(f"distance_archive: {distance_archive}")
            f_vector = (max(traceXTE), distance_archive)
        else:
            f_vector = (max(traceXTE))
        return f_vector

class MaxXTEFitnessPredict(Fitness):
    def __init__(self, diversify=False) -> None:
        super().__init__()
        self.diversify = diversify

    @property
    def min_or_max(self):
        if self.diversify:
            return "max","max"
        else:
            return "max",

    @property
    def name(self):
        if self.diversify:
            return "Max XTE (neg)","Distance Archive"
        else:
            return "Max XTE (neg)",

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        # HACK generate fitness values for disagreement based search; 
        # HACK one should be critical and one not
        # TODO better approach is to train directly the classifier to predict fitness values for a simulator
        
        max_xte = CRITICAL_XTE + 0.1 if kwargs["order"] == 0 else CRITICAL_XTE - 0.1
  
        distance_archive = 0
        if self.diversify:
            if "algorithm" in kwargs:
                algorithm = kwargs["algorithm"]
                
                if not hasattr(algorithm, 'archive_novelty'):
                    distance_archive = 0
                    print("no archive novelty")
                else:
                    _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                    kwargs["individual"])
                    # print(f"archive size: {len(algorithm.archive_novelty)}")
            # print(f"distance_archive: {distance_archive}")
            f_vector = (max_xte, distance_archive)
        else:
            f_vector = (max_xte)
        return f_vector
# class MaxXTEFitnessDiverseNoveltyArchive(Fitness):
#     @property
#     def min_or_max(self):
#         return "max", "max", "max"

#     @property
#     def name(self):
#         return "Min Adapted Distance", "Velocity At Min Adapted Distance", "Distance Archive"

#     def eval(self, simout: SimulationOutput, **kwargs) -> float:
#         # use only adapted distance and velocity of the fitness comupation of existing function
#         vector_fitness_all = FitnessAdaptedDistSpeedRelVelocity().eval(simout)
#         adapted_distance = vector_fitness_all[0]
#         speed = vector_fitness_all[1]
        
#         algorithm = kwargs.get("algorithm")

#         if not hasattr(algorithm, 'archive_novelty'):
#             distance_archive = 0
#         else:
#             _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
#                                             kwargs["individual"])
#         print(f"distance_archive: {distance_archive}")
#         return adapted_distance, speed, distance_archive

class MaxXTEFitnessNoveltyArchive(Fitness):
    @property
    def min_or_max(self):
        return "max", "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)","Max Distance", "Max Distance Archive"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]

        algorithm = kwargs.get("algorithm")

        if not hasattr(algorithm, 'archive_novelty'):
            distance_archive = 0
        else:
            _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                            kwargs["individual"])
        return (max(traceXTE), distance_archive)

class MaxXTEFitnessDiverse(Fitness):
    @property
    def min_or_max(self):
        return "max", "max",

    @property
    def name(self):
        return "Max XTE (neg)","Max Distance"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        distance = get_min_distance_from_archive(kwargs["individual"], kwargs["archive"]) if "archive" in kwargs else 0
        return (max(traceXTE), distance)

class MaxAvgXTEFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Average XTE (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        return (max(traceXTE), np.average(traceXTE))

class MaxAvgXTEVelocityFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Average XTE (neg)", "Velocity (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        ind = np.argmax(traceXTE)
        return (np.average(traceXTE),velocities[ind])
    
class MaxXTESteeringChangeFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Max Steering Change (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        max_xte = np.max(traceXTE)
        steerings = [x for x in simout.otherParams["steerings"]]
        steering_derivative = np.max([abs(d) for d in calc_derivation(values=steerings)])
        return (max_xte, steering_derivative)
    
class MaxXTEAvgVelocityFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Velocity (neg)", "Average XTE (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        ind = np.argmax(traceXTE)
        return (np.max(traceXTE),velocities[ind], np.sum(traceXTE) / len(traceXTE))
                
class MaxXTEVelocityFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Velocity (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        ind = np.argmax(traceXTE)
        return (np.max(traceXTE),velocities[ind])
        
class MaxXTEAccFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Acceleration (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        acc = simout.acceleration["ego"]
    
        return (np.max(traceXTE),np.max(acc))

class MaxXTECrossingsFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Crossings (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
    
        return (np.max(traceXTE),
                calc_cross(simout.otherParams["xte"]))

##############

class MaxXTECriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_XTE

class AvgXTECriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_AVG_XTE

class MaxAccCriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[1]) > MAX_ACC

class MaxXTECriticalitySteering_Simple(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > 2.5

class MaxXTECriticalitySteering(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        return abs(vector_fitness[0]) > CRITICAL_XTE or abs(vector_fitness[1]) > CRITICAL_STEERING
        
######### Constrainted criticality functions ################### 
       
class MaxXTECriticalitySteering_Constrained(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > 2.5 and abs(vector_fitness[1]) > 0.2
    
class MaxXTEAvgXTECriticality_Constrained(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_XTE and abs(vector_fitness[1]) > 0.5
        
class MaxXTECriticality_Constrained(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > 2.5 and abs(vector_fitness[1]) > 27
    
class MaxXTEAvgXTECriticality_Constrained(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_XTE and abs(vector_fitness[1]) > 0.5
#########################

def calc_cross(trace_xte):
    last_dir = 1
    crosses = 0
    for i in range(1,len(trace_xte)):
        current = mysign(trace_xte[i])
        if current != last_dir:
            crosses +=1
            last_dir = current
    return crosses

def mysign(num):
    if np.sign(num) == 0:
        return 1
    else:
        return np.sign(num)
    
def calc_derivation(values: List, fps = 20, scale: int = 1):
    res=[0]
    for i in range(1,len(values)):
        a = (values[i] - values[i-1]) * fps * scale
        res.append(a)
    return res

def get_min_distance_from_archive(ind_X, archive: PopulationExtended):
    distances = []
    for ind_a in archive:
        dist = np.linalg.norm(ind_X - ind_a.get("X"))
        distances.append(dist)
    if len(distances) == 0:
        return 0
    else:
        min_dist = min(distances)
    return min_dist
