from opensbt.model_ga.problem import *
from opensbt.simulation.simulator import *
from opensbt.algorithm.algorithm import *
from opensbt.experiment.search_configuration import *

class Experiment(object):
    # TODO refactor nsag2/nsga2-DT into a class to use algorithm instance in algorithm parameter
    def __init__(self, problem: Problem, algorithm: AlgorithmType, search_configuration: SearchConfiguration):
        self.problem = problem
        self.algorithm = algorithm
        self.search_configuration = search_configuration