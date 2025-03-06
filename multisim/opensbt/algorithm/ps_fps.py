import pymoo

from opensbt.algorithm.ps import PureSampling
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import ProblemExtended
from opensbt.utils.fps import FPS
pymoo.core.problem.Problem = ProblemExtended
from pymoo.core.problem import Problem
from opensbt.utils.sampling import CartesianSampling

import os

class PureSamplingFPS(PureSampling):
    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = FPS):
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type)
        
