from dataclasses import dataclass
from typing import Dict
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import *
import logging as log
from fitness.migration import Migration
@dataclass
class ADSMultiSimProblem(Problem):
    def __init__(self,
                 xl: List[float], 
                 xu: List[float], 
                 scenario_path: str, 
                 fitness_function: Fitness, 
                 simulate_functions: List,
                 migrate_function: Migration,
                 critical_function: Critical, 
                 simulation_time: float, 
                 sampling_time: float, 
                 simulation_variables: List[float], 
                 design_names: List[str] = None, 
                 objective_names: List[str] = None, 
                 problem_name: str = None, 
                 other_parameters: Dict = None,
                 approx_eval_time: float =None):

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

        assert xl is not None
        assert xu is not None
        assert scenario_path is not None
        assert fitness_function is not None
        assert len(simulate_functions) > 0
        assert migrate_function is not None
        assert simulation_time is not None
        assert sampling_time is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.fitness_function = fitness_function
        self.simulate_functions = simulate_functions
        self.migrate_function = migrate_function
        self.critical_function = critical_function
        self.simulation_time = simulation_time
        self.sampling_time = sampling_time
        self.simulation_variables = simulation_variables

        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name
            
        self.scenario_path = scenario_path
        self.problem_name = problem_name

        if other_parameters is not None:
            self.other_parameters = other_parameters

        if approx_eval_time is not None:
            self.approx_eval_time = approx_eval_time

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")

        self.counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.counter = self.counter + 1
        log.info(f"Running evaluation number {self.counter}")

        # Add individual to be process by fitness function
        kwargs["individual"] = x
        try:
            simout_sims = []
            for simulate_function in self.simulate_functions:
                simout_list = simulate_function(x, self.simulation_variables, self.scenario_path, sim_time=self.simulation_time,
                                                    time_step=self.sampling_time)
                simout_sims.append(np.asarray(simout_list))
        except Exception as e:
            log.info("Exception during simulation ocurred: ")
            # TODO handle exception, terminate, so that results are stored
            raise e

        # out["SO"] = [[],[]]
        fitness_sims = []
        critical_sims = []

        out["SO"] = []

        # invert array to have dimension: len(inds) * len(simulators)
        simout_sims = list(map(list, zip(*simout_sims)))

        # print(f"shape_x: {len(simout_sims)}")
        # print(f"shape_y: {len(simout_sims[0])}")

        # print(simout_sims)

        # store fitness, criticality and simout values
        for i in range(len(x)):
            critical_sims.append([])
            fitness_sims.append([])
            
            out["SO"].append(np.asarray(simout_sims[i]))
    
            for j in range(len(self.simulate_functions)):
                simout = simout_sims[i][j]
                fitness = np.asarray(self.signs) * np.array(self.fitness_function.eval(simout, **kwargs))
                critical = self.critical_function.eval(fitness)

                critical_sims[i].append(critical)
                fitness_sims[i].append(fitness)

        vector_list = []
        label_list = []

        # calculate for each scenario the migrated result
        for i in range(len(x)):
            fitness_migrated, critical_migrated, migrated = \
                    self.migrate_function.eval(
                                        np.asarray(fitness_sims)[i,:],
                                        np.asarray(critical_sims)[i,:],
                                        fitness_function=self.fitness_function)

            vector_list.append(fitness_migrated)
            label_list.append(critical_migrated)
        out["F"] = np.vstack(np.array(vector_list))
        out["CB"] = np.array(label_list)
        out["CB_all"] = np.asarray(critical_sims, dtype=bool)
        out["F_all"] = np.asarray(fitness_sims, dtype=float)

        # TODO store fitness and criticality values for each simulator
        assert len(out["SO"][0]) == len(self.simulate_functions), \
            "Number of simout entry per individuals differs from number of simulators used."
        assert len(out["SO"]) == len(x), \
            "Number of simout array stored differs from number of individuals."


    # self.counter = self.counter + 1
    # log.info(f"++ Evaluations executed {self.counter*100/(population_size*num_gen)}% ++")

    def is_simulation(self):
        return True
