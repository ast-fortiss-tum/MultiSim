from dataclasses import dataclass
from typing import Dict
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import *
import logging as log
from fitness.migration import Migration

''' MultiSimAgreement Problem with diversified fitness function (diversity score will \
     be hold in the last ff). For now only for single valued ff'''
@dataclass
class ADSMultiSimAgreementProblemDiverse(Problem):
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
                         n_obj=len(simulate_functions) + 2, # fitness value for each + diversity score + distance fitness values
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
        assert fitness_function.diversify == True
        assert len(fitness_function.min_or_max) == 2

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
            objective_names = []
            for i,_ in enumerate(simulate_functions): 
                objective_names.append(fitness_function.name[0] + f"_S{i}")
            objective_names.append("archive_distance")
            objective_names.append("fitness_distance")

            self.objective_names = objective_names
            
        self.scenario_path = scenario_path
        self.problem_name = problem_name

        if other_parameters is not None:
            self.other_parameters = other_parameters

        if approx_eval_time is not None:
            self.approx_eval_time = approx_eval_time

        self.signs = []

        value = self.fitness_function.min_or_max[0]

        for _ in range(len(self.simulate_functions)):
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")
        # archive distance
        self.signs.append(-1)

        # distance fitness values
        self.signs.append(1)

        self.counter = 0

    # to be used if problem already initalized
    def update_name(self, 
                    attribute, 
                    value):
        import re
        pattern = fr"{attribute}\d+"
        replacement = f"{attribute}{value}"
        updated_name = re.sub(pattern, replacement, self.problem_name)
        self.problem_name = updated_name

    def _evaluate(self, x, out, *args, **kwargs):
        self.counter = self.counter + 1
        log.info(f"Running evaluation number {self.counter}")

        # Add individual to be process by fitness function
        kwargs["individual"] = x

        try:
            simout_sims = []
            for simulate_function in self.simulate_functions:
                simout_list = simulate_function(x,
                                                self.simulation_variables, 
                                                self.scenario_path, 
                                                sim_time=self.simulation_time,
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

        # store fitness, criticality and simout values
        for i in range(len(x)):
            critical_sims.append([])
            fitness_sims.append([])
            
            out["SO"].append(np.asarray(simout_sims[i]))

            archive_distance = 0
            for j in range(len(self.simulate_functions)):
                simout = simout_sims[i][j]
                fvalues = self.fitness_function.eval(simout, **kwargs)
                fitness = np.asarray(self.signs[j]) * np.array(fvalues[0:1])
                # the last value is the archive distance
                archive_distance = self.signs[j + 1] * np.array(fvalues[-1])
                fitness = np.append(fitness, archive_distance)

                # print(f"Fitness calculated: {fitness}")
                critical = self.critical_function.eval(fitness)

                critical_sims[i].append(critical)
                fitness_sims[i].append(fitness)

            # add the archive distance score
            # print(f"archive_distance: {archive_distance}")
            # print(f"fitness_sims {i}: {fitness_sims[i]}")
            # structure
            # f1: quality_fitness sim1, f2: quality fitness sim2, f3: archive distance

        vector_list = []
        label_list = []

        def get_euclid_dist(fitness_values):
            return 2*(np.max(fitness_values) - np.min(fitness_values)) / len(fitness_values)
        
        # print(f"fitness_sims: {fitness_sims}")
        # print(f"critical_sims: {critical_sims}")

        for i in range(len(x)):
            # get the criticality value in case we use algorithms that require that value
            # however it is not expressive for dual problems
            _, critical_migrated, migrated = \
                   self.migrate_function.eval(
                                       np.asarray(fitness_sims)[i,:],
                                       np.asarray(critical_sims)[i,:],
                                      fitness_function=self.fitness_function)
            
            fitness_migrated = np.asarray([])
            for f_sims in fitness_sims[i][:]:
                quality_fitness = f_sims[:-1] 
                fitness_migrated = np.concatenate((fitness_migrated, quality_fitness)) 

            fitness_migrated = np.concatenate((fitness_migrated, np.asarray([archive_distance]))) 

            # add distance fitness values as objective 
            dist_sims_fitness = get_euclid_dist(fitness_migrated[:-1])
            fitness_migrated = np.concatenate((fitness_migrated, np.asarray([dist_sims_fitness]))) 

            # print(f"fit migrated: {fitness_migrated}")

            # fitness_migrated = np.concatenate(fitness_sims[i][:])
            # the last ff is distance, take only once in combine fitness


            # print(f"fitness_migrated: {fitness_migrated}")
            # dist = get_avg_distance(np.asarray(fitness_all_sims))
        
            # log.info(f"Using migrated fitness vector: {fitness_migrated}")

            vector_list.append(fitness_migrated)
            label_list.append(critical_migrated)

        # print(f"vector_list: {vector_list}")
        out["F"] = np.vstack(np.array(vector_list))
        out["CB"] = np.array(label_list)
        out["CB_all"] = np.asarray(critical_sims, dtype=bool)
        out["F_all"] = np.asarray(fitness_sims, dtype=float)

        # print(f'F: {out["F"].shape}')
        # print(out["F"])
        # print(f'CB: {out["CB"].shape}')
        # print(f'CB_all: {out["CB_all"].shape}')
        # print(f'F_all: {out["F_all"].shape}')
        
        assert len(out["SO"][0]) == len(self.simulate_functions), \
            "Number of simout entry per individuals differs from number of simulators used."
        assert len(out["SO"]) == len(x), \
            "Number of simout array stored differs from number of individuals."
        print("evaluation finished")

    def is_simulation(self):
        return True
