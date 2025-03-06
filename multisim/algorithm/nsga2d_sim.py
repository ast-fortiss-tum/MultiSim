
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from output import output_fmap, output_other

from algorithm.nsga2d.nsga2d import NSGA2D

from opensbt.simulation.simulator import SimulationOutput
from opensbt.utils.archive import MemoryArchive

import os
import sys
from pathlib import Path
from typing import List

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from opensbt.algorithm.classification.decision_tree.decision_tree import *
from opensbt.evaluation.critical import Critical, CriticalBnhDivided
from opensbt.experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.visualization import output
import opensbt.quality_indicators.metrics.spread as qi
import logging as log
from opensbt.problem import *
from opensbt.model_ga.result import *
from config import *
from pymoo.operators.sampling.lhs import LHS
from features import opensbt_feature
from opensbt.visualization import output_predict
from opensbt.problem.adas_multi_sim_aggreement_problem_diverse_predict import ADSMultiSimAgreementProblemDiversePredict

class NSGAIID_SIM(object):
    
    algorithm_name =  "NSGAII-D"

    save_folder: str

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):

        self.config = config
        self.problem = problem
        self.res = None

        if self.config.prob_mutation is None:
            self.config.prob_mutation = 1 / problem.n_var

        if config.sampling is not None:
            sampling = config.sampling
        else:
            sampling = LHS()
        
        if config.crossover is not None:
            crossover = config.crossover
        else:
            crossover = SBX(prob=config.prob_crossover, eta=config.eta_crossover)

        if config.mutation is not None:
            mutation = config.mutation
        else:
            mutation = PM(prob=config.prob_mutation, eta=config.eta_mutation)

        self.algorithm = NSGA2D(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            n_repopulate_max=config.n_repopulate_max,
            archive_threshold=config.archive_threshold,
            seed = config.seed,
            archive= MemoryArchive(),
            bounds_normalize=
                [
                    self.problem.xu,
                    self.problem.xl
                ]
            )

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True
        
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> ResultExtended:
            # create a backup during the search for each generation
        algorithm = self.algorithm
        
        algorithm.setup(problem = self.problem, 
                        termination = self.termination,
                        save_history = self.save_history)
        
        save_folder = output.create_save_folder(self.problem, 
                                RESULTS_FOLDER,
                                algorithm_name=self.algorithm_name,
                                is_experimental=EXPERIMENTAL_MODE,
                                folder_name=self.config.write_subfolder_name)
        
        # output.backup_object(self.problem,
        #                      save_folder,
        #                      name = f"problem")
        
        # setup_logging(save_folder)

        while(algorithm.termination.do_continue()):
            algorithm.next()
            if BACKUP_ITERATIONS:
                n_iter = algorithm.n_iter - 1
                output.backup_object(algorithm, 
                              save_folder, 
                              name = f"algorithm_iter_{n_iter}")

        res = algorithm.result()
        self.res = res
        self.save_folder = save_folder
        
        # print("save_folder set:", self.save_folder)
        return res

    def write_results(self, results_folder = RESULTS_FOLDER, store_result = True):
        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info("Result object is None. Execute algorithm first, before writing results.")
            return
        log.info(f"=====[{self.algorithm_name}] Writing results...")
        problem = self.problem
        config = self.config
        res = self.res
        algorithm_parameters = {
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation)
        }
        
        save_folder = output.create_save_folder(res.problem, 
                                                results_folder, 
                                                algorithm_name, 
                                                is_experimental=EXPERIMENTAL_MODE,
                                                folder_name=config.write_subfolder_name)

        if store_result:
            res.persist(save_folder + "backup")
        # output.igd_analysis(res, save_folder)
        # output.gd_analysis(res,save_folder)
        # output.hypervolume_analysis(res, save_folder)
        # output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)

        output.all_critical_individuals(res, save_folder)
        output.write_generations(res, save_folder)
        output.write_criticality_all_sims(res, save_folder)
        output.write_fitness_all_sims(res, save_folder)
        output.backup_problem(res, save_folder)
        output.plot_roads_overview_generation(res, save_folder)

        if not EXPERIMENTAL_MODE:
            output.objective_space(res, save_folder)
            # output.design_space(res, save_folder)

            # output.plot_critical_roads(res, save_folder)

            # if DO_PLOT_GIFS:
            #     output.simulations(res, save_folder)
            
            # for type in ["X", "Y", "V", "XTE", "throttles", "steerings"]:
            #     output.comparison_trace(res, save_folder, mode=MODE_PLOT_RESULTS, type=type)
            # output.plot_generations_scenario(res, save_folder)
            output.write_simulation_output(res,save_folder)
            # output.scenario_plot_disagreements(res, save_folder)
        
            output.write_disagree_testcases(res, save_folder)        
            output.export_generations_roads(res, save_folder)
        
        output.export_roads(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)
  
        output_fmap.fmap_failure_sp(res, save_folder,max_fitness=0,
                                    min_fitness=-3)
        output_fmap.fmap_coverage(res, save_folder,max_fitness=0,
                                    min_fitness=-3)
        output_fmap.fmap_sparseness(res, save_folder,max_fitness=0,
                                    min_fitness=-3)
        opensbt_feature.plot_feature_map(res,
                                    save_folder,
                                    max_fitness=0,
                                    min_fitness=-3,
                                    title_suffix=res.problem.problem_name)
        output.calculate_n_crit_distinct(res,save_folder,
                                        bound_min=problem.xl,
                                        bound_max=problem.xu, 
                                        var="X")
    
        output.calculate_n_crit_distinct(res,save_folder,
                                        bound_min=config.ideal,
                                        bound_max=config.nadir, 
                                        var="F")
                
        output.write_multisim_analysis(res, save_folder)
        output.write_config(save_folder)
        output_other.calculate_avg_euclid(res, save_folder, critical_only = True, var = "X")
        
        ####### prediction
        if type(problem) == ADSMultiSimAgreementProblemDiversePredict:
            output_predict.write_predictions(res, save_folder)
            output_predict.write_prediction_summary(res, save_folder)

        res.persist(save_folder + "backup")

        return save_folder
    
if __name__ == "__main__":        
            
    problem = PymooTestProblem(
        'bnh', critical_function=CriticalBnhDivided())
    config = DefaultSearchConfiguration()

    config.population_size = 100
    config.n_generations = 10
    config.prob_mutation = 0.5
    config.n_func_evals_lim = 20
    config.n_repopulate_max = 10

    optimizer = NSGAIID_SIM(problem,config)
    optimizer.run()
    optimizer.write_results(
        ref_point_hv=np.asarray([200,200]), 
        ideal = np.asarray([0,0]), 
        nadir = np.asarray([200,200]), 
        results_folder = RESULTS_FOLDER)