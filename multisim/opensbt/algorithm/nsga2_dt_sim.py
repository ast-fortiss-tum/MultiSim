
import pymoo
from opensbt.evaluation.critical import *

# from opensbt.model_ga.individual import IndividualSimulated
# pymoo.core.individual.Individual = IndividualSimulated

# from opensbt.model_ga.population import PopulationExtended
# pymoo.core.population.Population = PopulationExtended

# from opensbt.model_ga.result  import ResultExtended
# pymoo.core.result.Result = ResultExtended

# from opensbt.model_ga.problem import ProblemExtended
# pymoo.core.problem.Problem = ProblemExtended

import copy
import sys
import time

from pymoo.core.result import Result
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoo.core.problem import Problem
from opensbt.exception.configuration import RestrictiveConfigException
from utils.archive import MemoryArchive
from visualization import output
from opensbt.problem.adas_problem import ADASProblem
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.utils.time_utils import convert_pymoo_time_to_seconds
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.algorithm.classification.decision_tree import decision_tree
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.utils.sorting import *
from opensbt.utils.time_utils import convert_pymoo_time_to_seconds
from opensbt.visualization.configuration import *
import numpy as np
import os
from opensbt.experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
import opensbt.quality_indicators.metrics.spread as qi
from opensbt.model_ga.result import *
import logging as log
from opensbt.quality_indicators.quality import *

from config import RESULTS_FOLDER_NAME_PREDEFINED

'''General flags'''
ALGORITHM_NAME = "NSGA-II-DT"
RESULTS_FOLDER = os.sep + "results" + os.sep + "single" +  os.sep
DO_EVAL_APPROXIMATION = True
WRITE_ALL_INDIVIDUALS = False
EXPERIMENTAL_MODE = True


class NSGAII_DT_SIM(object):

    algorithm_name = ALGORITHM_NAME

    def __init__(self,
                 problem: Problem,
                 config: SearchConfiguration):

        self.problem = problem
        self.config = config
        self.done_tree_iterations = 0
        self.res = None

        log.info(f"Initialized algorithm with config: {config.__dict__}")
        
    def run(self) -> ResultExtended:
        problem = self.problem
        config = self.config

        population_size = config.population_size
        maximal_execution_time = config.maximal_execution_time
        max_tree_iterations = config.max_tree_iterations
        num_offsprings = config.num_offsprings
        prob_crossover = config.prob_crossover
        eta_crossover = config.eta_crossover
        prob_mutation = config.prob_mutation
        eta_mutation = config.eta_mutation
        inner_num_gen = config.inner_num_gen

        '''Output variables'''
        all_population = Population()
        best_population = Population()
        best_population_unranked = Population()

        '''Initial conditions (initial region)'''
        xl = problem.xl
        xu = problem.xu

        # sampling = FloatRandomSampling()
        sampling = LHS()  # Latin Hypercube Sampling
        initial_population = sampling(problem, population_size)
        hist_holder = []

        '''Parameters of the algorithm'''
        if prob_mutation is None:
            prob_mutation = 1 / problem.n_var

        '''Parameter for evaluation'''
        if maximal_execution_time is not None:
            _maximal_execution_time = convert_pymoo_time_to_seconds(
                maximal_execution_time)
            max_tree_iterations = sys.maxsize
            log.info("Search is constrained by maximal execution time")
        elif max_tree_iterations is not None:
            _maximal_execution_time = sys.maxsize
            log.info("Search is constrained by maximal number of tree generations")
        else:
            log.info("Parameters are not correctly set, cannot start search.")
            sys.exit()

        ''' Computation start '''
        start_time = time.time()

        evaluate_individuals(initial_population, problem)
        initial_region = decision_tree.Region(xl, xu, initial_population)
        critical_regions = [initial_region]
        hist_holder = []

        # inner_algorithm is a template for an algorithm object that is stored for every generation
        inner_algorithm = NSGA2(
            pop_size=None,
            n_offsprings=None,
            sampling=None,
            crossover=SBX(prob=prob_crossover, eta=eta_crossover),
            mutation=PM(prob=prob_mutation, eta=eta_mutation),
            eliminate_duplicates=True,
            archive=MemoryArchive())
        
        tree_iteration = 0 
        n_func_evals = 0
        while n_func_evals < config.n_func_evals_lim:
            # extend the history by one generation
            hist_holder.extend([inner_algorithm] * inner_num_gen)

            log.info(f"running iteration {tree_iteration}")
            for critical_region in critical_regions:
                sub_problem = problem

                if prob_mutation == None:
                    prob_mutation = 1 / problem.n_var
                    
                nd_individuals_region = calc_nondominated_individuals(
                    critical_region.population)
                initial_population = Population(
                    individuals=nd_individuals_region)
                pop_size = len(initial_population)


                algorithm = NSGA2(
                    pop_size=pop_size,
                    n_offsprings=num_offsprings,
                    sampling=initial_population,
                    crossover=SBX(prob=prob_crossover, eta=eta_crossover),
                    mutation=PM(prob=prob_mutation, eta=eta_mutation),
                    eliminate_duplicates=True,
                    archive=MemoryArchive())

                termination = get_termination("n_gen", inner_num_gen)

                res = minimize(sub_problem,
                               algorithm,
                               termination,
                               seed=1,
                               save_history=True,
                               verbose=False)

                n_func_evals += res.history[-1].evaluator.n_eval

                # TODO modify a history for NSGA-II-DT
                self.update_history(res, hist_holder, tree_iteration, inner_num_gen, inner_algorithm)

                hist = res.history

                # hist[i] is an object of <class 'pymoo.algorithms.moo.nsga2.NSGA2'>
                best_population_unranked = Population.merge(
                    best_population, res.opt)
                # TODO eliminate duplicates when merging
                best_population = get_nondominated_population(
                    best_population_unranked)
                for generation in hist:
                    all_population = Population.merge(
                        all_population, generation.pop)

            initial_region.population = best_population

            regions = decision_tree.generate_critical_regions(
                all_population, problem, save_folder=None)
            critical_regions = [
                region for region in regions if region.is_critical]
            if not critical_regions:
                critical_regions = [initial_region]
            tree_iteration += 1

        execution_time = time.time() - start_time

        # store the number of achieved tree iterations
        self.done_tree_iterations = tree_iteration
        result = self.create_result(problem, hist_holder, inner_algorithm, execution_time)
        self.res = result
        return result

    def update_history(self, res, hist_holder, tree_iteration, inner_num_gen, inner_algorithm):
        for i in range(inner_num_gen):
            pop = Population.merge(
                hist_holder[tree_iteration * inner_num_gen + i].pop, res.history[i].pop)
            # copy a template of the inner algorithm, and then modify its population and other properties
            algo = copy.deepcopy(inner_algorithm)
            algo.pop = pop
            opt_pop = Population(
                individuals=calc_nondominated_individuals(pop))
            algo.opt = opt_pop
            hist_holder[tree_iteration * inner_num_gen + i] = algo

    def create_result(self, problem, hist_holder, inner_algorithm, execution_time):
        # TODO calculate res.opt
        I = 0
        for algo in hist_holder:
            I += len(algo.pop)
            algo.evaluator.n_eval = I
            algo.start_time = 0
            algo.problem = problem
            algo.result()

        res_holder = ResultExtended()
        res_holder.algorithm = inner_algorithm
        res_holder.algorithm.evaluator.n_eval = I
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.history = hist_holder
        res_holder.exec_time = execution_time

        # calculate total optimal population using individuals from all iterations
        opt_all = Population()
        for algo in hist_holder:
            opt_all = Population.merge(opt_all, algo.pop)
        # log.info(f"opt_all: {opt_all}")
        opt_all_nds = get_nondominated_population(opt_all)
        res_holder.opt = opt_all_nds

        return res_holder
    
    def write_results(self, results_folder=RESULTS_FOLDER):
        algorithm_name = self.algorithm_name
        if self.res is None:
            log.info(
                "Result object is None. Execute algorithm first, before writing results.")
            return

        res_holder = self.res
        config = self.config
        problem = self.problem
        hist_holder = res_holder.history

        done_tree_iterations = self.done_tree_iterations

        '''For forwarding to plotter'''
        algorithm_parameters = {
            'Number of maximal tree generations': str(config.max_tree_iterations),
            'Number performed tree iterations': done_tree_iterations,
            "Population size": str(config.population_size),
            "Number of generations": str(config.inner_num_gen),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability": str(config.prob_crossover),
            "Crossover eta": str(config.eta_crossover),
            "Mutation probability": str(config.prob_mutation),
            "Mutation eta": str(config.eta_mutation)}

        log.info(f"=====[{algorithm_name}] Writing results...")

        save_folder = output.create_save_folder(
            problem, 
            results_folder, 
            algorithm_name, 
            is_experimental= False,
            folder_name = RESULTS_FOLDER_NAME_PREDEFINED)

        # output of plots for every tree iteration
        # for i in range(done_tree_iterations):
        #     res_holder_tmp = copy.deepcopy(res_holder)
        #     res_holder_tmp.history = hist_holder[0:(
        #         i + 1) * config.inner_num_gen]

        #     opt_all_tmp = Population()
        #     for algo in res_holder_tmp.history:
        #         opt_all_tmp = Population.merge(opt_all_tmp, algo.pop)
        #     opt_all_tmp_nds = get_nondominated_population(opt_all_tmp)
        #     res_holder_tmp.opt = opt_all_tmp_nds

        #     output.design_space(res_holder_tmp, save_folder +
        #                         "tree_iterations" + os.sep + "TI" + str(i) + os.sep)
        #     output.objective_space(
        #         res_holder_tmp, save_folder + "tree_iterations" + os.sep + "TI" + str(i) + os.sep)

        output.write_summary_results(res_holder, save_folder)
        output.igd_analysis(res_holder, save_folder)
        output.hypervolume_analysis(res_holder, save_folder)
        output.spread_analysis(res_holder, save_folder)
        output.write_calculation_properties(
            res_holder, save_folder, algorithm_name, algorithm_parameters)
        output.design_space(res_holder, save_folder)
        output.objective_space(res_holder, save_folder)
        output.optimal_individuals(res_holder, save_folder)
        output.all_critical_individuals(res_holder, save_folder)

        if WRITE_ALL_INDIVIDUALS:
           output.all_individuals(res_holder, save_folder)
        
        #persist results object
        #res_holder.persist(save_folder + "backup")

        return save_folder

if __name__ == "__main__":        
    
    problem = PymooTestProblem(
        'BNH', critical_function=CriticalBnhDivided())
    
    ########
    
    # class CriticalZDT3(Critical):
    #     def eval(self, vector_fitness: List[float], simout: SimulationOutput = None):
    #         if vector_fitness[0] <= 0.5 and vector_fitness[0] >= 0.4 and \
    #             vector_fitness[1] <= 0.25 and vector_fitness[0] >= 0:
    #             return True
    #         else:
    #             return False
            
    # problem = PymooTestProblem(
    #     'zdt3', critical_function=CriticalZDT3())
    
    ######
    
    config = DefaultSearchConfiguration()

    config.population_size = 50
    config.inner_num_gen = 5
    config.prob_mutation = 0.5
    config.n_func_evals_lim = 5000

    optimizer = NSGAII_DT_SIM(problem,config)
    optimizer.run()
    optimizer.write_results()