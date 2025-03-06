# from opensbt.simulation.prescan_simulation import PrescanSimulator
import pymoo
from opensbt.algorithm.nlsga2_sim import NLSGAII_SIM

from opensbt.model_ga.individual import IndividualSimulated
from opensbt.utils.analysis_utils import read_critical_set
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from opensbt.algorithm.nsga2_sim import NSGAII_SIM
from opensbt.algorithm.ps_rand import PureSamplingRand

from default_experiments import *
from opensbt.visualization.combined import *
import os
from datetime import datetime
from pathlib import Path
from visualization import output
from opensbt.experiment.search_configuration import *
import sys
import argparse
import re
from default_experiments import *
from opensbt.utils.path import get_subfolders_from_folder
import logging as log
import time
import psutil
import gc
from pympler import asizeof
import signal
from opensbt.utils.sampling import cartesian_reference_set
import traceback
from opensbt.visualization.configuration import COVERAGE_METRIC_NAME
from opensbt.visualization.combined import write_last_metric_values
import dill

''' This script works in two modes:
    1. Repeated Executions of experiments  + Analysis of Results (Applying Metrics) 
    2. Analysis of Results 

    For 1) just configure the algorithms, their config, n_runs (bottom part of this file)
    
    For IGDE/CIGD analysis the estimated critical set needs to be pre-computed and past through the PATH_CRITICAL_SET variable.

    For 2) just run analysis.py and pass via -p the path to stored experiment results. 
'''

''' 
    Reference sets for SSBSE23 Paper
'''

######## Grid sampled

PATH_CRITICAL_SET = ""
# 25 samples/axis
PATH_CRITICAL_SET = "C:\\Users\\sorokin\\Documents\\Projects\\Results\\cs_approximation\\Demo_AVP_Reflection\\GS\\27-06-2023_03-08-19\\all_critical_individuals.csv" #Updated SUT
# PATH_CRITICAL_SET = "C:\\Users\\Lev\\Documents\\fortiss\\projects\\foceta\\Results\\cs_approximation\\Demo_AVP_Reflection\\PS\\27-06-2023_03-08-19\\all_critical_individuals.csv" #Updated SUT
DO_COVERAGE_ANALYSIS = True
DO_DS_ANALYSIS = False

class Analysis(object):
    ## Flags for restarts 

    DEBUG = False # save results in temp folder
    BACKUP_FOLDER =  "backup" + os.sep

    REPEAT_RUN = True
    MAX_REPEAT_FAILURE = 50
    TIME_WAIT = 10 # for restart in sec

    @staticmethod
    def run(analysis_name, 
            class_algos, 
            configs, 
            n_runs, 
            problems, 
            n_func_evals_lim, 
            folder_runs=None, 
            path_metrics=None, 
            n_select=None, 
            n_fitting_points=8, 
            distance_tick=250, 
            ref_point_hv=None, 
            algo_names=None,
            output_folder=None):
        
        if algo_names is None:
            algo_name_1 = class_algos[0].algorithm_name
            algo_name_2 = class_algos[1].algorithm_name
            algo_names = [algo_name_1, algo_name_2]
        else:
            algo_name_1 = algo_names[0]
            algo_name_2 = algo_names[1]
        class_algo = {}
        class_algo[algo_name_1] = algo_class_1
        class_algo[algo_name_2] = algo_class_2
        
        run_paths_all = {}
        if folder_runs is not None and len(folder_runs) > 1:
            folder_runs_1 = folder_runs[0]
            folder_runs_2 = folder_runs[1]
        else:
            folder_runs_1 = folder_runs
            folder_runs_2 = folder_runs_1

        # if no output folder provided, write analysis results in folder of first experiment;
        # Output folder is ONLY used for writing combined analysis results

        if output_folder is None:
            if len(folder_runs) > 1:
                output_folder = folder_runs_1  + f"comparison_{algo_names[0]}_{algo_names[1]}" + os.sep
                Path(output_folder).mkdir(parents=True, exist_ok=True)
            else:
                output_folder = folder_runs_1 
        else: 
            Path(output_folder).mkdir(parents=True, exist_ok=True)

        if (folder_runs is not None) and (path_metrics is not None) :
            log.info("Regenerationg comparison plot from given anaylsis data.")
            Analysis.regenerate_comparison_plot(algo_names, path_metrics, output_folder, n_func_evals_lim, n_fitting_points=n_fitting_points, distance_tick=distance_tick)
            log.info("Regeneration completed.")
        else: 
            if (folder_runs is not None):
                ''' provide folder of finished runs to do analysis '''
                log.info("Loading data from completed runs.")
                run_paths_all[algo_name_1] = get_subfolders_from_folder(
                                                        folder_runs_1 + algo_name_1 + os.sep )
                run_paths_all[algo_name_2] = get_subfolders_from_folder(
                                                        folder_runs_2  + algo_name_2 + os.sep) 
                
                # we can select subset of runs to use for evaluation
                if n_select is not None:
                    run_paths_all[algo_name_1] = run_paths_all[algo_name_1][1:(n_select+1)]
                    run_paths_all[algo_name_2] = run_paths_all[algo_name_2][1:(n_select+1)]

                log.basicConfig(
                                handlers=[  
                                log.FileHandler(filename=output_folder + os.sep + "log.txt",encoding='utf-8', mode='w'),
                                log.StreamHandler()
                                ], 
                                level=log.INFO)
                
                log.info(f"Analysed algorithms: {list(run_paths_all.keys())} \n")
            else:
                log.info("Executing algorithms for analysis.")
                if Analysis.DEBUG:
                    analysis_folder = str(os.getcwd()) + os.sep + "results" + os.sep + "analysis" + os.sep + analysis_name + os.sep +  str(n_runs) + "_runs" + os.sep + "temp" + os.sep
                else:
                    analysis_folder = str(os.getcwd()) + os.sep + "results" + os.sep + "analysis" + os.sep + analysis_name + os.sep +  str(n_runs) + "_runs" + os.sep + datetime.now().strftime(
                            "%d-%m-%Y_%H-%M-%S") + os.sep

                Path(analysis_folder).mkdir(parents=True, exist_ok=True)

                log.basicConfig(
                                handlers=[  
                                log.FileHandler(filename=analysis_folder + os.sep + "log.txt",encoding='utf-8', mode='w'),
                                log.StreamHandler()
                                ], 
                                level=log.INFO)


                ''' run the algorithms first and then do analysis '''
                configs = {algo_name_1 : configs[0],
                        algo_name_2 : configs[1]}

                ##### results are written in results/analysis/<problem>/<n_runs>/<date>/
                log.info(f"Search config 1: {config_1.__dict__} \n")
                log.info(f"Search config 2: {config_2.__dict__} \n")
                run_paths_all = Analysis.execute_algos(configs, class_algo, algo_names, n_runs, analysis_folder)
            log.info("Evaluating runs...")


            Analysis.evaluate_runs(algo_names, run_paths_all, output_folder, n_func_evals_lim, n_fitting_points=n_fitting_points, distance_tick=distance_tick, ref_point_hv=ref_point_hv)

    @staticmethod
    def get_memory_load():
       return psutil.Process().memory_info().rss / (1024 * 1024)

    @staticmethod
    def create_run_folder(analysis_folder, algorithm, run_num):
        i = run_num
        run_folder = analysis_folder + str(algorithm) + os.sep + str(f"run_{i}") + os.sep
        Path(run_folder).mkdir(parents=True, exist_ok=True)
        return run_folder
    
    @staticmethod
    def write_results_reduced(res, results_folder, algorithm_name,algo_parameters):
        # quality
        output.hypervolume_analysis(res, results_folder)
        output.spread_analysis(res, results_folder)
        output.design_space(res, results_folder)
        output.objective_space(res, results_folder)
        output.optimal_individuals(res, results_folder)
        output.write_summary_results(res, results_folder)
        output.write_calculation_properties(res, results_folder, algorithm_name=algorithm_name,algorithm_parameters=algo_parameters)
        output.write_simulation_output(res,results_folder)
        output.simulations(res, results_folder)
        output.all_individuals(res, results_folder)
        output.all_critical_individuals(res,results_folder)
        output.write_generations(res,results_folder)
        output.backup_problem(res, results_folder)
        
        # experimental
        output.spread_analysis_hitherto(res, results_folder)

    @staticmethod
    def execute_algos(configs, class_algo, algo_names, n_runs, analysis_folder):
        print("in execute_algos")
        ### make robust against unintended exits
        print("before loop")

        def handler(signum, frame):
            res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
            if res == 'y':
                log.info("Terminating program...")
                sys.exit(1)
    
        signal.signal(signal.SIGINT, handler)

        run_paths_all =  { algo_names[0] : [], algo_names[1]: []}

        for j, algo_name in enumerate(algo_names):
            problem = problems[j]
            for i in range(1,n_runs+1):
                do_repeat = True
                cnt = 0
                while(do_repeat and cnt <= Analysis.MAX_REPEAT_FAILURE):
                    try:
                        log.info(f"------ Running run {i} from {n_runs} with {algo_name} ------ \n")
                        run_folder = Analysis.create_run_folder(analysis_folder,algo_name,i)
                        config = configs[algo_name]
                        algo = class_algo[algo_name](
                                            problem=problem,
                                            config=config)
                        res = algo.run()
                        do_repeat = False

                        log.info("----- Storing result object ------")
                        res.persist(run_folder + BACKUP_FOLDER)
                        
                        log.info("----- Reduced writing of results ------")
                        Analysis.write_results_reduced(res, run_folder,algorithm_name=algo.algorithm_name, algo_parameters=config.__dict__)

                        run_paths_all[algo_name].append(run_folder)

                        log.info(f"---- Evaluating run {i} from {n_runs} with {algo_name} completed ----\n")
                        #log.info(f"Memory used before: {get_memory_load()} MB")
                        # print(f"Size of result is : {asizeof.asizeof(res.algorithm)}")
                        ### clean up
                        gc.collect()
                        #####################
                        #log.info(f"Memory user after: {get_memory_load()} MB")
                    except Exception as e:
                        traceback.print_exc()
                        # if problem.is_simulation():
                        #     PrescanSimulator.kill()
                        gc.collect()
                        if Analysis.REPEAT_RUN:
                            log.error(f"\n---- Repeating run {i} due to exception: ---- \n {e} \n")
                            time.sleep(Analysis.TIME_WAIT)
                            cnt =+ 1
                        else:
                            do_repeat = False
        log.info("---- All runs completed. ---\n-")
        
        return run_paths_all
    
    @staticmethod
    def regenerate_comparison_plot(algo_names, paths_results_csv, output_folder, n_func_evals_lim, n_fitting_points, distance_tick):
        # temporary some params hard coded
        subplot_names = ["CID"]
        metric_data_loaded = retrieve_metric_data_from_csv([paths_results_csv])
        make_comparison_plot(n_func_evals_lim, output_folder, metric_data_loaded, subplot_names, algo_names, distance_tick=distance_tick, suffix="_ds")

    @staticmethod
    def evaluate_runs(algo_names, 
                      run_paths_all, 
                      output_folder, 
                      n_func_evals_lim, 
                      n_fitting_points, 
                      distance_tick, 
                      ref_point_hv):
        
        algo_name_1 = algo_names[0]
        algo_name_2 = algo_names[1]

        #### Analysis

        if DO_DS_ANALYSIS:
            # Real pareto front is known
            pf_true = problem.pareto_front_n_points()

            # OR Estimate pareto front by aggregating run results
            paths_all = run_paths_all[algo_name_1] + run_paths_all[algo_name_2]
            
            log.info("---- Calculating estimated pareto front. ----\n")
            pf_estimated = calculate_combined_pf(paths_all, critical_only=True)
        
        if DO_COVERAGE_ANALYSIS:
            log.info("---- Reading approximated critical solutions set. ----\n")
            if PATH_CRITICAL_SET != "":
                critical_set_path = PATH_CRITICAL_SET
                cs_estimated = read_critical_set(critical_set_path)
            else:
                cs_estimated = cartesian_reference_set(problem, n_evals_by_axis = 5)
        
        # calculate estimated set of critical solutions
        # critical_all_algo1 = calculate_combined_crit_pop(run_paths_all[algo_name_1])
        # critical_all_algo2 = calculate_combined_crit_pop(run_paths_all[algo_name_2])

        # critical_all = Population.merge(critical_all_algo1,critical_all_algo2)
        # log.info(f"estimated pf: {pf_estimated}")
        # perform igd analysis/create plots 

        # result_runs_all = { algo_name_1 : [], 
        #                     algo_name_2 : []} 

        for algo_name in [algo_name_1, algo_name_2]:   

            for run_path in run_paths_all[algo_name]:            
                backup_path = run_path + BACKUP_FOLDER

                # load result
                res = None
                log.info(f"[Analysis] Reading result object from {backup_path}")

                with open(backup_path + "result", "rb") as f:
                    res = dill.load(f)

                # Unccoment following if "write_analysis_results" has to performed later. Release of memory is then not possible
                # result_runs_all[algo_name].append(res)  
                #       

                if DO_DS_ANALYSIS:
                    if pf_true is not None:
                        # output.gd_analysis(res, run_path, input_pf=pf_true, filename='gd_true')
                        output.gd_analysis(res, run_path, input_pf=pf_true, critical_only = True, filename='gd_true')
                        output.igd_analysis(res, run_path, input_pf=pf_true, filename='igd_true')
                
                    output.gd_analysis(res, run_path, input_pf=pf_estimated, critical_only=True, filename='gd')
                    output.hypervolume_analysis(res, run_path, ref_point_hv)

                if DO_COVERAGE_ANALYSIS:
                    # temporily faster readin, as problem object is smaller
                    # problem_path = backup_path + "problem" 
                    # with open(problem_path, "rb") as f:
                    #     problem = dill.load(f)
                    # ###################
                    # igde_analysis_hitherto_gens(problem, reference_set=cs_estimated, save_folder=run_path)
                    igde_analysis_hitherto(res, reference_set=cs_estimated, save_folder=run_path)

                if res is not None:
                    # release memory 
                    del res

        ########## create combined criticality plots

        # output_temp.design_space(problem, population=critical_all_algo1, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix=f"_{algo_name_1}", classification_type=None)
        # output_temp.design_space(problem, population=critical_all_algo2, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix=f"_{algo_name_2}", classification_type=None)
        # output_temp.design_space(problem, population=critical_all, save_folder = analysis_folder + os.sep + "critical_set" + os.sep, suffix="_all", classification_type=None)

        ######################### Objective Space metrics 

        if DO_DS_ANALYSIS:
            metric_names = ['hv', 'gd', 'sp']
            algo_names = [algo_name_1, algo_name_2]

            plot_array_hv = plot_combined_analysis('hv_global', run_paths_all, output_folder + "hv" + os.sep , n_func_evals_lim, n_fitting_points)
            plot_array_sp = plot_combined_analysis('sp', run_paths_all, output_folder + "sp" + os.sep, n_func_evals_lim, n_fitting_points)
            #plot_array_igd = plot_combined_analysis('igd', run_paths_all, analysis_folder + "igd" + os.sep, n_func_evals_lim, n_fitting_points)
            plot_array_gd = plot_combined_analysis('gd', run_paths_all, output_folder + "gd" + os.sep, n_func_evals_lim, n_fitting_points)

            # paths = write_metric_data_to_csv(analysis_folder, metric_names, algo_names, plot_array_hv, plot_array_gd, plot_array_sp)
            # plot_array_hv_loaded, plot_array_gd_loaded, plot_array_sp_loaded = retrieve_metric_data_from_csv(paths)

            subplot_metrics = [plot_array_hv, plot_array_gd, plot_array_sp]
            subplot_names = ["HV", "GD", "Spread"]
            make_subplots(n_func_evals_lim, output_folder, subplot_metrics, subplot_names, algo_names=[algo_name_1, algo_name_2])

        ########### Input Space Metrics/ Coverage Metric 
        if DO_COVERAGE_ANALYSIS:
            path_coverage_results = output_folder + "igde" + os.sep
            step_chkp = 250
            plot_array_igde = plot_combined_analysis("IGDE", run_paths_all, path_coverage_results, n_func_evals_lim, n_fitting_points, COVERAGE_METRIC_NAME, step_chkp=step_chkp)
            # subplot_metrics = [plot_array_igde]

            subplot_names = ["CID"]

            paths = write_metric_data_to_csv_igde(output_folder, algo_names=[algo_name_1, algo_name_2], metric_name="igde",plot_array=plot_array_igde)
            metric_data_loaded = retrieve_metric_data_from_csv(paths)
            make_comparison_plot(n_func_evals_lim, output_folder, metric_data_loaded, subplot_names, algo_names=[algo_name_1, algo_name_2], distance_tick=distance_tick, suffix="_ds")

            write_last_metric_values("IGDE",run_paths_all, path_coverage_results, COVERAGE_METRIC_NAME)
        ###############

        log.info("---- Analysis plots generated. ----")
        # write_analysis_results(result_runs_all, analysis_folder)
        log.info("---- Analysis summary written to file.")
        ######################

        log.info(f"Results written in: {output_folder}")

if __name__ == "__main__":        

    parser = argparse.ArgumentParser(description="Pass parameters for analysis.")
    parser.add_argument('-r', dest='n_runs', type=int, default=None, action='store',
                        help='Number runs to perform each algorithm for statistical analysis.')
    parser.add_argument('-p', dest='folder_runs',nargs="+", type=str, default=None, action='store',
                        help='The folder of the results written after executed runs of both algorithms. Path needs to end with "/".')
    parser.add_argument('-e', dest='exp_number', type=str, action='store',
                        help='Hardcoded example experiment to use.')
    parser.add_argument('-c', dest='path_metrics', type=str, default=None, action='store',
                        help='Path to csv file with metric results to regenerate comparison plot')
    
    args = parser.parse_args()
    
    ############# Set default experiment

    # we need to pass several exps, to be able to compare searches with different fitnessfnc 
    # (TODO check if fitness func should be part of a problem)
    # If the problem is the same, just pass the experiment number twice
        
    # exp_number_default =  1 # bnh
    exp_numbers_default = [103,105]  
     
    n_runs_default = 2
    n_func_evals_lim = 200 # this variable is require by an analysis function; TODO refactor 
    
    distance_tick = 20
    analyse_runs = n_runs_default

    # ref_point_hv = np.asarray([-0.6,0])
    ref_point_hv = None
    ###### Set parameters via flags

    if args.exp_number is None:
        exp_numbers = exp_numbers_default
    else:
        exp_numbers = [re.findall("[1-9]+", exp)[0] for exp in args.exp_number]

    if args.n_runs is None:
        n_runs = n_runs_default
    else:
        n_runs = args.n_runs

    
    folder_runs =  args.folder_runs
    path_metrics = args.path_metrics

    ##################### Override config

    config_1 = DefaultSearchConfiguration()
    config_1.population_size = 10
    config_1.n_generations = 20

    ####################

    config_2 = DefaultSearchConfiguration()
    config_2.population_size = 10
    config_2.n_generations = 20

    ###################
    problems = []

    for exp_n in exp_numbers:
        exp = experiment_switcher.get(int(exp_n))()
        problem = exp.problem
        problems.append(problem)
        
    analysis_name =  problem.problem_name
    ############### Specify the algorithm classes

    algo_class_1 = NSGAII_SIM
    algo_class_2 = PureSamplingRand

    algo_names = ["NSGA-II",
                 "RS"]
    

    # combined_analysis is written here
    output_folder = None #r"C:\\Users\\Lev\\Documents\\fortiss\\projects\\foceta\\SBT-research\\results\\analysis\\output_C1_C3\\"
    
    folder_runs = [
            r"C:\\Users\\sorokin\\Documents\\Projects\\Results\\analysis\\Demo_AVP_NSGAII_Diverse\\3_runs\\07-10-2023_22-08-18\\",
            r"C:\\Users\\sorokin\\Documents\\Projects\\SBT-research\\results\\analysis\\Demo_AVP_Metrics_Paper\\10_runs\\combined\\"       
    ]

    Analysis.run(
                analysis_name=analysis_name,
                algo_names = algo_names,
                class_algos = [algo_class_1, algo_class_2],
                configs = [config_1, config_2],
                n_runs = n_runs,
                problems = problems,
                n_func_evals_lim = n_func_evals_lim, 
                folder_runs = folder_runs,
                path_metrics = path_metrics,
                distance_tick=distance_tick,
                ref_point_hv=ref_point_hv,
                output_folder= output_folder
    )
