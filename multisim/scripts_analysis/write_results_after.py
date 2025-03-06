
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
from pathlib import Path
import dill

from config import EXPERIMENTAL_MODE, WRITE_ALL_INDIVIDUALS
from features import opensbt_feature
from output import output_fmap, output_other
from visualization import output

def write_results(res, save_folder,
                  algorithm_name,
                  algorithm_parameters):
    
    output.write_calculation_properties(res,save_folder, algorithm_name,algorithm_parameters)
    output.optimal_individuals(res, save_folder)
    output.write_summary_results(res, save_folder)

    output.all_critical_individuals(res, save_folder)
    output.write_generations(res, save_folder)
    output.write_criticality_all_sims(res, save_folder)
    output.write_fitness_all_sims(res, save_folder)
    output.backup_problem(res, save_folder)

    if not EXPERIMENTAL_MODE:
        output.objective_space(res, save_folder)
        output.design_space(res, save_folder)

        output.plot_roads_overview_generation(res, save_folder)
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
    # output.calculate_n_crit_distinct(res,save_folder,
    #                                 bound_min=problem.xl,
    #                                 bound_max=problem.xu, 
    #                                 var="X")

    # output.calculate_n_crit_distinct(res,save_folder,
    #                                 bound_min=config.ideal,
    #                                 bound_max=config.nadir, 
    #                                 var="F")
            
    output.write_multisim_analysis(res, save_folder)
    output.write_config(save_folder)
    output_other.calculate_avg_euclid(res, save_folder, critical_only = True, var = "X")

    return save_folder

# load result objects
path_result = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_1-runs_06-09-2024_01-06-13_20-gen_2O-pop/run_0/msim/VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed704/NSGAII-D/06-09-2024_07-09-09/backup/result"

with open(path_result, "rb") as f:
    result = dill.load(f)

# write results
output_path = r"/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/analysis_1-runs_06-09-2024_01-06-13_20-gen_2O-pop/run_0/msim/VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed704/NSGAII-D/06-09-2024_07-09-09/generated/"
Path(output_path).mkdir(exist_ok=True, parents=True)

write_results(result, output_path, algorithm_name="NSGAII-D", algorithm_parameters={})
