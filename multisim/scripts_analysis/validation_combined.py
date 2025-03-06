import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from features import opensbt_feature
from features.feature_map import FeatureMap
from opensbt.model_ga.individual import IndividualSimulated
from opensbt.model_ga.population import PopulationExtended
from opensbt.utils.encoder_utils import NumpyEncoder
from output import output_other
import logging as log
import dill
from pathlib import Path

from opensbt.simulation.simulator import SimulationOutput
from opensbt.utils.archive import MemoryArchive

# readin validation results from different simulators and calculate overall validity score
def validation_combined(results, 
                        problem_path,
                        valid_files, 
                        simout_files,
                        save_folder, 
                        thresholds_valid, 
                        step = 0.1,
                        csv_file_name = 'validation_combined.csv'):
    
    # assert results is not None or problem_path is not None
    print(valid_files)
    metric_names = ['valid_rate', 'n_valid', 'fmap_cov', 'fmap_s', 'fmap_fs', 'avg_euclid']
    header = ['threshold'] + metric_names

    Path(save_folder).mkdir(parents=True, exist_ok=True)

    with open(save_folder + os.sep + csv_file_name, 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        write_to.writerow(header)

        for threshold_valid in np.arange(thresholds_valid[0], thresholds_valid[1] + step, step):
            
            if threshold_valid > 1:
                continue

            overall_rate = []
            crit_rates  = []
            fitness_all = []
            data_sims = []
            simout_sims = []

            threshold_valid = np.round(threshold_valid,3)

            new_entry = [threshold_valid]

            for file in valid_files:
                with open(file, 'r') as file:
                    data = json.load(file)
                a = np.asarray(data["critical_ratio"])
                f = np.asarray(data["min_fitness"])
                crit_rates.append(a)
                fitness_all.append(f)
                data_sims.append(data)
            
            for file in simout_files:
                with open(file, 'r') as file:
                    data_so = json.load(file)
                so = np.asarray(data_so["simout_roads"])
                simout_sims.append(so)

            overall_rate = np.asarray(crit_rates).min(axis=0)
            index_simulator_min = np.asarray(crit_rates).argmin(axis=0)
            # simout_sims = np.asarray(simout_sims)

            num_valid = 0
            for r in overall_rate:
                if r >= threshold_valid:
                    num_valid += 1

            index_valid = np.where(overall_rate >= threshold_valid)[0]
            # print(f"index_valid: {index_valid}")

            valid_rate = num_valid / len(overall_rate) if len(overall_rate) > 0 else 0            # print(f"valid_rate: {'%.2f' % valid_rate}")
            
            #############
            
            # add to csv file

            new_entry.append(valid_rate)
            new_entry.append(num_valid)
            
            ################

            if len(index_valid) > 0:
                index_valid = np.array([int(i) for i in index_valid], dtype=int)
                angles_valid = np.asarray(data["roads_angles"])[index_valid]
                fitness = np.asarray(fitness_all).min(axis=0)[index_valid]
            else:
                angles_valid = np.asarray([])
                fitness = np.asarray([])

            ######### apply other metrics
            
            # create result object containing all valid roads in population
            # create new results object from given results objects
            # load problem or use results object

            if problem_path:
                with open(problem_path, 'rb') as f:
                    problem = dill.load(f)
            else:
                problem = results[0].problem
            
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from opensbt.utils.result_utils import create_result

            # create some algorithm instance for storage
            hist_holder = []

            inds = []
            for id,i in enumerate(index_valid):
                # get all roads with Fitness and Simout information to apply later further metrics
                ind = IndividualSimulated()
                ind.set("X", angles_valid[id])
                ind.set("F", fitness[id])

                data_sim = data_sims[index_simulator_min[i]]
                sim_roads = simout_sims[index_simulator_min[i]][i]
            
                index_min_fitness_road =  data_sim["ind_min_fitness"][i]

                simout = SimulationOutput.from_json(json.dumps(sim_roads[index_min_fitness_road]))

                ind.set("SO", simout)
                ind.set("CB", True)
                inds.append(ind)
            
            pop = PopulationExtended(individuals=inds)

            inner_algorithm = NSGA2(
                    pop_size=None,
                    n_offsprings=None,
                    sampling=None,
                    eliminate_duplicates=True,
                    archive = MemoryArchive())
            inner_algorithm.pop = pop
            inner_algorithm.archive = pop
            hist_holder.extend([inner_algorithm])

            res = create_result(hist_holder=hist_holder,
                        inner_algorithm=inner_algorithm,
                        execution_time=-1,
                        problem=problem)

            ############# apply metrics
            Path(save_folder).mkdir(exist_ok=True, parents=True)

            map: FeatureMap = opensbt_feature.get_feature_map_from_pop(pop = pop,
                                                max_fitness=0, 
                                                min_fitness=-3)
            # plot only if above 0.5
            if float(threshold_valid) == 0.50 or float(threshold_valid) == 0.6 or \
                    float(threshold_valid) == 0.8 or float(threshold_valid) == 1.0:
                map.plot_map(filepath=save_folder,
                            filename = f"fmap_valid_combined_{threshold_valid}.png",
                            title_suffix = f"{threshold_valid}_",
                            iterations= -1)
                
                map.export_to_json(save_folder + f"{threshold_valid}_" + "fmap.json")
                
            fm_fail_sp = map.get_failure_sparsness()
            fm_cov = map.get_fm_coverage()
            fm_sp = map.get_fm_sparseness()

            new_entry.append(fm_cov)
            new_entry.append(fm_sp)
            new_entry.append(fm_fail_sp)
            
            _, avg_euclids = output_other.calculate_avg_euclid(res, 
                                            save_folder, 
                                            critical_only = True, 
                                            var = "X",
                                            title_suffix = f"_{threshold_valid}")
            # take last one
            avg_euclid = avg_euclids[-1]

            new_entry.append(avg_euclid)

            ########################

            report = dict(threshold_critical = threshold_valid,
                        n_valid = len(angles_valid),
                        n_total = len(overall_rate),
                        valid_rate = valid_rate,
                        crit_rates = crit_rates,
                        overall_rate = overall_rate,
                        angles_valid = angles_valid,
                        fitness = fitness,
                        fm_fail_sp = fm_fail_sp,
                        fm_cov = fm_cov,
                        fm_sp = fm_sp,
                        avg_euclid = avg_euclid)
            
            with open(save_folder + os.sep + f"validation_{threshold_valid}.json", 'w') as json_file:
                    json.dump(report, json_file, indent=4,cls=NumpyEncoder)
        
            write_to.writerow(new_entry)

    # After the file is closed, reopen it for reading
    with open(save_folder + os.sep + csv_file_name, 'r', encoding='UTF8') as f:
        df = pd.read_csv(f)
    
    plot_folder = save_folder + os.sep + "plots" + os.sep
    Path(plot_folder).mkdir(parents = True, exist_ok = True)

    for metric in metric_names:
        plot_validation_combined(df,
                                    name_metric=metric,
                                    save_folder=plot_folder) 
        
    f.close()


def plot_validation_combined(df,
                             name_metric, 
                             save_folder,
                             fname = None):
    plt.figure(figsize=(10, 6))
    plt.plot(df["threshold"].tolist(), df[name_metric].tolist(), marker='o', label=name_metric)
    
    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel(name_metric)
    plt.title(f'Plot of {name_metric} against Threshold')
    plt.legend()
    plt.grid(True)

    if fname is None:
        fname = f"valid_combined_{name_metric}"
    plt.savefig(save_folder + os.sep + fname + ".png")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run validation combined process.")
    parser.add_argument('--results', type=str, required=False, default=None, help='Path to the results file.')
    parser.add_argument('--problem_path', type=str, required=True, help='Path to the problem directory.')
    parser.add_argument('--valid_files', nargs='+', type=str, required=True, help='Path to the valid files.')
    parser.add_argument('--simout_files',nargs='+', type=str, required=True, help='Path to the simout files.')
    parser.add_argument('--save_folder', type=str, required=True, help='Directory where outputs will be saved.')
    parser.add_argument('--thresholds_valid', type=float, nargs='+', default=[0.5, 1.0], help='List of thresholds for validation, default is [0.5, 1.0].')
    parser.add_argument('--step', type=float, default=0.25, help='Step size for the validation process, default is 0.25.')
    parser.add_argument('--csv_filename', type=str, help='CSV filename.')

    args = parser.parse_args()

    validation_combined(
        results=args.results,
        problem_path=args.problem_path,
        valid_files=args.valid_files,
        simout_files=args.simout_files,
        save_folder=args.save_folder,
        thresholds_valid=args.thresholds_valid,
        step=args.step,
        csv_file_name=args.csv_filename
    )

    # path = "./analysis_1-runs_02-09-2024_20-23-24_10-gen_10-pop/run_0/sim_1/"
    
    # valid_files = [
    #     f"{path}/validation_0/validation_results.json",
    #     f"{path}/validation_1/validation_results.json"
    # ]

    # simout_files = [
    #     f"{path}/validation_0/validation_simout.json",
    #     f"{path}/validation_1/validation_simout.json"
    # ]

    # save_folder = f"./{path}/validation_combined/"
    # problem_path = f"./{path}/backup/problem"
    
    # res = None

    # Path(save_folder).mkdir(exist_ok=True, parents=True)

    # validation_combined(results = res,
    #                     problem_path = problem_path,
    #                     valid_files = valid_files, 
    #                     simout_files= simout_files, 
    #                     save_folder = save_folder,
    #                     thresholds_valid = [0.5,1.0],
    #                     step=0.25)