import argparse
import matplotlib
from pymoo.core.population import Population
from features.feature_map import FeatureMap
from mock.mock_simulation import MockSimulator
from scripts.evaluate_validity import sel_sim_fnc
import os
from pathlib import Path
from scripts.validate_results import get_inds_from_fmap, get_name_simulator, read_pf_single, validate_results, validate_results_from_pop
from config import CRITICAL_XTE, LOG_FILE, PERCENTAGE_VALIDATION_SSIM, STEP_VALIDATION, VALIDATION_THRESHOLD_RANGE
import logging as log
from opensbt.utils.files import find_file, find_file_contains

from config import EXPERIMENTAL_MODE
from scripts_analysis.validation_combined import validation_combined
from opensbt.utils.duplicates import duplicate_free
from opensbt.utils.log_utils import setup_logging
from scripts_analysis.select_part_folder import select_part_folder
logger = log.getLogger(__name__)

log.getLogger('engineio').setLevel(log.ERROR)
log.getLogger('socketio').setLevel(log.ERROR)

matplotlib.use("Agg")

setup_logging(LOG_FILE)

def run_validation(simulators, 
                   n_repeat_validation, 
                   write_results_extended, 
                   save_folder, 
                   percentage_validation = PERCENTAGE_VALIDATION_SSIM,
                   validate_portion = True,
                   folder_validation_prefix = "validation_",
                   file_name_results = "validation_results",
                   file_name_simout = "validation_simout",
                   folder_name_combined_prefix = "validation_combined_",
                   do_combined_validation = False,
                   only_failing_cells = True,
                   path_problem = None,
                   from_part_fmap = None,
                   adaptive = False):
        
    log.info("Starting flakiness analysis script.")
    print("save folder", save_folder)
    if path_problem is None:
        path_problem = find_file(save_folder, "problem")
    # path_testcases = save_folder + os.sep + "all_critical_testcases.csv"
    # path_testcases = find_file(save_folder + os.sep, "all_critical_testcases")

    # select intermediate feature map
    if from_part_fmap is not None:
        path_fmap = os.path.join(
            select_part_folder(os.path.join(save_folder,"feature_map/"), from_part_fmap)
            ,
             "fmap.json")
    else:
        path_fmap =  find_file_contains(save_folder + os.sep, ["fmap","seed"])
    print("path problem", path_problem)
    assert path_fmap is not None
    
    # assert path_testcases is not None
    print("path_fmap", path_fmap)
    print("save_folder", save_folder)

    # select the tests first to validate and then pass
    if validate_portion:
        fmap: FeatureMap = FeatureMap.from_json(path_fmap)
        individuals = get_inds_from_fmap(fmap, 
                                                percentage=percentage_validation,
                                                threshold=CRITICAL_XTE,
                                                only_failing_cells = only_failing_cells,
                                                adaptive = adaptive)

        pop_mix = Population(individuals)
        # read in idnviduals from feature map
    else:
        # TODO implement this option
        raise Exception()
        #pop_mix = read_pf_single(path_testcases)

    pop_mix = duplicate_free(pop_mix)

    # pass tests to validate results
    valid_files = []
    simout_files = []

    results = []
    # evaluate flakiness using multiple simulators
    for i, sim_fnc in enumerate(simulators):
        # folder_validation = f"{folder_validation_prefix}{i}"
        folder_validation =  f"{folder_validation_prefix}{get_name_simulator(sim_fnc).lower()}"
        res =  validate_results_from_pop(pop_mix=pop_mix,
                                        path_problem=path_problem,
                                        n_repeat= n_repeat_validation,
                                        save_folder_parent = save_folder,
                                        simulate_function = sim_fnc, # to select specific simulator. function in problem will be overriden
                                        problem_obj = None,
                                        folder_validation = folder_validation,
                                        write_results_extended = write_results_extended)

        valid_files.append(save_folder + os.sep + \
                                folder_validation + os.sep + f"{file_name_results}.json")
        simout_files.append(save_folder + os.sep + \
                                folder_validation + os.sep + f"{file_name_simout}.json")    
        results.append(res)

    # combine validation results
    if do_combined_validation:
        sims_suffix = ""
        for sim_fnc in simulators:
            sims_suffix += get_name_simulator(sim_fnc).lower()[0]

        folder_name_combined_prefix += sims_suffix

        save_folder_combined = save_folder + os.sep + folder_name_combined_prefix + os.sep
        Path(save_folder_combined).mkdir(exist_ok=True, parents=True)
        validation_combined(results,
                            path_problem, 
                            valid_files,
                            simout_files, 
                            save_folder_combined, 
                            thresholds_valid = VALIDATION_THRESHOLD_RANGE,
                            step = STEP_VALIDATION)

def validate_results_analysis(n_repeat_validation,
                write_results_extended, 
                save_folder,
                percentage_validation,
                folder_name_combined_prefix = "validation_combined_",
                folder_validation_prefix = "validation_",
                simulators=["donkey", "udacity"],
                do_combined_validation=False,
                only_failing_cells=True,
                from_part_fmap=None,
                adaptive = False):
    
    if EXPERIMENTAL_MODE:
        simulators = [
            MockSimulator.simulate,
            MockSimulator.simulate
        ]
    else:      
        simulators = [sel_sim_fnc(sim) for sim in simulators]
    print("simulators", simulators)

    run_validation(simulators = simulators, 
                n_repeat_validation = n_repeat_validation,
                write_results_extended = write_results_extended, 
                save_folder = save_folder,
                percentage_validation = percentage_validation,
                folder_name_combined_prefix=folder_name_combined_prefix,
                folder_validation_prefix=folder_validation_prefix,
                do_combined_validation=do_combined_validation,
                only_failing_cells=only_failing_cells,
                from_part_fmap=from_part_fmap,
                adaptive=adaptive)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate results and analyze them.")

    # Required arguments
    parser.add_argument("--n_repeat_validation", type=int, help="Number of repeat validations", required=True)
    parser.add_argument("--write_results_extended", action="store_true", default=False, help="Flag to determine if extended results should be written")
    parser.add_argument("--save_folder", type=str, help="Folder to save the results", required=True)
    parser.add_argument('--simulators', nargs="+", action="store", type=str, required="True", 
                        help='Simulator names to use for validation")')
    parser.add_argument("--percentage_validation", type=float, help="Percentage of data to use for validation", required=True)
    parser.add_argument('--folder_name_combined_prefix', type=str, default="validation_combined_", 
                        help='Folder prefix name for the combined results (default: "validation_combined_")')
    parser.add_argument('--folder_validation_prefix', type=str, default="validation_", 
                        help='Prefix for the validation folder (default: "validation_")')
    parser.add_argument('--do_combined',action="store_true",required=False, 
                        help='Do combined validation at the end or not.')
    parser.add_argument('--only_failing_cells',action="store_true",required=False, default=False,
                        help='Use only failing cells for validation.')
    parser.add_argument('--from_part_fmap', type=int, action="store",required=False, default=None,
                        help='Use only fmap from median of iterations.')
    parser.add_argument('--adaptive',action="store_true",required=False, default=False,
                        help='Use adaptive validation strategy.')
    
    args = parser.parse_args()

    validate_results_analysis(
        n_repeat_validation=args.n_repeat_validation,
        write_results_extended=args.write_results_extended,
        save_folder=args.save_folder,
        percentage_validation=args.percentage_validation,
        folder_name_combined_prefix=args.folder_name_combined_prefix,
        folder_validation_prefix=args.folder_validation_prefix,
        simulators=args.simulators,
        do_combined_validation=args.do_combined,
        only_failing_cells=args.only_failing_cells,
        from_part_fmap=args.from_part_fmap,
        adaptive=args.adaptive)