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
from scripts.evaluate_validity import sel_sim_fnc
from scripts_analysis import validation
from opensbt.utils.files import find_file, find_file_contains
import dill

from features.feature_map import FeatureMap
from config import N_REPEAT_VALIDATION, PERCENTAGE_VALIDATION_SSIM, CRITICAL_XTE
import argparse
from scripts_analysis.select_part_folder import select_part_folder

# 1. first run validation with one 1 test per cell
# U -> D,B; D -> B,U; B -> U,D;

# HOME;
# Desktop: D-B, D-U 
# Server: B -> U,D

# 2. MERGE 
# D,B; D,U; B,U;

# 3. Run simulator
# U, B, D

def dsim_migrate_union_validate(path_problem,
                                base_sim_1,
                                base_sim_2,
                                save_folder,
                                simulators_fmap,
                                simulator_validate,
                                do_validate = True,
                                percentage_validation = PERCENTAGE_VALIDATION_SSIM,   
                                n_repeat_validation = N_REPEAT_VALIDATION,
                                prefix_validation_folder = "validation_quarter_all",
                                identifier = "all_qu"                           
                                ):
    
    simulators_fmap = [s.lower() for s in simulators_fmap]

    path_problem = find_file(base_sim_1, "problem")

    fmap_folder_1 =  select_part_folder(base_sim_1 + os.sep + "feature_map", part = 4)

    path_fmap_ds_1_1 = find_file_contains(fmap_folder_1, ["fmap"], ends_with="json")
    path_fmap_ds_1_2 = find_file_contains(base_sim_1 + os.sep + f"{prefix_validation_folder}_{simulators_fmap[1]}/", 
                                          ["fmap"],
                                          ends_with = "json")
    
    fmap_folder_2 =  select_part_folder(base_sim_2 + os.sep + "feature_map", part = 4)

    path_fmap_ds_2_1 = find_file_contains(fmap_folder_2, ["fmap"], ends_with="json")
    path_fmap_ds_2_2 = find_file_contains(base_sim_2 + os.sep + f"{prefix_validation_folder}_{simulators_fmap[0]}/",
                                        ["fmap"],
                                          ends_with = "json")
                
    Path(save_folder).mkdir(parents=True,exist_ok=True)

    print(path_fmap_ds_1_1)
    print(path_fmap_ds_1_2)
    print(path_fmap_ds_2_1)
    print(path_fmap_ds_2_2)
            
    # with open(path_problem, 'rb') as f:
    #     problem = dill.load(f)
    #     # print("Loaded Problem:", problem)

    fmap_ds_1_1 = FeatureMap.from_json(path_fmap_ds_1_1)
    fmap_ds_1_2 = FeatureMap.from_json(path_fmap_ds_1_2)      
    
    fmap_ds_2_1 = FeatureMap.from_json(path_fmap_ds_2_1)
    fmap_ds_2_2 = FeatureMap.from_json(path_fmap_ds_2_2)
    
    # migrate
    fmap_ds_1 = FeatureMap.migrate_equal(fmap_ds_1_1,fmap_ds_1_2)
    fmap_ds_2 = FeatureMap.migrate_equal(fmap_ds_2_1,fmap_ds_2_2)
    
    path_folder_ds_1 = os.path.join(save_folder,f"fmap_{identifier}_{simulators_fmap[0]}_migrated_in_{simulators_fmap[1]}")
    path_folder_ds_2 = os.path.join(save_folder,f"fmap_{identifier}_{simulators_fmap[1]}_migrated_in_{simulators_fmap[0]}")

    Path(path_folder_ds_1).mkdir(parents=True,exist_ok=True)
    Path(path_folder_ds_2).mkdir(parents=True,exist_ok=True)

    fmap_ds_1.export_to_json(filename=path_folder_ds_1 + os.sep +f"fmap_{identifier}_{simulators_fmap[0]}_migrated_in_{simulators_fmap[1]}.json")
    fmap_ds_2.export_to_json(filename=path_folder_ds_2 + os.sep + f"fmap_{identifier}_{simulators_fmap[1]}_migrated_in_{simulators_fmap[0]}.json")

    fmap_ds_1.plot_map(path_folder_ds_1)
    fmap_ds_2.plot_map(path_folder_ds_2)

    # write original maps
    path_folder_origin_sim_1 = os.path.join(path_folder_ds_1, "original", simulators_fmap[0])
    path_folder_origin_sim_2 = os.path.join(path_folder_ds_2, "original", simulators_fmap[1])
    
    Path(path_folder_origin_sim_1).mkdir(parents=True,exist_ok=True)
    Path(path_folder_origin_sim_2).mkdir(parents=True,exist_ok=True)

    fmap_ds_1_1.plot_map(path_folder_origin_sim_1)
    fmap_ds_1_2.plot_map(path_folder_origin_sim_2)
    
    fmap_ds_1_1.export_to_json(path_folder_origin_sim_1 + os.sep + "fmap.json")
    fmap_ds_1_2.export_to_json(path_folder_origin_sim_2 + os.sep + "fmap.json")

    sims_short = "".join([sim[0] for sim in simulators_fmap])

    path_folder_ds_final = os.path.join(save_folder,f"fmap_{identifier}_union_{sims_short}")
    Path(path_folder_ds_final).mkdir(parents=True,exist_ok=True)

    fmap_final = FeatureMap.migrate(fmap_ds_1, fmap_ds_2)

    fmap_final.export_to_json(filename=path_folder_ds_final + os.sep + f"fmap_{identifier}_union_{sims_short}_seedX.json")
    fmap_final.plot_map(path_folder_ds_final)

    if do_validate:
        validation.run_validation( 
                                simulators = [sel_sim_fnc(simulator_validate)], 
                                n_repeat_validation = n_repeat_validation,
                                write_results_extended = False,
                                save_folder = path_folder_ds_final,
                                percentage_validation=percentage_validation,
                                folder_name_combined_prefix=f"validation_{identifier}_dsim_combined_count-{percentage_validation}_{sims_short}_",
                                folder_validation_prefix=f"validation_{identifier}_dsim_count-{percentage_validation}_{sims_short}_",
                                do_combined_validation=True,
                                only_failing_cells=True,
                                path_problem=path_problem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate and validate feature maps for different simulators.")
    
    parser.add_argument("--path_problem", type=str, help="Path to the problem file.", required=True)
    parser.add_argument("--base_sim_1", type=str, help="Path to the base folder for simulator 1.", required=True)
    parser.add_argument("--base_sim_2", type=str, help="Path to the base folder for simulator 2.", required=True)
    parser.add_argument("--save_folder", type=str, help="Folder to save the output.",  required=True)
    parser.add_argument("--simulators_fmap", nargs=2, type=str, help="Names of the two simulators for feature mapping.", required=True)
    parser.add_argument("--simulator_validate", type=str, help="Name of the simulator used for validation.", required=True)
    
    parser.add_argument("--do_validate", action="store_true", default=False,
                        help="Whether to run validation after feature map migration. Default is True.")
    parser.add_argument("--percentage_validation", type=float, default=3, 
                        help="Percentage of the data to be used for validation. Default is 3.")
    parser.add_argument("--n_repeat_validation", type=int, default=5, 
                        help="Number of times validation should be repeated. Default is 5.")

    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    dsim_migrate_union_validate(
        path_problem=args.path_problem,
        base_sim_1=args.base_sim_1,
        base_sim_2=args.base_sim_2,
        save_folder=args.save_folder,
        simulators_fmap=args.simulators_fmap,
        simulator_validate=args.simulator_validate,
        do_validate=args.do_validate,
        percentage_validation=args.percentage_validation,
        n_repeat_validation=args.n_repeat_validation)