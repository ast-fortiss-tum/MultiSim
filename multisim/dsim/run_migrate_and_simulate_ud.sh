#!/bin/bash

# Base directory
base_dir1="../msim-results/analysis/analysis_runs_01-01-2025_U"
base_dir2="../msim-results/analysis/analysis_runs_20-12-2024_D"

SIM1="Udacity"
SIM2="Donkey"

sim_validate="Beamng"

# Configurable parameters
runs=(0 1 2 3 4 5 6 7 8 9)  # Run numbers
# seeds=(702 794 655 904 684)  # Corresponding seeds for each run
seeds=(702 794 655 904 684)

config1="VARSEG_${SIM1}_A10_-180-180_XTE_DIST"  # Configuration name
config2="VARSEG_${SIM2}_A10_-180-180_XTE_DIST"  # Configuration name

generations="gen20"
population="pop20"
algorithm="NSGAII-D"
subfolder="myfolder"

# Validation parameters
percentage_validation=3
n_repeat_validation=5

# Loop through each run and generate the appropriate folder paths
for i in "${!runs[@]}"; do
    run=${runs[$i]}
    seed=${seeds[$i]}

    # Define folder paths for the two simulators (sim_1 and sim_2) based on the run and seed
    base_sim_1="$base_dir1/run_$seed/sim_1/${config1}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}/"
    base_sim_2="$base_dir2/run_$seed/sim_1/${config2}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}/"

    # Define the save folder for the results (saving in sim_1 path)
    save_folder=$base_sim_1

    # Log the current run information (optional)
    echo "Running simulation for run $run with seed $seed..."
    echo "Base Sim 1: $base_sim_1"
    echo "Base Sim 2: $base_sim_2"
    echo "Save folder: $save_folder"

    # Call the Python script with the parsed arguments
    python -m dsim.migrate_and_simulate \
        --path_problem "$base_sim_1" \
        --base_sim_1 "$base_sim_1" \
        --base_sim_2 "$base_sim_2" \
        --save_folder "$save_folder" \
        --simulators_fmap "$SIM1" "$SIM2" \
        --simulator_validate "$sim_validate" \
        --percentage_validation $percentage_validation \
        --n_repeat_validation $n_repeat_validation \
        --do_validate 
done