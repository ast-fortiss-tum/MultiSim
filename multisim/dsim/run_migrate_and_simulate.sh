#!/bin/bash

# Define the base simulation directories
base_sim_1="./results/VARSEG_Mock_A10_-180-180_XTE_DIST_gen5_pop5_seed1/NSGAII-D/temp/"
base_sim_2="./results/VARSEG_Mock_A10_-180-180_XTE_DIST_gen5_pop5_seed2/NSGAII-D/temp/"

# Define the simulators used for feature mapping
sim_fmap_1="mock"
sim_fmap_2="mock"

sim_validate="mock"

# Define the save folder for the results
save_folder=$base_sim_1

# Call the Python script with the parsed arguments
python -m dsim.migrate_and_simulate \
    --path_problem "$path_problem" \
    --base_sim_1 "$base_sim_1" \
    --base_sim_2 "$base_sim_2" \
    --save_folder "$save_folder" \
    --simulators_fmap "$sim_fmap_1" "$sim_fmap_2" \
    --simulator_validate "$sim_validate" \
    --do_validate \
    --percentage_validation 3 \
    --n_repeat_validation 5 