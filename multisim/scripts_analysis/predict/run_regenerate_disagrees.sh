#!/bin/bash

# Define common path pattern
BASE_PATH="../msim-results/analysis/analysis_runs_15-01-2025_BD_predict"
SEEDS=(859 1000)  # List of seed values

# Generate run paths dynamically
RUN_PATHS=()
for SEED in "${SEEDS[@]}"; do
    RUN_PATHS+=("$BASE_PATH/run_${SEED}/msim/PREDICT_SUR_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed${SEED}/NSGAII-D/myfolder")
done

# Call the Python script with multiple paths
python -m scripts_analysis.predict.regenerate_disagrees "${RUN_PATHS[@]}"

##########


# # Define common path pattern
# BASE_PATH="../msim-results/analysis/analysis_runs_13-02-2025_BD_predict"
# SEEDS=(28 684 701 702 858 903 904)  # List of seed values

# # Generate run paths dynamically
# RUN_PATHS=()
# for SEED in "${SEEDS[@]}"; do
#     RUN_PATHS+=("$BASE_PATH/run_${SEED}/msim/PREDICT_SUR_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed${SEED}/NSGAII-D/myfolder")
# done

# # Call the Python script with multiple paths
# python -m scripts_analysis.predict.regenerate_disagrees "${RUN_PATHS[@]}"

# ##########


# # Define common path pattern
# BASE_PATH="../msim-results/analysis/analysis_runs_09-02-2025_BD_predict"
# SEEDS=(28 110 112 114 858 903)  # List of seed values

# # Generate run paths dynamically
# RUN_PATHS=()
# for SEED in "${SEEDS[@]}"; do
#     RUN_PATHS+=("$BASE_PATH/run_${SEED}/msim/PREDICT_SUR_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed${SEED}/NSGAII-D/myfolder")
# done

# # Call the Python script with multiple paths
# python -m scripts_analysis.predict.regenerate_disagrees "${RUN_PATHS[@]}"