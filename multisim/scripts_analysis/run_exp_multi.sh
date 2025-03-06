#!/bin/bash

# Predefined input parameters
EXPERIMENT_NUMBER=64         # Experiment number (64 / 1000)
N_REPEAT_VALIDATION=5       # Number of repeat validations
WRITE_SUBFOLDER_NAME="myfolder"  # Subfolder name for output
#SIMS_VALIDATION=("udacity" "donkey") # Simulator names for validation
SIMS_VALIDATION=() # Simulator names for validation

SIM="msim"                # msim / sim_1
#SIM="sim_1"                # msim / sim_1

SUFFIX="BD_predict"
#SUFFIX="B"

# List of predefined seeds
# SEEDS=(33 45)     
SEEDS=(702)     

# Define the analysis folder dynamically
ANALYSIS_FOLDER=$(pwd)/../msim-results/analysis/"analysis_runs_$(date '+%d-%m-%Y')_"$SUFFIX

# Create the analysis folder if it doesn't exist
mkdir -p "$ANALYSIS_FOLDER"

# Loop through each predefined seed and run the experiment
for SEED in "${SEEDS[@]}"
do
  # Define the path for the output folder (run_{seed}/msim)
  OUTPUT_FOLDER="$ANALYSIS_FOLDER/run_${SEED}/${SIM}/"

  # Create the output folder for each run
  mkdir -p "$OUTPUT_FOLDER"

  # Construct and run the Python command
  python run.py -e $EXPERIMENT_NUMBER -s $SEED \
    -o "$OUTPUT_FOLDER" \
    -v -rv $N_REPEAT_VALIDATION \
    -sims_validation "${SIMS_VALIDATION[*]}" \
    -sf "$WRITE_SUBFOLDER_NAME"
done