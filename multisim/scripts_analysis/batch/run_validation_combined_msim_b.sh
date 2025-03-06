#!/bin/bash

# Define the seeds for each run
SEEDS=(702 794 655 904 684)

# Define the base directory path for the analysis
#BASE_DIR="../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop"
BASE_DIR="../msim-results/analysis/analysis_5-runs_17-09-2024_DB"
# BASE_DIR="../results/analysis/analysis-5-runs_19-09-2024_BU"

# Define common variables
THRESHOLDS="0.5 1.0"
STEP_SIZE=0.1

SIM_PATH="sim_1"
SIM="Beamng"

#COMBO="ub"
COMBO="ud"
#COMBO="db"

# COMBO="u"
#COMBO="b"
#COMBO="d"

#################

# CSV_FILENAME="validation_combined_count-3_ub.csv"
# CSV_FILENAME="validation_combined_count-3_db.csv"
# CSV_FILENAME="validation_combined_count-3_ud.csv"
CSV_FILENAME="validation_combined_count-3_${COMBO}.csv"

# Loop through each run (0 to 4)
for ((run=0; run<5; run++)); do
    SEED="${SEEDS[$run]}"

    # Arrays for problem and parent folder paths for the current run
    PROBLEM_PATHS=(
        #"$BASE_DIR/run_${run}/${SIM}/VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed${SEED}/NSGAII-D/myfolder/backup/problem"
        "$BASE_DIR/run_${run}/${SIM_PATH}/VARSEG_${SIM}_A10_-180-180_XTE_DIST_gen20_pop20_seed${SEED}/NSGAII-D/myfolder/backup/problem"
    )

    PARENT_FOLDERS=(
        #"$BASE_DIR/run_${run}/msim/VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH_gen20_pop20_seed${SEED}/NSGAII-D/myfolder/"
        "$BASE_DIR/run_${run}/${SIM_PATH}/VARSEG_${SIM}_A10_-180-180_XTE_DIST_gen20_pop20_seed${SEED}/NSGAII-D/myfolder/"
    )

    # Loop through each problem-parent folder pair for the current run
    for ((j=0; j<${#PROBLEM_PATHS[@]}; j++)); do
        PROBLEM_PATH="${PROBLEM_PATHS[$j]}"
        PARENT_FOLDER="${PARENT_FOLDERS[$j]}"

        # Arrays for valid and simout file paths
        VALID_FILES_PATHS=(
            "${PARENT_FOLDER}validation_count-3_beamng/validation_results.json"
            "${PARENT_FOLDER}validation_count-3_donkey/validation_results.json"
            "${PARENT_FOLDER}validation_count-3_udacity/validation_results.json"
        )
        SIMOUT_FILES_PATHS=(
            "${PARENT_FOLDER}validation_count-3_beamng/validation_simout.json"
            "${PARENT_FOLDER}validation_count-3_donkey/validation_simout.json"
            "${PARENT_FOLDER}validation_count-3_udacity/validation_simout.json"
        )

        # Initialize empty strings for concatenating file paths
        VALID_FILES=""
        SIMOUT_FILES=""

        SAVE_FOLDER="${PARENT_FOLDER}validation_combined_count-3_${COMBO}/"

        # Concatenate file paths into space-separated strings
        for ((i=0; i<${#VALID_FILES_PATHS[@]}; i++)); do
            VALID_FILES="$VALID_FILES ${VALID_FILES_PATHS[$i]}"
            SIMOUT_FILES="$SIMOUT_FILES ${SIMOUT_FILES_PATHS[$i]}"
        done

        select_paths() {
            local combo=$1
            echo $combo
            case "$combo" in
                ub)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[0]}" "${VALID_FILES_PATHS[2]}")   # Beamng and Udacity
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[0]}" "${SIMOUT_FILES_PATHS[2]}") # Beamng and Udacity
                    ;;
                ud)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[1]}" "${VALID_FILES_PATHS[2]}")   # Donkey and Udacity
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[1]}" "${SIMOUT_FILES_PATHS[2]}") # Donkey and Udacity
                    ;;
                bd)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[0]}" "${VALID_FILES_PATHS[1]}")   # BeamNG and Donkey
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[0]}" "${SIMOUT_FILES_PATHS[1]}") # BeamNG and Donkey
                    ;;
                b)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[0]}")   # BeamNG only
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[0]}") # BeamNG only
                    ;;
                d)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[1]}")   # Donkey only
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[1]}") # Donkey only
                    ;;
                u)
                    SELECTED_VALID_FILES=("${VALID_FILES_PATHS[2]}")   # Udacity only
                    SELECTED_SIMOUT_FILES=("${SIMOUT_FILES_PATHS[2]}") # Udacity only
                    ;;
                *)
                    echo "Invalid combo value. Please use 'ub, 'ud', 'bd', 'b', 'd', or 'u'."
                    return 1
                    ;;
            esac
        }
        select_paths "$COMBO"
        echo $SELECTED_VALID_FILES

        # Call the Python script with the arguments for the current pair
        python -m scripts_analysis.validation_combined \
            --problem_path "$PROBLEM_PATH" \
            --valid_files $SELECTED_VALID_FILES \
            --simout_files $SELECTED_SIMOUT_FILES \
            --save_folder "$SAVE_FOLDER" \
            --thresholds_valid $THRESHOLDS \
            --step $STEP_SIZE \
            --csv_filename "$CSV_FILENAME"

        echo "Finished processing pair $j for run $run"
    done
done