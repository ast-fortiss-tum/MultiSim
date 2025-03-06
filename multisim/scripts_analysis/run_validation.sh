# Base directory
# base_dir="../results/analysis/analysis-5-runs_19-09-2024_BU"
base_dir="../msim-results/analysis/analysis_5-runs_12-12-2024_BU"
sim_folder="msim"

# Configurable parameters
# runs=(0 2 3 4)  # Run numbers
# seeds=(702 794 655 904 684)  # Corresponding seeds for each run
seeds=(348)  # Corresponding seeds for each run

runs=(1)  # Run numbers

config="VARSEG_MOO_A10_-180-180_XTE_DIST_BOTH"  # Configuration name
generations="gen20"
population="pop20"
algorithm="NSGAII-D"
subfolder="myfolder"

# Loop through each run and generate the appropriate folder paths
save_folders=()
for i in "${!runs[@]}"; do
    run=${runs[$i]}
    seed=${seeds[$i]}
    folder="$base_dir/run_$run/$sim_folder/${config}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}"
    save_folders+=("$folder")
done

percentage_validation=3

for save_folder in "${save_folders[@]}"
do
    python -m scripts_analysis.validation --n_repeat_validation 5 \
                    --save_folder $save_folder \
                    --percentage_validation $percentage_validation \
                    --folder_name_combined_prefix "validation_combined_count-${percentage_validation}_" \
                    --folder_validation_prefix "validation_count-${percentage_validation}_" \
                    --simulators "donkey" \
                    --only_failing_cells
done
