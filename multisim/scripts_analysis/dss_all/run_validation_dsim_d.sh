# Base directory
base_dir="../msim-results/analysis/analysis_runs_20-12-2024_D"

SIM="Donkey"

# Configurable parameters
runs=(0 1 2 3 4)  # Run numbers
# seeds=(702 794 655 904 684)  # Corresponding seeds for each run
 # Run numbers
# seeds=(702 794 904 684)  # Corresponding seeds for each run
#seeds=(760 348 859 29 120) # Corresponding seeds for each run
seeds=(760 348 859 29 120)

config="VARSEG_${SIM}_A10_-180-180_XTE_DIST"  # Configuration name
generations="gen20"
population="pop20"
algorithm="NSGAII-D"
subfolder="myfolder"

# Loop through each run and generate the appropriate folder paths
save_folders=()

for i in "${!runs[@]}"; do
    run=${runs[$i]}
    seed=${seeds[$i]}
    folder="$base_dir/run_$seed/sim_1/${config}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}/"
    save_folders+=("$folder")
done

percentage_validation=10000
n_repeat_validation=1

for save_folder in "${save_folders[@]}"
do
    python -m scripts_analysis.validation --n_repeat_validation "${n_repeat_validation}" \
                    --save_folder $save_folder \
                    --percentage_validation $percentage_validation \
                    --folder_name_combined_prefix "validation_quarter_all_combined_all_" \
                    --folder_validation_prefix "validation_quarter_all_" \
                    --simulators "beamng" \
                    --from_part_fmap  4
done
