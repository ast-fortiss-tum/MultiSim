# Base directory
base_dir="../results/analysis/analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop"

SIM=Donkey

# Configurable parameters
runs=(0 1 2 3 4)  # Run numbers
seeds=(702 794 655 904 684)  # Corresponding seeds for each run
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
    folder="$base_dir/run_$run/sim_1/${config}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}/backup/resumed"
    save_folders+=("$folder")
done

percentage_validation=3
n_repeat_validation=1

for save_folder in "${save_folders[@]}"
do
    python -m scripts_analysis.validation --n_repeat_validation "${n_repeat_validation}" \
                    --save_folder $save_folder \
                    --percentage_validation $percentage_validation \
                    --folder_name_combined_prefix "validation_quarter_adaptive_combined_all_" \
                    --folder_validation_prefix "validation_quarter_adaptive_all_" \
                    --simulators "mock" \
                    --from_part_fmap  4 \
                    --adaptive
done
