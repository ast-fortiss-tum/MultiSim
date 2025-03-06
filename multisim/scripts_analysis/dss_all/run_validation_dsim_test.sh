# Base directory
base_dir="./results/"

SIM=Mock
# Configurable parameters
# runs=(0 1 2 3 4)  # Run numbers
# seeds=(702 794 655 904 684)  # Corresponding seeds for each run

runs=(0)  # Run numbers
seeds=(2)  # Corresponding seeds for each run

config="VARSEG_${SIM}_A10_-180-180_XTE_DIST"  # Configuration name
generations="gen6"
population="pop6"
algorithm="NSGAII-D"
subfolder="temp"

# Loop through each run and generate the appropriate folder paths
save_folders=()

for i in "${!runs[@]}"; do
    run=${runs[$i]}
    seed=${seeds[$i]}
    folder="$base_dir/${config}_${generations}_${population}_seed${seed}/${algorithm}/${subfolder}"
    save_folders+=("$folder")
done

# save_folders=( ./results/VARSEG_Mock_A10_-180-180_XTE_DIST_gen6_pop6_seed2
# )

percentage_validation=10000
n_repeat_validation=1

for save_folder in "${save_folders[@]}"
do
    python -m scripts_analysis.validation --n_repeat_validation "${n_repeat_validation}" \
                    --save_folder $save_folder \
                    --percentage_validation $percentage_validation \
                    --folder_name_combined_prefix "validation_quarter_all_combined_all_" \
                    --folder_validation_prefix "validation_quarter_all_" \
                    --simulators "mock" \
                    --from_part_fmap  2

done
