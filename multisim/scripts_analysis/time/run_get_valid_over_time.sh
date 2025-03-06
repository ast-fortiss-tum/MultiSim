#!/bin/bash

python -m scripts_analysis.time.get_valid_over_time \
  --paths "../msim-results/analysis/analysis_runs_20-12-2024_B/" \
        "../msim-results/analysis/analysis_runs_20-12-2024_D" \
        "../msim-results/analysis/analysis_runs_01-01-2025_U" \
        "../msim-results/analysis/analysis_runs_20-12-2024_BD" \
        "../msim-results/analysis/analysis_runs_26-12-2024_BU" \
        "../msim-results/analysis/analysis_runs_08-01-2025_UD" \
  --validation_folders "validation_combined_count-3_ud" \
                        "validation_combined_count-3_ub" \
                        "validation_combined_count-3_bd" \
                        "validation_combined_count-3_u" \
                        "validation_count-3_combined_d" \
                        "validation_combined_count-3_b" \
  --sim_map "sim_1" "sim_1" "sim_1" "msim" "msim" "msim" \
  --sim_names "b" "d" "u" "bd" "bu" "ud" \
  --save_folder_all "./scripts_analysis/time/out/test/" \
  --seeds 702 794 655 904 684 760 348 859 29 120  \
  --n_runs 10
