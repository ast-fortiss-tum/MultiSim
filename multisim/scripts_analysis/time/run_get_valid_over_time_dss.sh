#!/bin/bash
python -m scripts_analysis.time.get_valid_over_time_dss \
  --paths "../msim-results/analysis/analysis_runs_20-12-2024_B/" \
          "../msim-results/analysis/analysis_runs_20-12-2024_D/" \
          "../msim-results/analysis/analysis_runs_01-01-2025_U/" \
  --validation_folders "validation_all_qu_dsim_combined_count-3.0_bu_d" \
                       "validation_all_qu_dsim_combined_count-3.0_db_u" \
                       "validation_all_qu_dsim_combined_count-3.0_ud_b" \
  --paths_second_dss "../msim-results/analysis/analysis_runs_01-01-2025_U/" \
                     "../msim-results/analysis/analysis_runs_20-12-2024_B/" \
                     "../msim-results/analysis/analysis_runs_20-12-2024_D/" \
  --sim_map "sim_1" "sim_1" "sim_1" \
  --sim_names "bu" "bd" "ud" \
  --save_folder_all "../msim-results/analysis/analysis_runs_02-2025/" \
  --seeds 702 794 655 904 684 760 348 859 29 120 \
  --n_runs 10
