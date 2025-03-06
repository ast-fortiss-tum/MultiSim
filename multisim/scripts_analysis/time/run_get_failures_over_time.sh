#!/bin/bash

python -m scripts_analysis.time.get_failures_over_time \
  --paths "../msim-results/analysis/analysis_runs_20-12-2024_BD/" \
  --validation_folders "validation_combined_count-3_u" \
  --parent_folder "resumed" \
  --sim_map "msim"\
  --sim_names "bd" \
  --save_folder_all "./scripts_analysis/time/out/test_bd/" \
  --seeds 655 702 794 904 \
  --n_runs 7