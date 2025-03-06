python -m scripts_analysis.pareto.generate_pareto_plot \
                            --folder_paths  "../msim-results/analysis/analysis_runs_20-12-2024_BD/" \
                                             "../msim-results/analysis/analysis_runs_26-12-2024_BU/" \
                                             "../msim-results/analysis/analysis_runs_08-01-2025_UD/" \
                                             "../msim-results/analysis/analysis_runs_20-12-2024_D/" \
                                             "../msim-results/analysis/analysis_runs_20-12-2024_B/" \
                                            "../msim-results/analysis/analysis_runs_01-01-2025_U/" \
                                            "../msim-results/analysis/analysis_runs_20-12-2024_D/" \
                                             "../msim-results/analysis/analysis_runs_20-12-2024_B/" \
                                            "../msim-results/analysis/analysis_runs_01-01-2025_U/" \
                            --combos "u" "d" "b" "ub" "ud" "bd" "u" "d" "b" "u" \
                            --sim_paths "msim" "msim" "msim" "sim_1" "sim_1" "sim_1" "sim_1" "sim_1" "sim_1"\
                            --sims "BD" "BU" "UD" "D" "B" "U" "db" "bu" "ud" \
                            --n_runs 10 \
                            --seeds  702 794 655 904 684 760 348 859 29 120  \
                            --save_folder "../msim-results/analysis/analysis_runs_01-2025/stacked_plot/" \
                            --is_dss 0 0 0 0 0 0 1 1 1
