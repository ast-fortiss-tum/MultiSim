import os
from opensbt.visualization.combined import make_comparison_plot, retrieve_metric_data_from_csv

subplot_names_1 = ["CID [gs-25]",
                  "CID[gs-20]",
                  "CID[gs-10]"]

subplot_names_2 = ["CID [gs-8000]",
                 "CID[fps-8000]"]

# subplot_names_3 = ["CID [20]",
#                  "FPS[8000]"]

algo_names = ["NSGA-II","RS"]

paths = [
         os.getcwd() + r"\scripts\data\final\gs\gs_25\combined_igde.csv",
          os.getcwd() +r"\scripts\data\final\gs\gs_20\combined_igde.csv",
          os.getcwd() +r"\scripts\data\final\gs\gs_10\combined_igde.csv",
          os.getcwd() +r"\scripts\data\final\fps\fps_1000\combined_igde.csv",
          os.getcwd() +r"\scripts\data\final\fps\fps_8000\combined_igde.csv"
         ]
print(paths)

analysis_folder = os.getcwd() + "\scripts\output\cid_comparison\\"

n_func_evals_lim = 2000
distance_tick = 250

metric_data_loaded = retrieve_metric_data_from_csv(paths)

# gs comparison plot

make_comparison_plot(n_func_evals_lim, analysis_folder, metric_data_loaded[0:3], subplot_names_1, algo_names, distance_tick=distance_tick, suffix="_gs_fps")
make_comparison_plot(n_func_evals_lim, analysis_folder, [metric_data_loaded[1], metric_data_loaded[4]] , subplot_names_2, algo_names, distance_tick=distance_tick, suffix="_gs")

# gs - fps comparison plot

