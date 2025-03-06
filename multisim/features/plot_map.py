
import os
import random
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_map_of_elites(
    data: Dict,
    filepath: str,
    iterations: int,
    x_axis_label: str,
    y_axis_label: str,
    min_value_cbar: float,
    max_value_cbar: float,
    occupation_map: bool = True,
    multiply_probabilities: bool = False,
    failure_probability: bool = False,
    quality_metric: str = None,
    quality_metric_merge: str = None,
    label_metric: str = None,
    title_suffix: str = "",
    title : str = "Feature Map",
    filename: str = None
) -> None:
    plt.clf()
    plt.cla()

    ser = pd.Series(list(data.values()), index=pd.MultiIndex.from_tuples(data.keys()))
    # df = ser.unstack().fillna(0)
    df = ser.unstack().fillna(np.inf)

    # figure
    fig, ax = plt.subplots(figsize=(40, 40))

    colors = ["red", "gold", "green"]
    if not occupation_map:
        if failure_probability:
            label = "Failure probability"
            colors = ["green", "gold", "red"]
        else:
            label = "Success probability"
    else:
        label = "Fitness"
    if label_metric is None:
        if quality_metric is not None:
            colors = ["green", "gold", "red"]
            label = quality_metric
    else:
      label = label_metric
    cmap = LinearSegmentedColormap.from_list(name="test", colors=colors)
    # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
    # cmap.set_under('-1.0')
    # Plot NaN in white
    cmap.set_bad(color="white")

    # I had to transpose because the axes were swapped in the original implementation
    df = df.transpose()
    # sns.set(font_scale=7)
    ax = sns.heatmap(
        df,
        cmap=cmap,
        vmin=min_value_cbar,
        vmax=max_value_cbar,
        # cbar_kws={'label': label}
    )

    ax.invert_yaxis()
    ax.figure.axes[-1].set_ylabel(label, fontsize=40, weight="bold")
    # ax.figure.axes[-1].set_yticklabels(ax.figure.axes[-1].get_ymajorticklabels(), fontsize=80, weight="bold")
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=40, weight="bold")
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=40, weight="bold")

    # if y_axis_label == CURVATURE_FEATURE_NAME:
    #     ax.set_yticklabels(
    #         [
    #             str(round(int(label.get_text()) / 100, 2)) if i % 2 == 0 else ""
    #             for i, label in enumerate(ax.get_ymajorticklabels())
    #         ],
    #         fontsize=80,
    #         rotation=0,
    #         weight="bold",
    #     )
    plt.xlabel(x_axis_label, fontsize=35, weight="bold")
    plt.ylabel(y_axis_label, fontsize=35, weight="bold")
    
    plt.title(title_suffix + "\n" + title, fontsize = 60)

    # get figure to save to file
    if filepath:
        ht_figure = ax.get_figure()
        if filename is None:
            filename = "heatmap_occupation_{}_{}_iterations_{}".format(x_axis_label, y_axis_label, iterations)
            if not occupation_map:
                if multiply_probabilities:
                    if failure_probability:
                        filename = "heatmap_failure_probability_multiply_{}_{}_iterations_{}".format(
                            x_axis_label, y_axis_label, iterations
                        )
                    else:
                        filename = "heatmap_success_probability_multiply_{}_{}_iterations_{}".format(
                            x_axis_label, y_axis_label, iterations
                        )
                else:
                    if failure_probability:
                        filename = "heatmap_failure_probability_{}_{}_iterations_{}".format(x_axis_label, y_axis_label, iterations)
                    else:
                        filename = "heatmap_success_probability_{}_{}_iterations_{}".format(x_axis_label, y_axis_label, iterations)

            if quality_metric is not None:
                if quality_metric_merge is not None:
                    filename = "heatmap_{}_{}_{}_{}_iterations_{}".format(
                        quality_metric, quality_metric_merge, x_axis_label, y_axis_label, iterations
                    )
                else:
                    filename = "heatmap_{}_{}_{}_iterations_{}".format(quality_metric, x_axis_label, y_axis_label, iterations)
        
        fig_name = os.path.join(filepath, filename)
        ht_figure.savefig(fig_name)
    
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    n_cells = 10
    size = 100
    failure_prob_x = [random.randint(0,n_cells -1) for _ in range(100)]
    failure_prob_y = [random.randint(0,n_cells -1) for _ in range(100)]
    failure_prob_f = [random.random() for _ in range(100)]

    failure_prob_data = {}
    # for i in range(size):
    #   failure_prob_data[(failure_prob_x[i], failure_prob_y[i])] = failure_prob_f[i]

    failure_prob_data[(0.1,1)] = 0.33
    failure_prob_data[(0.3,2)] = 0.77
    failure_prob_data[(0.2,2)] = 0.77

    iterations = 10
    x_axis_label = "x_label"
    y_axis_label = "y_label"
    min_value_cbar = 0
    max_value_cbar = 1

    CURVATURE_FEATURE_NAME = "Curvature"

    plot_map_of_elites(data=failure_prob_data,
                    filepath = "./features/",
                    iterations = 10,
                    x_axis_label = "x_label",
                    y_axis_label = "y_label",
                    min_value_cbar = 0,
                    max_value_cbar = 1
                    )
