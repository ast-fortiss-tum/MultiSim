import pandas as pd
import numpy as np
import math
import itertools

# calculates the average euclidian distance in the input space
def distance(point1, point2):
    return np.linalg.norm(point2 - point1)

def avg_euclid(points, 
               ub = None, 
               lb = None, 
               do_normalize = True):
    """
    Calculate the average pairwise difference between points in 3D space.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return 0
    # normalization
    if do_normalize:
        if ub is None:
            ub = np.max(points, axis = 0)
        else:
            ub = np.asarray(ub)
        
        if lb is None:
            lb = np.min(points, axis = 0)
        else:
            lb = np.asarray(lb)
        points = (points - lb) / (ub - lb)

    if len(points) <= 1:
        return 0
    pairwise_distances = []
    for pair in itertools.combinations(points, 2):
        pairwise_distances.append(distance(pair[0], pair[1]))
    avg_pairwise_difference = sum(pairwise_distances) / len(pairwise_distances)
    return avg_pairwise_difference

def min_distance(points):
    pairwise_distances = []
    for pair in itertools.combinations(points, 2):
        pairwise_distances.append(distance(pair[0], pair[1]))
    min_distance = min(pairwise_distances)
    return min_distance

if __name__ == "__main__":
    from opensbt.visualization import combined
    path = r"./results/single/Donkey_A7_0-360_XTE_gen20_pop20_seed4623425/NSGA-II/21-07-2024_11-44-44/all_critical_testcases.csv"
    pop = combined.read_pf_single(path)
    avg_difference = avg_euclid(pop.get("F"))
    print("Average Pairwise Difference (Critical):", avg_difference)
