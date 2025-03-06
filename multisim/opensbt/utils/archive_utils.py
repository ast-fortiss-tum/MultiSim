import numpy as np

def dist_archive(scenario,archive):
    distances = list()
    for sc in archive:
        # print("Digit or some close digit is in archive.")
        if scenario is not sc:
            dist = np.linalg.norm(sc.get("X") - scenario)
            distances.append(dist)
    if len(distances) == 0:
        min_dist = 0
    else:
        min_dist = min(distances)
    return min_dist