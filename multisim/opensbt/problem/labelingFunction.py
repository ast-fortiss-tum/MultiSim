import random
import sys

import numpy as np

def label_none(x, out):
    """
    Labeling function assigning 'False' to all individuals.
    """

    n_ind = len(out.get("F"))
    labels = [False] * n_ind
    return np.array(labels)

def label_box(x, out, fl=None, fu=None):
    """
    Labeling function assigning 'True' to all individuals inside a predefined box.
    fl and fu are arrays indicating the lower and upper limits of the box.
    """

    if fl is None:
        raise Exception("Lower limit array is not given.")
        return label_none(x, out)

    if fu is None:
        raise Exception("Upper limit array is not given.")
        return label_none(x, out)

    F = out.get("F")

    if hasattr(F[0], '__len__'):
        n_obj = len(F[0])
    else:
        # single objective
        n_obj = 1


    if len(fl) != n_obj:
        raise Exception("Lower limit dimensionality is different from the number of objectives.")
        return label_none(x, out)

    if len(fu) != n_obj:
        raise Exception("Upper limit dimensionality is different from the number of objectives.")
        return label_none(x, out)

    labels = []
    for F_ind in F:
        label = np.all([F_ind > fl, F_ind < fu])
        labels.append(label)
    return np.array(labels)

def label_chessboard(x, out, fstart=None, fstep=None, sign=True):
    """
    Labeling function assigning 'True' to all individuals inside black pieces of the infinite chessboard.
    fstart is an array indicating the starting point of the chessboard.
    fstep is an array indicating the dimensions of each piece. The pieces are homogeneous.
    sign indicates whether the first quadrant of the chessboard is critical or not.
    """

    if fstart is None:
        raise Exception("Starting point is not given.")
        return label_none(x, out)

    if fstep is None:
        raise Exception("Step array is not given.")
        return label_none(x, out)

    if np.any(fstep <= [0]*len(fstep)):
        raise Exception("Step should be positive.")
        return label_none(x, out)


    F = out.get("F")

    if hasattr(F[0], '__len__'):
        n_obj = len(F[0])
    else:
        # single objective
        n_obj = 1

    if len(fstart) != n_obj:
        raise Exception("Starting point dimensionality is different from the number of objectives.")
        return label_none(x, out)

    if len(fstep) != n_obj:
        raise Exception("Step dimensionality is different from the number of objectives.")
        return label_none(x, out)

    labels = []
    for F_ind in F:
        label = sum(np.remainder(F_ind - fstart, np.array(fstep) * 2) < np.array(fstep)) % 2
        if sign:
            # if sign is True, the first quadrant is critical, otherwise it is not
            label = not label
        labels.append(label)
    return np.array(labels)

def label_directions(x, out, directions=None, directions_labels=None):
    """
    Labeling function assigning labels to individuals based on the closest direction vector.
    The label of an individual is the same as the label of the closest direction vector.
    directions and directions_labels indicate reference directions and their labels respectively.
    """

    if directions is None:
        raise Exception("Directions are not given.")
        return label_none(x, out)

    if directions_labels is None:
        raise Exception("Directions' labels are not given.")
        return label_none(x, out)

    F = out.get("F")

    if hasattr(F[0], '__len__'):
        n_obj = len(F[0])
    else:
        # single objective
        n_obj = 1

    if len(directions) != len(directions_labels):
        raise Exception("Number of directions is different from the number of labels.")
        return label_none(x, out)

    labels = []
    for F_ind in F:
        # normalise the vector
        F_vector = F_ind / np.linalg.norm(F_ind)
        max_scalar_prod = 0
        max_scalar_prod_index = None
        # find the direction, which has the smallest angle with the given vector
        for index, direction in enumerate(directions):
            if len(direction) != n_obj:
                raise Exception("Direction vector dimensionality is different from the number of objectives.")
                return label_none(x, out)

            # normalise the direction
            direction = np.array(direction / np.linalg.norm(direction))
            scalar_prod = np.dot(F_vector, direction)
            if scalar_prod > max_scalar_prod:
                max_scalar_prod = scalar_prod
                max_scalar_prod_index = index
        # assign the direction's label
        label = directions_labels[max_scalar_prod_index]
        labels.append(label)

    return np.array(labels)




