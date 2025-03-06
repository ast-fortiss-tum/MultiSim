from pathlib import Path
from sklearn import tree
import numpy as np
import copy
import pymoo.core.population
import csv
import logging as log

MIN_SAMPLES_SPLIT = 0.05
MIN_SAMPLES_LEAF = 5
CRITICALITY_THRESHOLD_MIN = 0.5
CRITICALITY_THRESHOLD_MAX = 1   
DELTA = 0.0  # delta can be set negative to make regions overlap
MAX_TREE_DEPTH = 100
MIN_IMPURITY_DECREASE = 0.03

class Region:
    def __init__(self, xl, xu, population):
        self.xl = xl
        self.xu = xu
        self.population = population
        self.critical_share = None
        self.is_critical = None

    def define_critical(self, threshold_min, threshold_max):
        self.critical_share = sum(self.population.get("CB")) / len(self.population)
        self.is_critical = (self.critical_share > threshold_min) and (self.critical_share < threshold_max)
        return



def generate_critical_regions(population, 
                             problem,
                             min_samples_split=MIN_SAMPLES_SPLIT,
                             min_samples_leaf=MIN_SAMPLES_LEAF, 
                             max_depth=MAX_TREE_DEPTH, 
                             min_impurity_decrease=MIN_IMPURITY_DECREASE,
                             criticality_threshold_min=CRITICALITY_THRESHOLD_MIN,
                             criticality_threshold_max=CRITICALITY_THRESHOLD_MAX,
                             save_folder=None):

    feature_names = problem.design_names

    xl = problem.xl
    xu = problem.xu
    return generate_critical_regions_raw(population,
                                  feature_names, 
                             xl,
                             xu,
                             min_samples_split,
                             min_samples_leaf, 
                             max_depth, 
                             min_impurity_decrease,
                             criticality_threshold_min,
                             criticality_threshold_max,
                             save_folder)
        
def generate_critical_regions_raw(population,
                             feature_names, 
                             xl,
                             xu,
                             min_samples_split=MIN_SAMPLES_SPLIT,
                             min_samples_leaf=MIN_SAMPLES_LEAF, 
                             max_depth=MAX_TREE_DEPTH, 
                             min_impurity_decrease=MIN_IMPURITY_DECREASE,
                             criticality_threshold_min=CRITICALITY_THRESHOLD_MIN,
                             criticality_threshold_max=CRITICALITY_THRESHOLD_MAX,
                             save_folder=None):
    
    X, F, CB, SO = population.get("X", "F", "CB", "SO")

    regions_index = []
    regions = []
    critical_regions = []

    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf,
                                      criterion="entropy",
                                      max_depth=max_depth,
                                      min_impurity_decrease=min_impurity_decrease)
    clf = clf.fit(X, CB)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            regions_index.append(node_id)

    classification = clf.apply(X)

    def individuals_in_node(node_id):
        individuals = []
        for i in range(0, len(classification)):
            if classification[i] == node_id:
                individuals.append(i)
        return individuals

    node_indicator = clf.decision_path(X)
    leave_id = classification
    for region_index in regions_index:
        individuals_in_region = individuals_in_node(region_index)
        lowerReg = copy.deepcopy(xl)
        upperReg = copy.deepcopy(xu)
        sample_id = individuals_in_region[0]
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:
            if leave_id[sample_id] == node_id:  # <-- changed != to ==
                pass
                # continue # <-- comment out
                # log.info("leaf node {} reached, no decision here".format(leave_id[sample_id]))  # <--
            else:  # < -- added else to iterate through decision nodes
                if (X[sample_id][feature[node_id]] <= threshold[node_id]):
                    threshold_sign = "<="
                    j = node_id
                    # set upper bound
                    # in a DT nodes might have same threshold features
                    if threshold[j] < upperReg[feature[j]]:
                        upperReg[feature[j]] = threshold[j]

                else:
                    threshold_sign = ">"
                    j = node_id
                    # set lower bound
                    if threshold[j] > lowerReg[feature[j]]:
                        lowerReg[feature[j]] = threshold[j] + DELTA

        assert np.less(np.array(lowerReg), np.array(upperReg)).all

        temp_X = [X[ind] for ind in individuals_in_region]
        temp_CB = [CB[ind] for ind in individuals_in_region]
        temp_F = [F[ind] for ind in individuals_in_region]
        temp_SO = [SO[ind] for ind in individuals_in_region]
        temp_population = pymoo.core.population.Population.new("X", temp_X, "CB", temp_CB, "F", temp_F, "SO", temp_SO)
        regions.append(Region(lowerReg, upperReg, temp_population))

    regions_ordered = []
    for region in regions:
        region.define_critical(criticality_threshold_min, criticality_threshold_max)
        if region.is_critical:
            regions_ordered.append(region)
            critical_regions.append(region)
        else:
            regions_ordered.insert(0, region)
    regions = regions_ordered

    if save_folder is not None:
        Path(save_folder).mkdir(exist_ok=True, parents=True)
        clns = ["non-critical", "critical"]
        tree.plot_tree(clf)
        from matplotlib import pyplot as plt
        fig, _ = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
        tree.plot_tree(clf,
               feature_names = feature_names, 
               class_names=clns,
               filled = True,
               rounded=True)
        fig.savefig(save_folder + "tree.pdf")
        plt.close(fig)
        
        save_bounds_regions = save_folder + "bounds_regions.csv"
        with open(save_bounds_regions, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = ['region', 'lower_bound', 'upper_bound']
            writer.writerow(header)
            for i in range(len(critical_regions)):
                region = critical_regions[i]
                writer.writerow([f"region {str(i)}", str(region.xl), str(region.xu)])
            f.close()

        save_bounds_regions = save_folder + "bounds_regions.txt"
        with open(save_bounds_regions, 'w', encoding='UTF8', newline='') as f:
            for i in range(len(critical_regions)):
                text = f"region_{i}: "
                region = critical_regions[i]
                for j, lv in enumerate(region.xl):
                    text += f"({feature_names[j]} >= {lv} AND {feature_names[j]} <= {region.xu[j]})"
                    if j == len(region.xu) -1:
                        pass
                    else:
                        text += " AND\n" 
                f.write(text)
                f.write("\n\n")
            f.close()
    return regions
