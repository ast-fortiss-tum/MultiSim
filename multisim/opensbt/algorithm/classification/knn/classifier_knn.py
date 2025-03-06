import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from opensbt.model_ga.problem import Problem
from opensbt.model_ga.population import Population
from mlxtend.plotting import plot_decision_regions
import logging as log

def classify_with_knn(population: Population, problem: Problem, save_folder: string):
    log.info("------ Performing classification of individuals with KNN ------- ")

    feature_names = problem.design_names
    X, y = population.get("X","CB")

    n_neighbors = 10

    if np.all(y):
        print("all label values are equal")
        return

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X, y)

    plt.figure(figsize=(14, 10))
    ax = plot_decision_regions(X,y.astype(np.int_),clf = knn_model, legend=0)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"Classification with kNN (nn = {n_neighbors})")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, 
            ['not critical', 'critical'], 
            framealpha=0.3, scatterpoints=1)
    save_file = save_folder + "knn_classification_" + feature_names[0] + '_' + feature_names[1] + ".png"

    plt.savefig(save_file)
    plt.clf()
    plt.close('all')
    
    return
