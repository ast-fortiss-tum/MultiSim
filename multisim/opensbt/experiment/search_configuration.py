from dataclasses import dataclass

@dataclass
class SearchConfiguration(object):
    n_generations = None 
    population_size = None
    maximal_execution_time = None # search time in "hh:mm:ss"
    num_offsprings = None
    prob_crossover = None
    eta_crossover = None
    prob_mutation = None
    eta_mutation = None

    # NSGAII-DT specific
    inner_num_gen = None
    max_tree_iterations = None

    # LEM-KNN specific
    n_replace = None

    # metrics
    ref_point_hv = None
    nadir = None
    ideal = None
    
    sampling = None
    mutation = None
    crossover = None

    n_repopulate_max = None

    archive_threshold = None

    seed = None

    write_subfolder_name = None

#TODO create a search configuration file specific for each algorithm
class DefaultSearchConfiguration(SearchConfiguration):
    n_func_evals_lim = 500
    n_generations = 5 
    population_size = 20
    maximal_execution_time = None # search time in "hh:mm:ss"
    num_offsprings = None
    prob_crossover = 0.7
    eta_crossover = 20
    prob_mutation = None
    eta_mutation = 15
    
    # NSGAII-DT specific
    inner_num_gen = 4
    max_tree_iterations = 4

    # LEM-KNN specific
    n_replace = 10

    # metrics
    ref_point_hv = None
    ideal = None
    nadir = ref_point_hv
    
    sampling = None
    mutation = None
    crossover = None

    n_repopulate_max = None

    archive_threshold = 5

    seed = None

    write_subfolder_name = None
