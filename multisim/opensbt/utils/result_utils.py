import copy
from pymoo.core.population import Population
from pymoo.core.result import Result

from opensbt.utils.sorting import calc_nondominated_individuals, get_nondominated_population

def update_history(res, hist_holder, tree_iteration, inner_num_gen, inner_algorithm):
    for i in range(inner_num_gen):
        pop = Population.merge(
            hist_holder[tree_iteration * inner_num_gen + i].pop, res.history[i].pop)
        # copy a template of the inner algorithm, and then modify its population and other properties
        algo = copy.deepcopy(inner_algorithm)
        algo.pop = pop
        opt_pop = Population(
            individuals=calc_nondominated_individuals(pop))
        algo.opt = opt_pop
        hist_holder[tree_iteration * inner_num_gen + i] = algo

def create_result(problem, hist_holder, inner_algorithm, execution_time):
    # TODO calculate res.opt
    I = 0
    pop_archive = Population()
    for algo in hist_holder:
        # print(len(algo.pop))
        I += len(algo.pop)
        algo.evaluator.n_eval = I
        algo.start_time = 0
        algo.problem = problem
        algo.result()
        pop_archive = Population.merge(pop_archive,algo.pop)

    res_holder = Result()
    res_holder.algorithm = inner_algorithm
    res_holder.algorithm.evaluator.n_eval = I
    res_holder.problem = problem
    res_holder.algorithm.problem = problem
    res_holder.history = hist_holder
    res_holder.exec_time = execution_time
    res_holder.archive = pop_archive

    # calculate total optimal population using individuals from all iterations
    opt_all = Population()
    for algo in hist_holder:
        opt_all = Population.merge(opt_all, algo.pop)
    # log.info(f"opt_all: {opt_all}")
    opt_all_nds = get_nondominated_population(opt_all)
    res_holder.opt = opt_all_nds

    return res_holder

    # res_holder.opt = hist_holder[-1].opt