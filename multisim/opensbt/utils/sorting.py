from pymoo.util.nds import efficient_non_dominated_sort
from pymoo.core.population import Population
import numpy as np

def calc_nondominated_individuals(population: Population):
    F = population.get("F")
    if len(F) == 0:
        return []
    best_inds_index = efficient_non_dominated_sort.efficient_non_dominated_sort(F)[0]
    best_inds = [population[i] for i in best_inds_index]
    return best_inds

def get_nondominated_population(population: Population):
    return Population(individuals=calc_nondominated_individuals(population))

def get_individuals_rankwise(population, number):
    ranks_pop = efficient_non_dominated_sort.efficient_non_dominated_sort(population.get("F"))
    # take individuals rankwise until limit reached
    inds_to_add = []
    for i in range(0,len(ranks_pop)):
        if len(inds_to_add) < number:
            remaining = number - len(inds_to_add)
            num_next_front = min(remaining,len(ranks_pop[i]))
            inds_to_add = np.concatenate([inds_to_add,ranks_pop[i][0:num_next_front]])
        else:
            break
    pop = Population(individuals = [population[int(i)] for i in inds_to_add])
    assert len(pop) == number
    return pop
