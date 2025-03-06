import math

from pymoo.core.infill import InfillCriterion

from operators.utils.validate_angle import max_angle_preserved


class MatingOpenSBT(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        
        print("MY MATING CALLED")

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, **kwargs)

        ########################## DEBUG
        # print("created offsprings:", off.get("X"))
        # from config import MAX_ANGLE

        # def angles_preserved_single_road(angles, max_angle):
        #     for i in range(1,len(angles)):
        #         preserved, dif = max_angle_preserved(angles[i-1], angles[i], max_angle = max_angle)
        #         if not preserved:
        #             return False, dif
        #     return True, dif
        
        # for road in off.get("X"):
        #     print(angles_preserved_single_road(road, max_angle=MAX_ANGLE))
        
        ########################### DEBUG
        
        return off



