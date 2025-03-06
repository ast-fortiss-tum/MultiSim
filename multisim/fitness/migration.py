from abc import ABC, abstractmethod
import numpy as np

class Migration(ABC):
    @abstractmethod
    def eval(self, 
            fitness_sims, 
            critical_sims, 
            fitness_function = None, 
            critical_function = None):
        pass

WORST_FIT_MIN = 100
WORST_FIT_MAX = 1  # When XTE is used, best XTE is 0

''' For diversified search '''

class CriticalAgreementDualMigrationDiverse(Migration):
    # if criticality values do not agree, worse this scenario as bad (to not regard that scenario)
    def eval(self, 
            fitness_sims, 
            critical_sims, 
            fitness_function):

        # assert (len(critical_sims) == len(fitness_sims) - 1 )

        critical_1 = critical_sims[0]
        critical_2 = critical_sims[1]

        fitness_1 = fitness_sims[0]
        fitness_2 = fitness_sims[1]

        if (critical_1 == critical_2):
            migrated = True
            return fitness_1, critical_1, migrated
        else:
            #assign bad fitness and non criticality
            bad_fitness = []
            critical = False
            migrated = True
            # print(f"[CritAgreeDualMigDiv] Critical values do not match in migration function.")
            # print(f"[CritAgreeDualMigDiv] fit_1/fit_2: {fitness_1} - {fitness_2}")
            # print(f"[CritAgreeDualMigDiv] crit_1/crit_2: {critical_1} - {critical_2}")

            for j in fitness_function.min_or_max:
                if j == 'min':
                    bad_fitness.append(WORST_FIT_MIN)
                else:   
                    bad_fitness.append(WORST_FIT_MAX)
            return bad_fitness, critical, migrated
        
class CriticalDisagreementDualMigrationDiverse(Migration):
    # if criticality values do not agree, worse this scenario as bad (to not regard that scenario)
    def eval(self, 
            fitness_sims, 
            critical_sims, 
            fitness_function):

        # assert (len(critical_sims) == len(fitness_sims) - 1 )

        critical_1 = critical_sims[0]
        critical_2 = critical_sims[1]

        fitness_1 = fitness_sims[0]
        fitness_2 = fitness_sims[1]

        if (critical_1 != critical_2):
            migrated = True
            return fitness_1, critical_1, migrated
        else:
            #assign bad fitness and non criticality
            bad_fitness = []
            critical = False
            migrated = True
            # print(f"[CritAgreeDualMigDiv] Critical values do not match in migration function.")
            # print(f"[CritAgreeDualMigDiv] fit_1/fit_2: {fitness_1} - {fitness_2}")
            # print(f"[CritAgreeDualMigDiv] crit_1/crit_2: {critical_1} - {critical_2}")

            for j in fitness_function.min_or_max:
                if j == 'min':
                    bad_fitness.append(WORST_FIT_MIN)
                else:   
                    bad_fitness.append(WORST_FIT_MAX)
            return bad_fitness, critical, migrated
        
class CriticalAgreementDualMigration(Migration):
    # if criticality values do not agree, worse this scenario as bad (to not regard that scenario)
    def eval(self, 
            fitness_sims, 
            critical_sims, 
            fitness_function):

        assert (len(fitness_sims) == 2)
        assert (len(fitness_sims) == len(critical_sims))

        critical_1 = critical_sims[0]
        critical_2 = critical_sims[1]

        fitness_1 = fitness_sims[0]
        fitness_2 = fitness_sims[1]

        if (critical_1 == critical_2):
            migrated = True
            return fitness_1, critical_1, migrated
        else:
            #assign bad fitness and non criticality
            bad_fitness = []
            critical = False
            migrated = True
            print("Critical values do not match in migration function.")
            print(f"fit_1/fit_2: {fitness_1} - {fitness_2}")
            print(f"crit_1/crit_2: {critical_1} - {critical_2}")

            for j in fitness_function.min_or_max:
                if j == 'min':
                    bad_fitness.append(WORST_FIT_MIN)
                else:   
                    bad_fitness.append(WORST_FIT_MAX)
            return bad_fitness, critical, migrated

class CriticalDisagreementDualMigration(Migration):
    # if criticality values do not agree, worse this scenario as bad (to not regard that scenario)
    def eval(self, 
            fitness_sims, 
            critical_sims, 
            fitness_function):

        assert (len(fitness_sims) == 2)
        assert (len(fitness_sims) == len(critical_sims))

        critical_1 = critical_sims[0]
        critical_2 = critical_sims[1]

        fitness_1 = fitness_sims[0]
        fitness_2 = fitness_sims[1]

        if (critical_1 != critical_2):
            migrated = True
            print("Critical values do not match in migration function.")
            print(f"fit_1/fit_2: {fitness_1} - {fitness_2}")
            print(f"crit_1/crit_2: {critical_1} - {critical_2}")
            fitness = abs(fitness_1 - fitness_2)
            critical = True
            return fitness, critical, migrated
        else:
            #assign bad fitness and non criticality
            bad_fitness = []
            critical = critical_1
            migrated = False
   
            for j in fitness_function.min_or_max:
                if j == 'min':
                    bad_fitness.append(WORST_FIT_MIN)
                else:   
                    bad_fitness.append(WORST_FIT_MAX)
            return bad_fitness, critical_1, migrated
             
# TODO create abstraction
class ConsensusMigration():
    def eval(self, fitness_sims, critical_sims, fitness_function = None):

        # print(f"[CM] fitness_sims: {fitness_sims}")
        # print(f"[CM] critical_sims: {critical_sims}")

        assert(len(critical_sims)%2 == 1)
        critical =  1 if np.count_nonzero(critical_sims) > len(critical_sims)/2 else 0
        index_critical = np.argwhere(critical_sims==critical)[0][0]

        #choose one fitness result based on the consensus criticality result
        fitness = fitness_sims[index_critical]

        return fitness, critical, None
    
# scenario is critical if at least on simulator output "critical"
class ConsensusMigrationWeak():
    def eval(self, fitness_sims, critical_sims, fitness_function = None):

        # print(f"[CM] fitness_sims: {fitness_sims}")
        # print(f"[CM] critical_sims: {critical_sims}")

        critical =  1 if np.count_nonzero(critical_sims) > 0 else 0

        index_critical_all = np.argwhere(critical_sims==critical)[0]

        #choose one fitness result based on the consensus criticality result
        #choose the worst fitness among the critical
        fitness = np.min(np.asarray([fitness_sims[ind] for ind in index_critical_all]))
        return fitness, critical, None