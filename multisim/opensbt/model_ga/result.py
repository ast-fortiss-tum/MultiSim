import numpy as np
from opensbt.utils.sorting import get_nondominated_population

from pymoo.core.result import Result
from opensbt.model_ga.population import PopulationExtended as Population
from opensbt.model_ga.individual import IndividualSimulated as Individual
import dill
import os
from pathlib import Path

class ResultExtended(Result):

    def __init__(self) -> None:
        super().__init__()
    
    def obtain_history_design(self):
        hist = self.history
        
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_X = []  # the objective space values in each 
            pop = Population()
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations                            
                pop = Population.merge(pop, algo.pop)
                feas = np.where(pop.get("feasible"))[
                    0]  # filter out only the feasible and append and objective space values
                hist_X.append(pop.get("X")[feas])
        else:
            n_evals = None
            hist_X = None
        return n_evals, hist_X
    
    def obtain_history(self, critical=False):
        hist = self.history
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                opt = algo.opt  # retrieve the optimum from the algorithm
                if critical:
                    crit = np.where((opt.get("CB"))) [0] 
                    feas = np.where((opt.get("feasible"))) [0] 
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(opt.get("feasible"))[0]  # filter out only the feasible and append and objective space values
                hist_F.append(opt.get("F")[feas])
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F
    
    def get_first_critical(self):
            hist = self.history
            archive = self.obtain_archive()
            res = Population() 
            if hist is not None and archive is not None:
                for index, algo in enumerate(hist):
                    #n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                    inds = archive[:algo.evaluator.n_eval]
                    crit = np.where((inds.get("CB"))) [0] 
                    feas = np.where((inds.get("feasible"))) [0] 
                    feas = list(set(crit) & set(feas))
                    res = inds[feas]
                    if len(res) == 0:
                        continue
                    else:
                        return index, res
            return 0, res
    
    def obtain_archive(self):
        return self.archive
    
    def obtain_history_archive(self, critical=False, optimal = False, var = "X"):
        hist = self.history
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the "var" space values 
            n_eval_last = 0
            for i, algo in enumerate(hist):
                n_eval = algo.evaluator.n_eval - n_eval_last # get the number of evals for the current iteration
                n_evals.append(n_eval)  # store the number of function evaluations
                inds = self.archive[n_eval_last : algo.evaluator.n_eval]
                if critical:
                    crit = np.where((inds.get("CB"))) [0] 
                    feas = np.where((inds.get("feasible"))) [0] 
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(inds.get("feasible"))[0]  # filter out only the feasible and append and objective space values
                hist_F.append(inds.get(var)[feas])
                # update for next calculation
                n_eval_last = algo.evaluator.n_eval
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F
     
    def obtain_history_hitherto_archive(self,critical=False, optimal=True, var = "F"):
            hist = self.history
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation

            all = Population()
            for i, algo in enumerate(hist):
                n_eval = algo.evaluator.n_eval
                n_evals.append(n_eval)
                all = self.archive[:n_eval]
                if optimal:
                    all = get_nondominated_population(all)
                
                if critical:
                    crit = np.where((all.get("CB"))) [0] 
                    feas = np.where((all.get("feasible")))[0] 
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(all.get("feasible"))[0]  # filter out only the feasible and append and objective space values
                hist_F.append(all.get(var)[feas])
            return n_evals, hist_F
    
    def obtain_all_population(self):
        all_population = Population()
        hist = self.history
        for generation in hist:
            all_population = Population.merge(all_population, generation.pop)
        return all_population
    
    def obtain_history_hitherto(self,critical=False, optimal=True, var = "F"):
        hist = self.history
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation

        all = Population()
        for algo in hist:
            n_evals.append(algo.evaluator.n_eval)
            all = Population.merge(all, algo.pop)  
            if optimal:
                all = get_nondominated_population(all)
            
            if critical:
                crit = np.where((all.get("CB"))) [0] 
                feas = np.where((all.get("feasible")))[0] 
                feas = list(set(crit) & set(feas))
            else:
                feas = np.where(all.get("feasible"))[0]  # filter out only the feasible and append and objective space values
            hist_F.append(all.get(var)[feas])
        return n_evals, hist_F
    
    def persist(self, save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(save_folder + os.sep + "result", "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(save_folder, name="result"):
        with open(save_folder + os.sep + name, "rb") as f:
            return dill.load(f)
