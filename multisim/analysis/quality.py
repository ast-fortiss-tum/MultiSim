from ast import List

from numpy import NaN
from config import N_CELLS
from analysis import avg_euclid
from dataclasses import dataclass
import logging as log
from opensbt.quality_indicators.quality import EvaluationResult

class Quality(object):
    
    @staticmethod
    def calculate_avg_euclid(result, 
                                  critical_only = True,
                                  var = "X"):
        res = result
        hist = res.history
        if hist is not None:
            n_evals, hist_F = res.obtain_history_hitherto_archive(
                                                         critical=critical_only, 
                                                         var = var,
                                                         optimal=False)
            if var == "X":
                ub = result.problem.xu
                lb = result.problem.xl
            else:
                ub = None
                lb = None
            n_dist_crit =  [ 
                        avg_euclid.avg_euclid(_F, 
                                              ub = ub,
                                              lb = lb)
                            for _F in hist_F
                        ]
            return EvaluationResult(f"avg_euclid_{var}{'_C' if critical_only else ''}", n_evals, n_dist_crit)
        else:
            return None

if __name__ == "__main__":
    import dill
    path = r"/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/results/Mock_A7_-180-180_XTE_STEER_gen4_pop4_seed1341/NSGA-II/temp/backup/result"
    with open(path, 'rb') as f:
        result = dill.load(f)
    avg_difference = Quality.calculate_avg_euclid(result, critical_only=True, var = "X")
    print("Average Pairwise Difference (Critical):", avg_difference)
