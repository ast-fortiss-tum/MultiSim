import pymoo

from opensbt.model_ga.individual import IndividualSimulated
from opensbt.quality_indicators.quality import Quality
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from opensbt.algorithm.nsga2_dt_sim import NSGAII_DT_SIM
from opensbt.evaluation.critical import CriticalBnhDivided
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.pymoo_test_problem import *
import matplotlib.pyplot as plt
import os
from visualization import output, output_temp, combined, configuration

problem = PymooTestProblem(
    'BNH', critical_function= CriticalBnhDivided())

config = DefaultSearchConfiguration()

config.population_size = 50
config.inner_num_gen = 2
config.prob_mutation = 0.5
config.n_func_evals_lim = 1000

optimizer = NSGAII_DT_SIM(problem,config)
res = optimizer.run()
optimizer.write_results()

all_pop = res.obtain_all_population()
crit_pop = all_pop.divide_critical_non_critical()[0]

eval_result = Quality.calculate_digd(res, crit_pop.get("X"))
n_evals, digd = eval_result.steps, eval_result.values

plt.figure(figsize=(7, 5))
plt.plot(n_evals, digd, color='black', lw=0.7)
plt.scatter(n_evals, digd, facecolor="none", edgecolor='black', marker='o')
plt.title("Design Space Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("dIGD")
plt.savefig(os.getcwd() + os.sep + "quality_indicators" + os.sep + "test.png")
plt.close()


output.digd_analysis(res, 
        save_folder=os.getcwd() + os.sep + "quality_indicators" + os.sep, 
        input_crit=crit_pop.get("X"), 
        filename='digd')