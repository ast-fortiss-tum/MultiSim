# name generator for road generation problem using angles
from opensbt.model_ga.problem import ProblemExtended
from opensbt.problem.adas_multi_sim_problem import ADSMultiSimProblem
from opensbt.problem.adas_problem import ADASProblem
from sims.udacity.udacity_simulator import UdacitySimulator

def generate_problem_name(problem: ProblemExtended, 
                          category: str = None, 
                          fitness_name: str = None,
                          angle_sampling: bool = True,
                          prefix: str = None,
                          suffix: str = None):
    name = []
    if prefix is not None:
        name.append(prefix)
    # sim name
    if problem.__class__ is ADASProblem:
        if "udacity" in str(problem.simulate_function).lower():
            name.append("Udacity")
        elif "donkey" in str(problem.simulate_function).lower():
            name.append("Donkey")
        elif "beamng" in str(problem.simulate_function).lower():
            name.append("Beamng")
        elif "mock" in str(problem.simulate_function).lower():
            name.append("Mock")
        elif "dummy" in str(problem.simulate_function).lower():
            name.append("DummySimulator")
        else:
            print("Simulator name unknown")
            name.append("Simulator")
    elif problem.__class__ is ADSMultiSimProblem:
        name_combi = ""
        for simfnc in problem.simulate_functions:
            if "udacity" in str(simfnc).lower():
                name_combi += "U"
            elif "donkey" in str(simfnc).lower():
                name_combi += "D"
            elif "beamng" in str(simfnc).lower():
                name_combi += "B"
            elif "mock" in str(simfnc).lower():
                name_combi += "Mock"
            elif "dummy" in str(simfnc).lower():
                name_combi += "Dummy" 
            else:
                print("Simulator name unknown")
                name_combi += "Simulator"
        name.append(name_combi)

    # specific label
    if category is not None:
        name.append(category)

    # number angles
    name.append("A" + str(len(problem.xl)))

    # angle variation
    if angle_sampling:
        name.append(str(int(min(problem.xl))) + "-" + str(int(max(problem.xu))))
    if fitness_name is not None:
        name.append(fitness_name)
    else:
        name.append(problem.fitnes_function.name.replace(" ","_"))

    if suffix is not None:
        name.append(suffix)
    
    return "_".join(name)

if __name__ == "__main__":

    import pymoo
    from opensbt.model_ga.individual import IndividualSimulated
    from sims.udacity.udacity_simulator import UdacitySimulator
    pymoo.core.individual.Individual = IndividualSimulated

    from opensbt.model_ga.population import PopulationExtended
    pymoo.core.population.Population = PopulationExtended

    from opensbt.model_ga.result  import ResultExtended
    pymoo.core.result.Result = ResultExtended

    from opensbt.model_ga.problem import ProblemExtended
    pymoo.core.problem.Problem = ProblemExtended
    from opensbt.model_ga.individual import IndividualSimulated

    from sims.donkey_simulation import DonkeySimulator
    from fitness.fitness import MaxXTEFitness, MaxXTECriticality
    from config import NUM_CONTROL_NODES
    from opensbt.problem.adas_problem import ADASProblem
    #############

    n_cp = NUM_CONTROL_NODES

    problem = ADASProblem(
        problem_name=f"Udacity_Flakiness_A{n_cp-2}_XTE",
        scenario_path="",
        xl=[0]*(n_cp-2),
        xu=[360]*(n_cp-2),
        simulation_variables=[f"angle{i}" for i in range(1,n_cp - 1)],
        fitness_function=MaxXTEFitness(),
        critical_function=MaxXTECriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25
    )
    res = generate_problem_name(problem, fitness_name="XTE", category="Flakiness")
    
    print(f"name is: {res}")
    assert ( res == f"Udacity_Flakiness_A{n_cp-2}_XTE")
    
    ######################

    from fitness.migration import ConsensusMigration
    from sims.donkey_simulation import DonkeySimulator
    from fitness.fitness import MaxAvgXTEFitness
    from sims.udacity.udacity_simulator import UdacitySimulator

    problem = ADSMultiSimProblem(
                          problem_name="DUU_Consensus_A4_0-90_XTE_AVG",
                          scenario_path="",
                          xl=[0, 0, 0, 0],
                          xu=[90, 90, 90, 90],
                          simulation_variables=[
                            "angle1",
                            "angle2",
                            "angle3",
                            "angle4"],
                          fitness_function=MaxAvgXTEFitness(),
                          critical_function=MaxXTECriticality(),
                          simulate_functions = [
                            DonkeySimulator.simulate,
                            UdacitySimulator.simulate,
                            UdacitySimulator.simulate],
                          migrate_function=ConsensusMigration(),
                          simulation_time=10,
                          sampling_time=0.25
                          )
    res = generate_problem_name(problem, 
                                category="Consensus",
                                angle_sampling=True,
                                fitness_name="XTE_AVG")

    print(f"name is: {res}")
    assert ( res == "DUU_Consensus_A4_0-90_XTE_AVG")
        



    



