# imports related to OpenSBT
import numpy as np
import pymoo
from self_driving.agent_utils import calc_yaw_ego
from opensbt.simulation.simulator import Simulator, SimulationOutput
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from opensbt.model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

# all other imports
import config
from typing import List
import paramiko
import select

#HOSTNAME = '10.0.10.3'
#USERNAME = 'sorokin'
HOSTNAME = '192.168.178.62'
USERNAME = "lev"

# SCRIPT_PATH = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/udacity/udacity_simulator_remote_server_test.py"
SCRIPT_PATH = "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/sims/udacity/udacity_simulator_remote_server.py"
VENV_ACTIVATE_PATH = "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/venv/bin/activate"

import config

class UdacitySimulatorRemote(Simulator):
    # initial_pos = (125, 1.90000000, 1.8575, 8)
    initial_pos=(125.0, 0, -28.0, 8)
    
    @staticmethod
    def simulate(
        list_individuals,
        variable_names: List,
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        results = []
            
        for ind in list_individuals:
            try:
                output = execute_udacity_remote(hostname=HOSTNAME,
                                                username=USERNAME,
                                                angles=ind.tolist(),
                                                variable_names=variable_names,
                                                seg_length=config.SEG_LENGTH,
                                                script_path=SCRIPT_PATH
                                                )               
                result = SimulationOutput.from_json(output)
                results.append(result)
                print("[LOCAL] Remote call successfull")
            except Exception as e:
                print(f"[LOCAL] Received exception during simulation {e}")
                raise e
            finally:
                pass
                #print("Finished individual")
      
        return results

def execute_udacity_remote(hostname, 
                                  username, 
                                  script_path,
                                  angles,
                                  variable_names,
                                  seg_length,
                                  password = None):
    # Establish SSH connection
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if password is not None:
        client.connect(hostname, username=username, password = password)
    else:
        client.connect(hostname, username=username)

    angles_str = ",".join(map(str, angles))
    variable_names_str =  ",".join(map(str, variable_names))
    
    # cmd_string = f'python {script_path} "{angles_str}" "{variable_names_str}" {seg_length}'
    cmd_string = f'export DISPLAY=:1 \
                && source "{VENV_ACTIVATE_PATH}" \
                && python {script_path} "{angles_str}" "{variable_names_str}" {seg_length}'

    #cmd_string = f'python "{script_path}"'

    # print(f"Command str is: {cmd_string}")

    # Execute command remotely
    stdin, stdout, stderr = client.exec_command(cmd_string, get_pty=True)
    
    simout = ""
    is_simout = False
    for line in iter(stdout.readline, ""):
        print(line, end="")
        if "SIMOUT start" in line:
            is_simout = True
            continue
        if "SIMOUT end" in line:
            is_simout = False
            continue
        if  is_simout:
            simout = simout + line

    json_output= simout # the last line is considered as the return value of the script called
    # write_data(simout)

    # Wait for the script to finish
    exit_status = stdout.channel.recv_exit_status()
    
    # Close the SSH connection
    client.close()

    return json_output

if __name__ == "__main__":
    UdacitySimulatorRemote.simulate(
        list_individuals = [np.asarray([10,20,30,40,50, 10,20,10,20,10])],
        variable_names = ["angle1","angle2","angle3","angle4","angle5", \
                          "seg_length1","seg_length2","seg_length3","seg_length4","seg_length5"],
        scenario_path= "",
        sim_time = 10,
        time_step = 1,
        do_visualize = True)

