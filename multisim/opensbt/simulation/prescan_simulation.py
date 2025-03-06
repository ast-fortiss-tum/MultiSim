# from prescan_runner import runner
from opensbt.simulation.simulator import SimulationOutput, Simulator
import json
import time
import sys
import os
from typing import List
from opensbt.model_ga.individual import *
from pathlib import Path
import logging as log
import traceback

DEBUG = False
TIME_STEP = 1
DO_VISUALIZE = False
SIM_TIME = 10

OUTPUT_FILENAME = "results.csv"
TRACES_FILENAME = "trace_online.csv"
INPUT_FILENAME = "input.json"
EXP_EXECUTABLE = "Demo_AVP_cs"
PATH_KILL_SCRIPT = os.getcwd() + "\\..\\FOCETA\\experiments\\PrescanHangKill.bat"

class PrescanSimulator(Simulator):

    @staticmethod
    def simulate(list_individuals: List[Individual],
                 variable_names: List[str],
                 scenario_path: str,
                 sim_time: float = SIM_TIME,
                 time_step: float = TIME_STEP,
                 do_visualize: bool = DO_VISUALIZE):

        parent_dir = os.path.dirname(scenario_path)
        traces_path = os.path.join("", parent_dir + os.sep + TRACES_FILENAME)
        try:
            results = []
            for ind in list_individuals:
                # Write to input.json individual
                json_input = PrescanSimulator.get_ind_as_json(ind, variable_names)
                with open(parent_dir + os.sep + INPUT_FILENAME, "w") as outfile:
                    outfile.write(json.dumps(json_input))
                log.info(f"++ Prescan input file for experiment update created for scenario {ind} ++")
                
                # Make robust against simulation failures
                MAX_REPEAT = 10
                TIME_WAIT = 10 # in seconds

                do_repeat = True
                repeat_counter = 0
                while do_repeat and repeat_counter <= MAX_REPEAT:
                    try:
                        start_time_simulation = time.time()
                        # Call Prescan Runner
                        ouput_runner = runner.run_scenario(input_json_name=INPUT_FILENAME,
                                                        exp_file=scenario_path,
                                                        name_executable=EXP_EXECUTABLE,
                                                        sim_time=sim_time,
                                                        do_visualize=do_visualize,
                                                        output_filename=OUTPUT_FILENAME,
                                                        traces_filename=TRACES_FILENAME)   
                        
                        end_time_simulation = time.time()
                        # Succeeded
                        do_repeat = False

                        log.info(f"Simulation Time is: {end_time_simulation - start_time_simulation}")
                    except Exception as e:
                        log.info("Exception during simulation ocurred: ")
                        traceback.print_exc()
                        PrescanSimulator.kill()

                        time.sleep(TIME_WAIT)                
                        log.error(f"\n---- Repeating run for {repeat_counter}.time due to exception: ---- \n {e} \n")
                        repeat_counter += repeat_counter
                            
                simout = SimulationOutput.from_json(json.dumps(ouput_runner))
                results.append(simout)

                if DEBUG:
                    check_if_continue_by_user()

                # delete file where traces are stored from simulation
                PrescanSimulator.delete_traces(traces_path)
        except Exception as e:
            raise e
        finally:
            PrescanSimulator.delete_traces(traces_path)
        return results

    @staticmethod
    def kill():
        import subprocess
        filepath = PATH_KILL_SCRIPT
        p = subprocess.Popen(filepath, shell=True, stdout=subprocess.PIPE)

    ''' Examplary produced input.json:
            {   
                "HostVelGain": 1.4026142683114282,   //adresses egos'velocity
                "Other":                             //adresses parameters of the other actor
                    {   
                        "Velocity_mps": 2.7790308908407315, 
                        "Time_s": 0.4617562491805899, 
                        "Accel_mpss": 1.0067168680742395
                    }
            }
    '''

    @staticmethod
    def get_ind_as_json(ind, features):
        resJson = {
        }
        for i in range(len(features)):
            actor = features[i].split('_', 1)[0]
            featureName = features[i].split('_', 1)[1]
            if actor != "Ego":
                if actor not in resJson:
                    resJson[actor] = {}
                resJson[actor][featureName] = ind[i]
            else:
                resJson[featureName] = ind[i]

        if DEBUG:
            log.info(resJson)
        return resJson

    @staticmethod
    def delete_traces(path):
        if Path(path).exists():
            os.remove(path)
            log.info(f'Traces files trace_online.csv removed')


def check_if_continue_by_user():
    doContinue = input("Continue search? press Y for yes, else N.")
    log.info(f"++ Input was {doContinue}")
    if doContinue == 'N' or doContinue == 'n':
        log.info("Terminating search after user input")
        sys.exit()
    else:
        log.info("Continuing search")       
