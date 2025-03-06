from enum import Enum
import sys
from sims.donkey_simulation import DonkeySimulator
from sims.beamng_simulation import BeamngSimulator
from sims.udacity.udacity_simulator import UdacitySimulator
from mock.mock_simulation import MockSimulator
from sims.udacity.udacity_simulator_remote import UdacitySimulatorRemote

class SimulatorName(Enum):
    Mock = 1
    Udacity = 2
    Donkey = 3
    Beamng = 4
    Dummy = 5


def sel_sim_fnc(kwd: str):
    if kwd.lower() == "donkey":
        return  DonkeySimulator.simulate
    elif kwd.lower() == "udacity":
        return UdacitySimulator.simulate
    elif kwd.lower() == "udacity_remote":
        return UdacitySimulatorRemote.simulate
    elif kwd.lower() == "beamng":
        return BeamngSimulator.simulate
    elif kwd.lower() == "mock":
        return MockSimulator.simulate
    else:
        print("Simulator not known.")
        sys.exit(1)

def get_name_simulator(simulate_function):
    if simulate_function == DonkeySimulator.simulate:
        return "Donkey"
    elif simulate_function == UdacitySimulator.simulate or \
         simulate_function == UdacitySimulatorRemote.simulate:
        return "Udacity"
    elif simulate_function == BeamngSimulator.simulate:
        return "Beamng"
    elif simulate_function == MockSimulator.simulate:
        return "Mock"
    else:
        print("Simulator not known.")
        sys.exit(1)