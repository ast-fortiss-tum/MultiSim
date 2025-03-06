import os
import socket
import sys

sim_map = {
    "for1789lnx" : {
        "DONKEY_EXE_PATH" : "/home/sorokin/Downloads/donkeysim-maxibon-linux/donkeysim-maxibon-linux.x86_64",
        "UDACITY_EXE_PATH" :"/home/sorokin/Downloads/udacitysim-maxibon-linux/udacitysim-maxibon-linux.x86_64",
        "BEAMNG_USER" : None,
        "BEAMNG_HOME" : None
    },
    "r5496229" : {    
        "DONKEY_EXE_PATH" : "C:\\Users\\sorokin\\Downloads\\donkeysim-maxibon-win64\\donkey_sim.exe",
        "UDACITY_EXE_PATH" : r"C:\Users\sorokin\Downloads\udacitysim-maxibon-win64\\self_driving_car_nanodegree_program.exe",
        "BEAMNG_USER" : r"C:\\Users\sorokin\\Documents\\BeamNG_User\\",
        "BEAMNG_HOME" : r"C:\BeamNG.drive-0.23.5.1.12888\\"
    }
}


########## DNN Model
DNN_MODEL_PATH =  "./dnn_models/maxibon/mixed-dave2.h5"

######## Simulators
DONKEY_EXE_PATH = sim_map[socket.gethostname()]["DONKEY_EXE_PATH"]
UDACITY_EXE_PATH = sim_map[socket.gethostname()]["UDACITY_EXE_PATH"]
BEAMNG_USER = sim_map[socket.gethostname()]["BEAMNG_USER"]
BEAMNG_HOME = sim_map[socket.gethostname()]["BEAMNG_HOME"]

###############
IN_WIDTH = 320
IN_HEIGHT = 160#240

DEFAULT_THROTTLE = 0.1

MAX_XTE = 3.0

ROAD_WIDTH = 8.0
NUM_CONTROL_NODES = 7
NUM_SAMPLED_POINTS = 20
MAX_ANGLE = 82

MAX_EPISODE_STEPS = 2000

IMAGE_CHANNELS = 3
INPUT_SHAPE = (IN_HEIGHT, IN_WIDTH, IMAGE_CHANNELS)
INPUT_DIM = INPUT_SHAPE
DISPLACEMENT = 2

MAX_SPEED = 30
MAX_SPEED_DONKEY = MAX_SPEED + 4
MIN_SPEED = 10

MAP_SIZE = 250
BBOX_SIZE = (-125, -125, 250, 250)
CROP_UDACITY = [60,-25]
CROP_DONKEY = [60,0]

MAX_NUM_GIFS = sys.maxsize

MODE_GIF_WRITING = "opt"

UDACITY_SLEEP = 2

STEERING_CORRECTION = 1

TIME_LIMIT = 30

MODE_PLOT_RESULTS = "opt"

CAP_XTE = True

MIN_SIM_TIME = 1
        
# USI
BEAMNG_SIM_NAME = "beamng"
DONKEY_SIM_NAME = "donkey"
UDACITY_SIM_NAME = "udacity"
MOCK_SIM_NAME = "mock"

SIMULATOR_NAMES = [BEAMNG_SIM_NAME, DONKEY_SIM_NAME, UDACITY_SIM_NAME, MOCK_SIM_NAME]
AGENT_TYPE_RANDOM = "random"
AGENT_TYPE_SUPERVISED = "supervised"
AGENT_TYPE_AUTOPILOT = "autopilot"
AGENT_TYPES = [AGENT_TYPE_RANDOM, AGENT_TYPE_SUPERVISED, AGENT_TYPE_AUTOPILOT]
TEST_GENERATORS = ["random", "sin"]

IMAGE_HEIGHT = IN_HEIGHT
IMAGE_WIDTH = IN_WIDTH

WAIT_RESETCAR = 2

MAX_CTE_ERROR = MAX_XTE

# OPENSBT
RESULTS_FOLDER = os.sep + "results" + os.sep + "single" +  os.sep
WRITE_ALL_INDIVIDUALS = True
CONSIDER_HIGH_VAL_OS_PLOT = True
PENALTY_MAX = 1000
PENALTY_MIN = -1000

# N CRIT METRICS
N_CELLS = 20
LAST_ITERATION_ONLY_DEFAULT = True
LOG_FILE = os.getcwd() + os.sep + "log.txt"
SEG_LENGTH = 20

MAX_SEG_LENGTH = 20
MIN_SEG_LENGTH = 10

CRITICAL_XTE = 2.2
CRITICAL_AVG_XTE = 1
MAX_ACC = 2.0
CRITICAL_STEERING = 8

DO_PLOT_GIFS = False
MAX_PLOT_DISAGREE = 10000
FPS_DESIRED_DONKEY = 15
PLOT_TESTS_DESIGN = False
BEAMNG_STEP_SIZE = 200

# road mutation parameters
MAX_DISPLACEMENT = 5 # add/sub max x degrees
MUT_PROB = 0.5
NUM_TRIALS = 3
BEAMNG_KILL_INTERVAL = 600 # need to kill because of the hardware error dialogue
BEAMNG_KILL_MAX = 24000
KILL_BMNG_PERIODICALLY = False

BACKUP_ITERATIONS = False
MAX_NUM_CRIT_GIFS = sys.maxsize

DUPLICATE_COMP_PRECISION = 8
OUTPUT_PRECISION = 8

NUM_TRIALS_MUT = 1
ROAD_MUT_PROB = 0.7

CROSS_RATE = 0
SEED = 760

THRESHOLD_VALIDATION = 0.5
VALIDATION_THRESHOLD_RANGE = [0.5,1]
STEP_VALIDATION = 0.1
EXECUTE_TWICE = False
RESULTS_FOLDER_NAME_PREDEFINED = "myfolder"

# use for multi and single sim; 
N_GENERATIONS = 20 # not considered if time is set
POPULATION_SIZE = 20
EXPERIMENTAL_MODE = False
PERCENTAGE_VALIDATION_SSIM = 3#0.00001 #equal to msim as we run it now once only
PERCENTAGE_VALIDATION_MSIM = 3#0.00001 #very low percentage rate to get only one test per cell
MAXIMAL_EXECUTION_TIME  = "06:00:00"
N_REPEAT_VALIDATION = 5
ARCHIVE_THRESHOLD = 4.5

REPLACE_ROAD_CX = True
MUTATION_CHECK_VALID_ALL = True

CLASSIFIER_DISAGREE_1 = "<SPECIFY THE CLASSIFIER PATH>"
CLASSIFIER_DISAGREE_2 = CLASSIFIER_DISAGREE_1

TH_PRED = 0.70 # UNCERTAINTY THRESHOLD FOR DISAGREEMENT PREDICTOR

EVALUATE_DISAGREEMENTS_PREDICT = True

# first last valid analysis
MAX_EVALUATIONS_FIRST_LAST_VALID = 500
MAX_EVALS_FIRST_VALID_MAP = {
    "b" : 510,
    "d" : 788,
    "u" : 870,
    "bd" : 302,
    "bu": 280, 
    "ud": 463,
    "dss-bd" : 534,
    "dss-bu" : 584,
    "dss-ud" : 692
}