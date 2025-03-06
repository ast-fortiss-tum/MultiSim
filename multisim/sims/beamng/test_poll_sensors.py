from beamngpy import BeamNGpy, Scenario, Vehicle
import config
import logging

from sims.beamng.beamng_brewer import BeamNGCamera

logging.basicConfig(level = logging.INFO)

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home=config.BEAMNG_HOME, user=config.BEAMNG_USER)
# Launch BeamNG.tech
bng.open()
# Create a scenario in west_coast_usa called 'example'

scenario = Scenario('west_coast_usa', 'example')

# scenario = Scenario('tig', 'example')

# Create an ETK800 with the licence plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 105, 118), rot=None, rot_quat=(0, 0, 0.3826834, 0.9238795))
# Place files defining our scenario for the simulator to read
scenario.make(bng)

camera = BeamNGCamera(bng, "brewer_camera")

# Load and start our scenario
bng.load_scenario(scenario)
bng.start_scenario()
# Make the vehicle's AI span the map
vehicle.ai_set_mode('manual')#

image = camera.get_rgb_image()

print(f"sensors output: {vehicle.poll_sensors()}")


input('Hit enter when done...')