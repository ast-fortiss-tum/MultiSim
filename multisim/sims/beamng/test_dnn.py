import time
from typing import Dict
from beamngpy import BeamNGpy, Scenario, Vehicle

import config
from sims.beamng.beamng_gym_env import BeamngGymEnv
from sims.beamng_simulation import ScenarioOutcome
from sims.donkey.core import donkey_sim
from road_generator.custom_road_generator import CustomRoadGenerator
from self_driving.supervised_agent import SupervisedAgent

import logging

logging.basicConfig(level = logging.INFO)

road_generator = CustomRoadGenerator(map_size = config.MAP_SIZE, 
                                    num_control_nodes=config.NUM_CONTROL_NODES,
                                        seg_length=config.SEG_LENGTH)
env = BeamngGymEnv(
            seed=1,
            add_to_port=0,
            test_generator=road_generator,
            #simulator_scene=GeneratedTrack(),
            autopilot=False,
            beamng_home=config.BEAMNG_HOME,
            beamng_user=config.BEAMNG_USER
)

starting_pos = (125.0, 0.0, -28.0, config.ROAD_WIDTH)
angles = [10,10,10,0,0]
road = road_generator.generate(starting_pos=starting_pos,
                                angles=angles, 
                                seg_length=config.SEG_LENGTH,
                                simulator_name=config.BEAMNG_SIM_NAME)

obs = env.reset(skip_generation=False,
          road=road)
input("Press enter to proceed...")

agent = SupervisedAgent(env=env,
                env_name=config.BEAMNG_SIM_NAME,
                model_path=config.DNN_MODEL_PATH,
                min_speed=config.MIN_SPEED,
                max_speed=config.MAX_SPEED,
                input_shape=config.INPUT_SHAPE,
                predict_throttle=False,
                fake_images=False
                )

# set up params for saving data
pos_list = []
xte_list = []
actions_list = []
speed_list = []
isSuccess = False
state = {}
done = False

# run the scenario
while not done:
    try:
        # TODO: Play around with this value
        time.sleep(0.01)
        state["simulator_name"] = config.BEAMNG_SIM_NAME
        actions = agent.predict(obs,
                    state = state)
        
        obs, done, info = env.step(actions)
    
        state["xte"] = info.get("cte", None)
        state["xte_pid"] = info.get("xte_pid", None)
        state["speed"] = info.get("speed", None)
        state["image"] = obs

        # steering_angles.append(action[0])
        pos = info.get("pos", None)
        print(f"lateral position is: {pos}")
        # check if we are done
        print(f"XTE is: {state['xte']}")
        
        # save data for output
        pos_list.append([
                            pos[0], 
                            pos[1], 
                            0
                        ])
        xte_list.append(state["xte"])
        speed_list.append(state["speed"])
        actions_list.append(actions)
        
        # if done:
        #     isSuccess = True
        #     break
        # elif abs(state["xte"]) > config.MAX_XTE:
        #     done = True
        #     break

    except KeyboardInterrupt:
        print(f"{5 * '+'} SDSandBox Simulator Got Interrupted {5 * '+'}")
        # self.client.stop()
        raise KeyboardInterrupt


d = Dict(
    frames=[x for x in range(len(pos_list))],
    pos=pos_list,
    xte=xte_list,
    speeds=speed_list,
    actions=[(f"{action[0][0]}", f"{action[0][0]}") for action in actions_list],
    road=road,
    isSuccess=isSuccess,
)
print(d)


# Make the vehicle's AI span the map
# vehicle_1.ai_set_mode('span')
# vehicle_2.ai_set_mode('manual')

input('Hit enter when done...')