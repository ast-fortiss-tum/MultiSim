from datetime import datetime
import os
from pathlib import Path
from PIL import Image

def write_image(scenario_cnt, image, frequency = 1, last_time = 0, time = 0, simname = "udacity"):

    assert f"{simname}" == "udacity" or f"{simname}" == "donkey" or f"{simname}" == "beamng", "Name not known."

    save_folder = f"./recording/" + "images" + f"/{simname}"+ os.sep + f"{scenario_cnt}" + os.sep
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    output_file = f'scenario_{scenario_cnt}_{simname}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f")}.jpg'
    image = Image.fromarray(image)
    image.save(save_folder + output_file)
