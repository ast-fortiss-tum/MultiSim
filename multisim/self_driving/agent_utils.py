import math
import cv2
import numpy as np
from config import UDACITY_SIM_NAME, DONKEY_SIM_NAME
from config import IN_WIDTH, IN_HEIGHT, CROP_DONKEY, CROP_UDACITY

def crop(image: np.ndarray, simulator_name: str) -> np.ndarray:
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    if simulator_name == UDACITY_SIM_NAME:
        return image[CROP_UDACITY[0]:CROP_UDACITY[1], :, :]  # remove the sky and the car front (from SelfOracle)
    elif simulator_name == DONKEY_SIM_NAME:
        return image[CROP_DONKEY[0]:CROP_DONKEY[1], :, :]  # remove the sky and the car front (from SelfOracle)
    else:
        print("Name not known for applying cropping.")

def resize(image: np.ndarray, width = None, height = None) -> np.ndarray:
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IN_WIDTH  if width == None else width , IN_HEIGHT if height == None else height), cv2.INTER_AREA)


def bgr2yuv(image: np.ndarray) -> np.ndarray:
    """
    Convert the image from BGR to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def preprocess(image: np.ndarray, width = None, height = None, simulator_name = None) -> np.ndarray:
    """
    Combine all preprocess functions into one
    """
    image = crop(image=image, simulator_name=simulator_name)
    image = resize(image=image, width=width, height=height)
    image = bgr2yuv(image=image)
    return image


# calc yaw based on trajectory
# pos is an array of arrays
def calc_yaw_ego(pos):
    yaw = []
    start_yaw = 90
    yaw.append(start_yaw)
    for i in range(len(pos) - 1):
        # if vehicle is at the same position
        if (pos[i+1][0] - pos[i][0] == 0):
            value = yaw[-1]
        else:
            div = (pos[i+1][1] - pos[i][1]) / (pos[i+1][0] - pos[i][0])
            value = math.atan(div) * 180 / math.pi
        yaw.append(value)
    return yaw
