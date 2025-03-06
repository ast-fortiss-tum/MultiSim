import platform
import subprocess
from self_driving.global_log import GlobalLog
import logging as log

def kill_donkey_simulator() -> None:

    logg = GlobalLog("kill_donkey_simulator")

    plt = platform.system()
    
    if plt.lower() == "linux":
        keyword = "donkey"
        try:
            # Execute the pkill command with the specified keyword
            subprocess.run(['pkill', '-f', keyword], check=True)
            print(f"All processes containing '{keyword}' have been terminated.")
        except subprocess.CalledProcessError:
            print(f"No processes containing '{keyword}' found.")
    else:
        beamng_program_name = "donkey_sim"

        # windows
        cmd = "tasklist"

        ret = subprocess.check_output(cmd)
        output_str = ret.decode("utf-8")

        program_name = beamng_program_name
        if program_name in output_str:
            cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
            ret = subprocess.check_output(cmd)
            output_str = ret.decode("utf-8")
            logg.info(output_str)
        else:
            logg.warn("The program {} is not in the list of currently running programs".format(beamng_program_name))


def kill_beamng_simulator() -> None:

    logg = GlobalLog("kill_beamng_simulator")

    beamng_program_name = "BeamNG.drive.x64"

    plt = platform.system()
    assert plt.lower() == "windows", "Platform {} not supported yet".format(plt.lower())

    cmd = "tasklist"

    ret = subprocess.check_output(cmd)
    output_str = ret.decode("utf-8")

    program_name = beamng_program_name
    if program_name in output_str:
        try:
            cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
            ret = subprocess.check_output(cmd)
            output_str = ret.decode("utf-8")
            logg.info(output_str)
        except subprocess.CalledProcessError:
            print(f"No processes containing '{program_name}' found.")
    else:
        logg.warn("The program {} is not in the list of currently running programs".format(beamng_program_name))

def kill_udacity_simulator() -> None:

    logg = GlobalLog("kill_udacity_simulator")

    plt = platform.system()
    if plt.lower() == "linux":
        keyword = "udacity_sim_lin"
        try:
            # Execute the pkill command with the specified keyword
            subprocess.run(['pkill', '-f', keyword], check=True)
            print(f"All processes containing '{keyword}' have been terminated.")
        except subprocess.CalledProcessError:
            print(f"No processes containing '{keyword}' found.")
    else:
        beamng_program_name = "udacity"

        # windows
        cmd = "tasklist"

        ret = subprocess.check_output(cmd)
        output_str = ret.decode("utf-8")

        program_name = "udacity"
        if program_name in output_str:
            cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
            ret = subprocess.check_output(cmd)
            output_str = ret.decode("utf-8")
            logg.info(output_str)
        else:
            logg.warn("The program {} is not in the list of currently running programs".format(beamng_program_name))



if __name__ == "__main__":
    kill_udacity_simulator()
