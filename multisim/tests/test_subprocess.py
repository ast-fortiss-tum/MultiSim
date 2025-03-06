import subprocess
import sys
import os

# dir_path = os.getcwd() + os.sep + "venv/"
# sys.path.insert(0, dir_path)

venv_path = os.path.join(os.getcwd(), "venv","Scripts","activate.bat")
command = f"cmd /c {venv_path}" + " && python run.py -e 1"

process = subprocess.Popen(command, shell=True)
process.communicate()

process = subprocess.Popen(command, shell=True)
process.communicate()
