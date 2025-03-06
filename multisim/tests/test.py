import subprocess

executable = r'C:\Program Files\Git\bin\bash.exe'
command = "source venv/Scripts/activate && python run_analysis.py"
process = subprocess.Popen([executable,"-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
stdout, stderr = process.communicate()
# Decode and print the output
print(stdout.decode())
if stderr:
    print(stderr.decode())