
import argparse
import os
from pathlib import Path
import subprocess
import platform

if os.name == "posix":
    venv_path = os.path.join(os.getcwd(), "venv","bin","activate")
else:
    venv_path = os.path.join(os.getcwd(), "venv","Scripts","activate")


def run_exp(exp_num, 
            seed, 
            output_folder, 
            pop_size = None, 
            n_gen = None,
            do_validate = False,
            n_repeat_validation = 5,
            write_subfolder_name = None,
            sims_validation = None):
    
    Path(output_folder).mkdir(parents = True, exist_ok = True)

    command = ""     

    if do_validate:
        command = f"source {venv_path}" + f" && python run.py -e {exp_num} -s {seed} \
                                            -o {output_folder} -v -rv {n_repeat_validation}" + " -sims_validation " + " ".join(sims_validation) 
    else:
        command = f"source {venv_path}" + f" && python run.py -e {exp_num} -s {seed} \
                                            -o {output_folder} -rv {n_repeat_validation}"
        
    if write_subfolder_name is not None:
        command += f" -sf '{write_subfolder_name}'"
        
    if n_gen is not None:
        command += f" -i {n_gen}"  
    if pop_size is not None:
        command += f" -n {pop_size}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,  executable="/bin/bash")    
    process.communicate()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the experiment with specified parameters.")

    # Required arguments
    parser.add_argument("--exp_num", type=int, help="Experiment number",required=True)
    parser.add_argument("--seed", type=int, help="Random seed for the experiment",required=True)
    parser.add_argument("--output_folder", type=str, help="Path to the output folder", required=True)

    # Optional arguments
    parser.add_argument("--pop_size", type=int, help="Population size", default=None)
    parser.add_argument("--n_gen", type=int, help="Number of generations", default=None)
    parser.add_argument("--do_validate", action="store_true", help="Flag to perform validation")
    parser.add_argument("--n_repeat_validation", type=int, help="Number of repeat validations", default=5)
    parser.add_argument("--write_subfolder_name", type=str, help="Subfolder name to write the output", default=None)
    parser.add_argument('--sims_validation', nargs="+", type=str, action="store", default="udacity donkey", 
                            help='Simulator names to use for validation")')
    
    args = parser.parse_args()

    run_exp(
        exp_num=args.exp_num,
        seed=args.seed,
        output_folder=args.output_folder,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        do_validate=args.do_validate,
        n_repeat_validation=args.n_repeat_validation,
        write_subfolder_name=args.write_subfolder_name,
        sims_validation=args.sims_validation
    )