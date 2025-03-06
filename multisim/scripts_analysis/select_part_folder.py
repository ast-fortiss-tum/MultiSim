import os
import re

def select_part_folder(directory, part = 2):
    # Path to the directory containing the "run_X" folders

    # List all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        
    # Use regular expression to match folders with the name pattern "run_X" and extract the number X
    run_folders = []
    for folder in folders:
        match = re.match(r'gen_(\d+)', folder)
        if match:
            run_folders.append((folder, int(match.group(1))))

    # Sort the folders based on the numeric value (X)
    sorted_run_folders = sorted(run_folders, key=lambda x: x[1])

    # Find the folder in the middle (median)
    middle_index = len(sorted_run_folders) // part
    selected_folder = sorted_run_folders[middle_index - 1][0]

    # Get the absolute path of the selected folder
    absolute_selected_folder = os.path.join(directory, selected_folder)

    # print("Selected folder (absolute path):", absolute_selected_folder)

    return absolute_selected_folder

if __name__ == "__main__":
    select_part_folder(r"C:\Users\Lev\Documents\testing\Multi-Simulation\opensbt-multisim\VARSEG_Mock_A10_-180-180_XTE_DIST_gen5_pop5_seed1\NSGAII-D\temp\\feature_map", 4)