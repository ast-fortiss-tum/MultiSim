import argparse
import os
import pandas as pd

from opensbt.utils.files import find_file


def write_validation_compared(output_folder, 
                              names,
                              names_sim_folders = None,
                              input_file_name = "validation_combined.csv",
                              output_file_name = "validation_compared.csv"):
    
    if names_sim_folders == None:
        names_sim_folders = names
    name_msim = names[0]
    name_sim1 = names[1]
    name_sim2 = names[2]

    # write together validation results in one csv file
    path_msim = find_file(output_folder + names_sim_folders[0] + os.sep, input_file_name)

    path_sim1 = find_file(output_folder + names_sim_folders[1] + os.sep, input_file_name)
    path_sim2 = find_file(output_folder + names_sim_folders[2] + os.sep, input_file_name)
    
    print(path_sim1)
    print(path_sim2)
    print(path_msim)

    df_msim = pd.read_csv(path_msim)
    df_sim1 = pd.read_csv(path_sim1)
    df_sim2 = pd.read_csv(path_sim2)

    # Ensure that the 'threshold' column is preserved
    df_msim.set_index('threshold', inplace=True)
    df_sim1.set_index('threshold', inplace=True)
    df_sim2.set_index('threshold', inplace=True)

    df_msim.columns = [f"{name_msim}_{col}" for col in df_msim.columns]
    df_sim1.columns = [f"{name_sim1}_{col}" for col in df_sim1.columns]
    df_sim2.columns = [f"{name_sim2}_{col}" for col in df_sim2.columns]

    df = pd.concat([df_msim, df_sim1, df_sim2], axis=1)

    # Reset the index to have 'threshold' as a regular column
    df.reset_index(inplace=True)

    df.to_csv(output_folder + os.sep + output_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write validation comparison results.")

    # Required arguments
    parser.add_argument("--output_folder", type=str, help="Path to the output folder", required=True)
    parser.add_argument("--names", nargs=3, type=str, help="List of three simulator names (e.g., msim sim1 sim2)", required=True)
    parser.add_argument("--input_file_name", type=str, help="Name of the input file.")
    parser.add_argument("--output_file_name", type=str, help="Name of the output file.")
    parser.add_argument("--names_sim_folders", nargs=3, type=str, help="Name of the folder for the three simulator results.")

    args = parser.parse_args()

    write_validation_compared(
        output_folder=args.output_folder,
        names=args.names,
        input_file_name=args.input_file_name,
        output_file_name=args.output_file_name,
        names_sim_folders=args.names_sim_folders
    )

# if __name__ == "__main__":
#     output_folder = "./analysis_3-runs_03-09-2024_17-21-03_4-gen_4-pop/run_0/"
#     names = ["msim", "sim_1","sim_2"]

#     write_validation_compared(names = names,
#                               output_folder = output_folder)