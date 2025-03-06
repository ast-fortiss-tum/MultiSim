import os
import logging as log
import pandas as pd

def find_file(root_dir, prefix):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(prefix):
                matching_file  = os.path.join(dirpath, filename)
                return matching_file
    return None

def find_file_contains(root_dir, words, ends_with = None):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            contains = all([w in filename for w in words])  # Check if all words are in the filename
            if contains:
                if ends_with is not None and filename.endswith(ends_with):  # Also check if filename ends with the specified suffix
                    matching_file = os.path.join(dirpath, filename)
                    return matching_file
                if ends_with is None:
                    matching_file = os.path.join(dirpath, filename)
                    return matching_file
    return None

def find_files(root_dir, prefix):
    matching_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(prefix):
                matching_file  = os.path.join(dirpath, filename)
                matching_files.append(matching_file)
    return matching_files

def find_files_with_parent(root_dir, prefix, parent_folder):
    matching_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # print(dirpath)
            # print(filename)
            # input()
            if filename.startswith(prefix) and os.path.basename(dirpath) == parent_folder:
                matching_file = os.path.join(dirpath, filename)
                matching_files.append(matching_file)
    return matching_files

def find_parent(root_dir, folder_name):
    for root, dirs, files in os.walk(root_dir):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

def concatenate_csv(file1,file2, filename):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Concatenate the dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_file_path = filename + ".csv"
    merged_df.to_csv(merged_file_path, index=False)
    
def find_folders_with_name(start_dir, folder_name):
    matching_folders = []
    for root, dirs, files in os.walk(start_dir):
        for dir_name in dirs:
            if dir_name == folder_name:
                matching_folders.append(os.path.join(root, dir_name))
    return matching_folders