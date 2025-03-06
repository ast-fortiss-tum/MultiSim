import os
import dill
import argparse
from opensbt.visualization import output_predict

def process_run_path(run_path):
    path_result = os.path.join(run_path, "backup", "result")
    
    # Read result object
    with open(path_result, 'rb') as f:
        res = dill.load(f)
        print(f"Loaded Result from {run_path}:", res)
    
    # Call related output functions
    output_predict.write_predictions(res, save_folder=run_path)
    output_predict.write_prediction_summary(res, save_folder=run_path)

def main(run_paths):
    for run_path in run_paths:
        process_run_path(run_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple result files and generate output predictions.")
    parser.add_argument("run_paths", type=str, nargs='+', help="Paths to the analysis run directories")
    
    args = parser.parse_args()
    main(args.run_paths)
