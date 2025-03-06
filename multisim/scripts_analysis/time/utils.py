import numpy as np

def normalize_with_bounds(arr, min_bound, max_bound):
    # Convert array to numpy array for easy processing
    arr = np.array(arr)
    
    # Calculate the min and max of the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Apply the min-max normalization formula with given bounds
    normalized_arr = ((arr - min_val) / (max_val - min_val)) * (max_bound - min_bound) + min_bound
    
    return normalized_arr