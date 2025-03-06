#!/bin/bash

# Define the array of test cases as a JSON string
test_cases='-90, -110, 173, -120, -140, 10, 20, 18, 17,15'

# Run the Python script with the specified arguments
python -m scripts.evaluate_validity \
    -s "udacity" \
    -n 1 \
    -w \
    -prefix "emse_motivation" \
    -custom_lengths \
    -tests "$test_cases"
