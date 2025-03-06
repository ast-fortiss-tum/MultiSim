import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib

plt.rcParams.update({
    'xtick.labelsize': 20,  # X-axis tick labels font size
    'ytick.labelsize': 20,  # Y-axis tick labels font size
    'axes.labelsize': 20,   # Axis labels font size (both X and Y)
    'axes.titlesize': 20,   # Title font size
})
# matplotlib.use('TkAgg')

# Generate synthetic data
np.random.seed(42)
methods = ["Udacity", "Donkey", "BeamNG"]
conditions = [f"R{i+1}" for i in range(16)]

json_files = {
    "Udacity": "/home/lev/Documents/testing/MultiSimulation/msim-results/validation/Udacity_Validity_A5_0-360_XTE_ARLT/21-02-2025_22-41-26/validation_results.json",
    "Donkey": "/home/lev/Documents/testing/MultiSimulation/msim-results/validation/Donkey_Validity_A7_0-360_XTE/20-05-2024_23-01-08/validation_results_20-05-2024_23-01-08.json",
    "Beamng": "/home/lev/Documents/testing/MultiSimulation/msim-results/validation/Beamng_Validity_A7_0-360_XTE_step12/24-05-2024_15-34-00/validation_results_24-05-2024_15-34-00.json"
}

# Initialize list to store data
data_list = []

# Load and process each JSON file
for method, file in json_files.items():
    with open(file, "r") as f:
        json_data = json.load(f)
        all_fitness = json_data["all_fitness"]  # Extract main data
        
        # Iterate over conditions
        for condition_idx, condition_data in enumerate(all_fitness):
            # Flatten values (since each inner list contains a single value)
            values = [val[0] for val in condition_data]
            
            # Store as rows in data list
            for value in values:
                data_list.append({"Road": f"R{condition_idx+1}", "Method": method, "Value": value})

# Convert to DataFrame
df = pd.DataFrame(data_list)

# Create a grouped boxplot
plt.figure(figsize=(18, 9))
sns.boxplot(x="Road", y="Value", hue="Method", data=df, palette="Set2", width=0.8)

# TH
CRITICAL_XTE = 2.2
plt.axhline(y=-CRITICAL_XTE, color='r', linestyle='--')  # 'r' stands for red color, '--' stands for dashed line

plt.ylim(bottom = -3)

# Increase font sizes
plt.xticks(rotation=45)  # X-axis labels
plt.yticks()  # Y-axis labels
plt.xlabel("Road")
plt.ylabel("XTE")
plt.legend(fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("./scripts_analysis/out/preliminary_validation_u-d-b.pdf", format = "pdf")