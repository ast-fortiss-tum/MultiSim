import matplotlib.pyplot as plt
import numpy as np

# Configure global settings
plt.rcParams.update({
    'xtick.labelsize': 15,  # X-axis tick labels font size
    'ytick.labelsize': 15,  # Y-axis tick labels font size
    'axes.labelsize': 15,   # Axis labels font size (both X and Y)
    'axes.titlesize': 15,   # Title font size
})
# Data
approaches = ["BD", "BU", "UD", "U", "B", "D", "DSS-BD", "DSS-BU", "DSS-UD"]
efficiency = np.array([39.7, 35.3, 55.6, 22.8, 25.5, 22.4, 56.6, 39.9, 33.5])
effectiveness_vr = np.array([-98, -58, -54, -33, -62, -46, -89, -50, -56])
effectiveness_vn = np.array([-23, -19.2, -11.6, -24.9, -22.7, -20.2, -14.9, -16.6, -12.2])

def pareto_front(x, y):
    print(x,y)
    inds = []
    num_points = len(x)
    
    for i in range(num_points):
        dominated = False
        for j in range(num_points):
            if i != j:  # Avoid self-comparison
                # Check if point i is dominated by point j
                if (x[j] <= x[i] and y[j] < y[i]) or (x[j] < x[i] and y[j] <= y[i]):
                    dominated = True
                    break  # No need to check further if already dominated
        if not dominated:
            inds.append(i)
    result = [(x[i],y[i]) for i in inds]
    return result

# Function to create Pareto plot
def pareto_plot(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8,6))

    # Compute Pareto front
    pf = pareto_front(x,y)

    pareto_x = [p[0] for p in pf]
    pareto_y = [p[1] for p in pf]

    # Plot Pareto front correctly
    plt.plot(pareto_x, pareto_y, linestyle="--", color = "red", label="Pareto Front")
    
    plt.scatter(x[:3], y[:3], color='b', label="Approaches")
    plt.scatter(x[3:6], y[3:6], color='g', label="Approaches")
    plt.scatter(x[6:], y[6:], color='#F39C12', label="Approaches")

    # Annotate points
    for i, txt in enumerate(approaches):
        plt.annotate(txt, (x[i], y[i]), textcoords="offset points", size=16, xytext=(3,5), ha='right')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Plot efficiency vs effectiveness_vr
pareto_plot(efficiency, effectiveness_vr, "search_budget (%)", "valid_rate (%)", "")

# Plot efficiency vs effectiveness_vn
pareto_plot(efficiency, effectiveness_vn, "search_budget", "n_valid", "")
