########### create box plot ############
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from config import CRITICAL_XTE
import os

n_roads = 2
n_repeat = 3

save_folder = os.getcwd() + "/results/"
fvalues_roads = [np.asarray([-2,-2,-3]),
                  np.asarray([-2.2,-2.2,-2.3])
                  ]
fig = plt.figure(figsize =(6, 4))
ax = fig.gca()

# Creating plot

plt.boxplot(fvalues_roads)
plt.xticks([i + 1 for i in range(n_roads)], [f'Road_{i + 1}' for i in range(n_roads)])
plt.title(f"Simulator Validation and Flakiness Analysis {n_repeat} Repetitions")
plt.ylim(-3.5, 0)

for i,fvalues in enumerate(fvalues_roads):
    ax.plot([i + 1]*n_repeat, np.asarray(fvalues),'ko', alpha=0.7,markerfacecolor='none')  # 'ro' stands for red circles
   
plt.axhline(y=-CRITICAL_XTE, color='r', linestyle='--')  # 'r' stands for red color, '--' stands for dashed line

# show plot
plt.show()
plt.savefig(save_folder + f"test_boxplot.jpg")
plt.clf()