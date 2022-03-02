"""
File name :
Authors   : 
Email     : aarc.88@gmail.com
Date      : 
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
Inputs:
    -
Outputs:
    -
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
import matplotlib.pyplot as plt
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


# fig, ax1 = plt.subplots()
# line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
# ax1.set_xlabel("Generation")
# ax1.set_ylabel("Fitness", color="b")
# for tl in ax1.get_yticklabels():
#     tl.set_color("b")

# ax2 = ax1.twinx()
# line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
# ax2.set_ylabel("Size", color="r")
# for tl in ax2.get_yticklabels():
#     tl.set_color("r")

# lns = line1 + line2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc="center right")

# plt.show()
import pandas as pd
df = pd.read_csv('Database/Results/Optimization/optim_statistics02.txt', sep=",",header=None)
df.columns = ['gen','nevals','average','standard','minimum','maximum']

print(df.nevals)

import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
t = df.gen
data1 = df.maximum
data2 = df.average

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('generations')
ax1.set_ylabel('maximum', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('average', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
