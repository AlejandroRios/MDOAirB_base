import pickle
with open(r"Database/Results_Multi_Optim/variables/vars_multi_obj_GCD_profit_CO2.pkl", "rb") as input_file:
    df_vars = pickle.load(input_file)

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend',fontsize=12) # using a size in points
plt.rc('legend',fontsize='medium') # using a named size
plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(projection='3d')

n = 100

x=df_vars['X1'].values.tolist()
y=df_vars['X2'].values.tolist()
z=df_vars['X3'].values.tolist()

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
ax.scatter(x,y,z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()