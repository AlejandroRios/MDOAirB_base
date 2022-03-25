# Sample code for generation of first example
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
# from matplotlib import pyplot as plt
# pyplot imported for plotting graphs
 
x = np.linspace(10,100,10)
 
# numpy.linspace creates an array of
# 9 linearly placed elements between
# -4 and 4, both inclusive
y = np.linspace(10,100,10)
 
# The meshgrid function returns
# two 2-dimensional arrays
x_1, y_1 = np.meshgrid(x, y)
 
print("x_1 = ")
print(x_1)
print("y_1 = ")
print(y_1)

random_data = np.load("Performance/Engine/Turboprop/ANN_skl_ff/mse_data.npy")/100
plt.contourf(x_1, y_1, random_data, cmap='viridis',norm=LogNorm())
# print(im))
 
plt.colorbar()
plt.show()