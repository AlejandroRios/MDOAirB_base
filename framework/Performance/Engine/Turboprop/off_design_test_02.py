from matplotlib import projections
import numpy as np

from framework.Performance.Engine.Turboprop.PW120model import PW120model
M0 = np.linspace(0.01,0.7,11)
altitude = np.linspace(0,35000,11)
throttle_position = np.linspace(0.1,1,11)



F_vec = []
fuel_flow_vec = []

X_data = [[i, j] for i in altitude for j in M0]

X_data = np.asarray(X_data)
# print(X_data[0:,0])

for i in X_data:
    F, fuel_flow = PW120model(i[0], i[1], 1)
    
    F_vec.append(F)
    fuel_flow_vec.append(fuel_flow)


y1 = np.asarray(F_vec)
y2 = np.asarray(fuel_flow_vec)



x = np.load("X_datat.npy")
y1 = np.load("y1_datat.npy")
y2 = np.load("y2_datat.npy")



import matplotlib.pyplot as plt

plt.plot(x[0:,0],y2)
plt.show()
 