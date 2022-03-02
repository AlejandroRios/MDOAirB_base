from matplotlib import projections
import numpy as np

from tqdm import trange, tqdm

from framework.Performance.Engine.Turboprop.PW120model import PW120model
M0 = np.linspace(0.01,0.7,21)
altitude = np.linspace(0,35000,21)
throttle_position = np.linspace(0.1,1,11)

F, fuel_flow = PW120model(25000,0.5,0.8)

F_vec = []
fuel_flow_vec = []

X_data = [[i, j, k] for i in altitude for j in M0 for k in throttle_position]

X_data = np.asarray(X_data)
# print(X_data[0:,0])

print(len(X_data))
# for j in trange(len(X_data)):
for i in tqdm(X_data):
    # print('i=',i)
    F, fuel_flow = PW120model(i[0], i[1], i[2])
    
    F_vec.append(F)
    fuel_flow_vec.append(fuel_flow)


y1 = np.asarray(F_vec)
y2 = np.asarray(fuel_flow_vec)


np.save("X_data2.npy", X_data)
np.save("y1_data2.npy", y1)
np.save("y2_data2.npy", y2)

print(F_vec)
print(fuel_flow_vec)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(X_data[0:,0], X_data[0:,1], X_data[0:,2], c=F_vec, cmap=plt.hot())
fig.colorbar(img)
plt.show()



fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
img = ax2.scatter(X_data[0:,0], X_data[0:,1], X_data[0:,2], c=fuel_flow_vec, cmap=plt.hot())
fig2.colorbar(img)
plt.show()
