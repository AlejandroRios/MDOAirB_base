from joblib import dump, load
import numpy as np

scaler = load('scaler_force_PW120_in.bin') 
scaler2 = load('scaler_force_PW120_out.bin')
nn_unit = load('nn_force_PW120.joblib') 



test = nn_unit.predict(scaler.transform([(0,0.1,1)]))
test = scaler2.inverse_transform(np.reshape((test),(-1,1)))

print(test)


M0 = np.linspace(0.01,0.7,51)
altitude = np.linspace(0,35000,51)
throttle_position = np.linspace(0.01,1,11)



F_vec = []
fuel_flow_vec = []

# X_data = [[i, j] for i in altitude for j in M0]
X_data = [[i] for i in M0]

X_data = np.asarray(X_data)
# print(X_data[0:,0])

for i in X_data:
    
    test = nn_unit.predict(scaler.transform([(0, i[0], 1)]))
    test = scaler2.inverse_transform(np.reshape((test),(-1,1)))

    # F, fuel_flow = PW120model(25000, i[0], 1)

    F_vec.append(float(test))
    # fuel_flow_vec.append(fuel_flow) 

F_vec = np.asarray(F_vec)
y1 = np.asarray(F_vec)
# y2 = np.asarray(fuel_flow_vec)

x = np.load("X_datat.npy")

# x = np.load("X_datat.npy")
# y1 = np.load("y1_datat.npy")
# y2 = np.load("y2_datat.npy")



import matplotlib.pyplot as plt

x = np.load("X_data2.npy")
y1 = np.load("y1_data2.npy")
y2 = np.load("y2_data2.npy")

y2 = np.where(y2<0, 0, y2)

plt.plot(M0,F_vec,'x')
plt.plot(x[0:,1],y1,'o')
plt.show()