from joblib import dump, load
import numpy as np
from framework.Performance.Engine.Turboprop.PW120model import PW120model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

scaler = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
scaler2 = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_out.bin')
nn_unit = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib') 



test = nn_unit.predict(scaler.transform([(0,0.1,1)]))
# test = scaler2.inverse_transform(np.reshape((test),(-1,1)))

print(test)

n = 11
M0 = np.linspace(0.01,0.7,n)
altitude = np.linspace(0,35000,n)
throttle_position = np.linspace(0.01,1,n)



F_vec_ANN = []
fuel_flow_vec_ANN = []

F_vec_model = []
fuel_flow_vec_model = []

altitude_vec = []


for i in altitude:
    for j in M0:

        F_ANN = nn_unit.predict(scaler.transform([(i, j, 1)]))
        F_vec_ANN.append(float(F_ANN))

        
        F_model, fuel_flow = PW120model(i,j,1)
        F_vec_model.append(float(F_model))

        altitude_vec.append(i)

        


# np.save('Performance/Engine/Turboprop/ANN_skl_ff/altitude_model',altitude_vec)


F_vec_ANN = np.reshape(F_vec_ANN, (n, n))

F_vec_model = np.reshape(F_vec_model, (n, n))
altitude_vec = np.reshape(altitude_vec, (n, n))



# y1 = np.asarray(F_vec)
# # y2 = np.asarray(fuel_flow_vec)

# x = np.load("Performance/Engine/Turboprop/ANN_skl_force/X_datat.npy")

# # x = np.load("X_datat.npy")
# # y1 = np.load("y1_datat.npy")
# # y2 = np.load("y2_datat.npy")



# import matplotlib.pyplot as plt

# x = np.load("Performance/Engine/Turboprop/X_data2.npy")
# y1 = np.load("Performance/Engine/Turboprop/y1_data2.npy")
# y2 = np.load("Performance/Engine/Turboprop/y2_data2.npy")

# y2 = np.where(y2<0, 0, y2)


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)

# plt.plot(altitude,F_vec,'-')
# plt.plot(x[0:,0],y1,'o',alpha=0.5)
# plt.show()


ax1.plot(altitude_vec,F_vec_model,'^b',alpha=0.5)
ax1.plot(altitude_vec,F_vec_ANN, '-k',linewidth=2)
# ax1.plot(x[0:,0],y1, 'kx',label='linear regression',linewidth=2)

ax1.set_xlabel('Altitude [ft]')
ax1.set_ylabel('Thrust Force [N]')
# ax1.set_title('Activation function: ReLU')

ax1.set_xlim([None,None])
ax1.set_ylim([None,None])

plt.grid(True)
first_leg = mpatches.Patch(label='PW120 model')
second_leg = mpatches.Patch(label='PW120 ANN')
plt.legend(handles=[first_leg ,second_leg])


plt.show()
