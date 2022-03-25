from joblib import dump, load
import numpy as np
from framework.Performance.Engine.Turboprop.PW120model import PW120model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


scaler = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
nn_unit = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib') 



test = nn_unit.predict(scaler.transform([(0,0.1,1)]))
# test = scaler2.inverse_transform(np.reshape((test),(-1,1)))

print(test)

n = 11
M0 = np.linspace(0.01,0.7,n)
altitude = np.linspace(0,35000,n)
throttle_position = np.linspace(0.01,1,n)
mach_vec_model = np.linspace(0.01,0.7,11)


FC_vec_ANN = []
fuel_flow_vec_ANN = []

# FC_vec_model = []
# F_vec_model = []
# fuel_flow_vec_model = []

mach_vec = []


for i in M0:
    for j in altitude:

        FC_ANN = nn_unit.predict(scaler.transform([(j, i, 1)]))
        FC_vec_ANN.append(float(FC_ANN))

        
        # F_model, fuel_flow = PW120model(j,i,1)
        # FC_vec_model.append(float(fuel_flow))
        # F_vec_model.append(float(F_model))

        mach_vec.append(i)


FC_vec_model = np.load('Performance/Engine/Turboprop/ANN_skl_ff/FC_model.npy')
F_vec_model = np.load('Performance/Engine/Turboprop/ANN_skl_ff/F_model.npy')
mach_vec_model = np.load('Performance/Engine/Turboprop/ANN_skl_ff/mach_model.npy')
# np.save('Performance/Engine/Turboprop/ANN_skl_ff/FC_model.npy', FC_vec_model)
# np.save('Performance/Engine/Turboprop/ANN_skl_ff/F_model.npy', F_vec_model)
# np.save('Performance/Engine/Turboprop/ANN_skl_ff/mach_model',mach_vec)


FC_vec_ANN = np.reshape(FC_vec_ANN, (n,n))
# FC_vec_model = np.reshape(FC_vec_model, (n,n))
mach_vec = np.reshape(mach_vec, (n,n))


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)

# plt.plot(altitude,F_vec,'-')
# plt.plot(x[0:,0],y1,'o',alpha=0.5)
# plt.show()


ax1.plot(mach_vec_model,FC_vec_model,'^b',alpha=0.5,label='PW120 model')
ax1.plot(mach_vec,FC_vec_ANN, '-k',linewidth=2)
# ax1.plot(x[0:,0],y1, 'kx',label='linear regression',linewidth=2)

ax1.set_xlabel('Altitude [ft]')
ax1.set_ylabel('FC [Kg/hr]')
# ax1.set_title('Activation function: ReLU')

ax1.set_xlim([None,None])
ax1.set_ylim([None,None])


first_leg = mpatches.Patch(label='PW120 model')
second_leg = mpatches.Patch(label='PW120 ANN')
plt.legend(handles=[first_leg ,second_leg])

plt.grid(True)
plt.show()