import numpy as np
import matplotlib.pyplot as plt

eta_propmax = 0.9564

M0_vec = np.linspace(0,0.85)

eta_prop_vec = []
for i in range(len(M0_vec)):
    # eta_prop as function of Mach

    M0 = M0_vec[i]
    if M0 <= 0.1:
        eta_prop = 10*M0*eta_propmax
    elif M0 > 0.1 and M0 <= 0.7:
        eta_prop = eta_propmax
    elif M0 > 0.7 and M0 <= 0.85:
        eta_prop = (1 - (M0-0.7)/3)*eta_propmax
    
    eta_prop_vec.append(eta_prop)


plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend',fontsize=12) # using a size in points
plt.rc('legend',fontsize='medium') # using a named size
plt.rc('axes',labelsize=12, titlesize=12) # using a size in points
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.plot(M0_vec, eta_prop_vec, '-', color='k',)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('$M_0$')
ax.set_ylabel('$\eta_{prop}$')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.grid(True)
# =============================================================================

plt.show()

