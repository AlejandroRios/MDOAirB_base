
import matplotlib.pyplot as plt
import numpy as np


eps = 1.45
omega = -10*np.pi/180
fi = -10*np.pi/180
beta = 0.54 + 0.1*np.tan(omega-fi)

d_v = 4.08

x = np.linspace(0,eps*d_v,100)

z1 =  (d_v**(1-beta))/2 * (x/eps)**beta + (eps*d_v - x)*np.tan(fi)
z2 = -(d_v**(1-beta))/2 * (x/eps)**beta + (eps*d_v - x)*np.tan(fi)


fig, ax = plt.subplots(figsize=(14,5))
plt.plot(x,z1)
plt.plot(x,z2)

eps = 2.5
omega = 0*np.pi/180
fi = 5*np.pi/180
beta = 0.54 + 0.1*np.tan(omega-fi)

d_v = 4.08


x = np.linspace(0,eps*d_v,100)

z3=  (d_v**(1-beta))/2 * (x/eps)**beta + (eps*d_v - x)*np.tan(fi)
z4 = -(d_v**(1-beta))/2 * (x/eps)**beta + (eps*d_v - x)*np.tan(fi)

x = np.flip(x)

plt.plot(25+x,z3)
plt.plot(25+x,z4)
ax.set_aspect('equal')
plt.ylim(-35,35)
plt.xlim(-35,35)

plt.legend()
plt.show()