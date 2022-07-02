
"""
MDOAirB

Description:
    - This test module performs parametric analysis of the
    engine model

TODO's:

| Authors: Alejandro Rios
           
  
| Email: aarc.88@gmail.com
| Creation: January 2022
| Last modification: 
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Performance.Engine.engine_performance import turbofan

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================

# Aircraft initialization
vehicle = initialize_aircraft_parameters()

# Call engine parameters
engine = vehicle['engine']

engine['fan_pressure_ratio'] = 1.5
engine['compressor_pressure_ratio'] = 27
engine['bypass'] = 5
engine['fan_diameter'] = 2.5
engine['turbine_inlet_temperature'] = 1350

# Single ponint run
engine_thrust, ff , vehicle = turbofan(0, 0.01, 1, vehicle)
print(engine_thrust, ff)

# =============================================================================
# parametric analysis 
"""Plot Thrust force [N] vs fuel flow [kg/hr] as 
function of mach and throttle"""

altitude = np.linspace(0, 42000,50)
mach = np.linspace(0, 0.85,50)
throttle = np.linspace(0.4, 1,50)

thrust_vec = []
ff_vec = []
altitude_vec = []
mach_vec = []
throttle_vec = []
sfc_vec = []
for i in altitude:
    for j in mach:
        for k in throttle:
            engine_thrust, ff , vehicle = turbofan(i,j,k, vehicle)
            sfc = (ff)/(engine_thrust/10)  # sfc in kg/h/daN

            # if engine_thrust > 0 and ff > 0:
            if sfc > 10:

                

                thrust_vec.append(engine_thrust)
                ff_vec.append(ff)
                altitude_vec.append(i)
                mach_vec.append(j)
                throttle_vec.append(k)

                sfc_vec.append(sfc)



plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(10, 9))
# ax = fig.add_subplot(1, 1, 1)
ax = plt.axes(projection='3d')

# x = np.linspace(1., 8., 30)
ax.plot(altitude_vec,throttle_vec,sfc_vec, 'o', color='k',alpha=0.2)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('altitude')
ax.set_ylabel('throttle')
ax.set_zlabel('sfc')
# plt.show()
# =============================================================================
# parametric analysis 
# """Plot fuel flow [kg/hr] vs Turbine inlet temperature"""

# TiT = np.linspace(1000, 2000,10)

# thrust_vec = []
# ff_vec = []
# TiT_vec = []
# for i in TiT:

#     engine['turbine_inlet_temperature'] = i
#     engine_thrust, ff , vehicle = turbofan(0, 0.8, 1, vehicle)

#     thrust_vec.append(engine_thrust)
#     ff_vec.append(ff)
#     TiT_vec.append(i)

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(10, 9))
ax = plt.axes(projection='3d')

ax.plot(altitude_vec,mach_vec,throttle_vec, 'o', color='k',alpha=0.2)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('altitude')
ax.set_ylabel('mach')
ax.set_zlabel('throttle')
# =============================================================================

plt.show()
