
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

engine['fan_pressure_ratio'] = 1.1
engine['compressor_pressure_ratio'] = 25
engine['bypass'] = 6
engine['fan_diameter'] = 2
engine['turbine_inlet_temperature'] = 1500

# Single ponint run
engine_thrust, ff , vehicle = turbofan(0, 0.01, 1, vehicle)

# =============================================================================
# parametric analysis 
"""Plot Thrust force [N] vs fuel flow [kg/hr] as 
function of altitude and mach number"""

altitude = np.linspace(0, 40000,10)
mach = np.linspace(0, 0.8,10)

thrust_vec = []
ff_vec = []
altitude_vec = []
mach_vec = []
for i in altitude:
    for j in mach:
        engine_thrust, ff , vehicle = turbofan(i, j, 1, vehicle)

        thrust_vec.append(engine_thrust)
        ff_vec.append(ff)
        altitude_vec.append(i)
        mach_vec.append(j)
        

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

x = np.linspace(1., 8., 30)
ax.plot(thrust_vec,ff_vec, 'o', color='k',)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('Thrust (N)')
ax.set_ylabel('Fuel flow (kg/hr)')

# =============================================================================
# parametric analysis 
"""Plot fuel flow [kg/hr] vs Turbine inlet temperature"""

TiT = np.linspace(1000, 2000,10)

thrust_vec = []
ff_vec = []
TiT_vec = []
for i in TiT:

    engine['turbine_inlet_temperature'] = i
    engine_thrust, ff , vehicle = turbofan(0, 0.8, 1, vehicle)

    thrust_vec.append(engine_thrust)
    ff_vec.append(ff)
    TiT_vec.append(i)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.plot(TiT_vec,ff_vec, 'o', color='k',)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('TiT (K)')
ax.set_ylabel('Fuel flow (kg/hr)')
# =============================================================================

# =============================================================================
# parametric analysis 
"""Plot Fan Diameter [m] vs Thrust (N)"""

FD = np.linspace(1, 3,10)
engine['turbine_inlet_temperature'] = 1500
thrust_vec = []
ff_vec = []
TiT_vec = []
for i in FD:

    engine['fan_diameter'] = i
    engine_thrust, ff , vehicle = turbofan(0, 0.8, 1, vehicle)

    thrust_vec.append(engine_thrust)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.plot(FD, thrust_vec, 'o', color='k',)
# ax.plot(x, y, color='0.50', ls='dashed')
ax.set_xlabel('Fan Diameter (m)')
ax.set_ylabel('Thrust (N)')
# =============================================================================

plt.show()
