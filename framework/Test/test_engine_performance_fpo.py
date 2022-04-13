
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
engine['bypass'] = 6
engine['fan_diameter'] = 2
engine['turbine_inlet_temperature'] = 1350

# Single ponint run
engine_thrust, ff , vehicle = turbofan(0, 0.01, 1, vehicle)

# =============================================================================
# parametric analysis 
"""Plot Thrust force [N] vs fuel flow [kg/hr] as 
function of altitude and mach number"""

altitude = 35000
mach = np.linspace(0.3, 0.82,3)
throttle = np.linspace(0.3,0.95,100)

step = 1
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize=(10, 9))
plt.suptitle('Altitude = {} ft'.format(altitude))

for j in mach:
    sfc_vec = []
    throttle_vec = []
    
    for i in throttle:
        engine_thrust, ff , vehicle = turbofan(altitude, j, i, vehicle)
        sfc = ff/(engine_thrust/10) #kg/(h*daN)
        #sfc_vec.append(ff/engine_thrust)
        sfc_vec.append(sfc)
        throttle_vec.append(i)
        
    ax = fig.add_subplot(1, 3, step)
    ax.plot(throttle_vec,sfc_vec, 'o', color='k',)
    ax.set_xlabel('Thrust Throttle')
    ax.set_ylabel('SFC (kg/(h.daN))')
    ax.title.set_text('Mach = {}'.format(j))
    # add title with mach number and altitude ax.set
    
    step = step + 1
    
plt.show()


