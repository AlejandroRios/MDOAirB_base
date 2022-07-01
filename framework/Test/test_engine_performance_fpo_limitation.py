
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
altitude = 29564.737478520674
mach = 0.8076856826859163
throttle_position = 0.43000000000000005

#SFC = 80.41579970448574
#Fuel Flow = 269.38946522474015
#Thrust = 33.49956926557967
#Altitude = 29564.737478520674, Mach = 0.8076856826859163, Throttle position = 0.43000000000000005
# Single ponint run
engine_thrust, ff , vehicle = turbofan(altitude, mach, throttle_position, vehicle)

print('SFC = {}'.format(ff/(engine_thrust/10)))
print('Fuel Flow = {}'.format(ff))
print('Thrust = {}'.format(engine_thrust))

# =============================================================================
# parametric analysis 
"""Plot Thrust force [N] vs fuel flow [kg/hr] as 
function of altitude and mach number"""

altitude = np.linspace(1500,35000,100)
mach = np.linspace(0.78, 0.82,3)
throttle = 0.43

step = 1
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(figsize=(10, 9))
plt.suptitle('Throttle = {}'.format(throttle), fontsize=20)

for j in mach:
    sfc_vec = []
    altitude_vec = []
    
    for i in altitude:
        engine_thrust, ff , vehicle = turbofan(i, j, throttle, vehicle)
        sfc = ff/(engine_thrust/10) #kg/(h*daN)
        #sfc_vec.append(ff/engine_thrust)
        sfc_vec.append(sfc)
        altitude_vec.append(i)
        
    ax = fig.add_subplot(1, 3, step)
    ax.plot(altitude_vec,sfc_vec, color='k')
    ax.set_xlabel('Altitude (ft)', fontsize=20)
    ax.set_ylabel('SFC (kg/(h.daN))', fontsize=20)
    ax.set_title('Mach = {}'.format(j), fontsize=20)
    # add title with mach number and altitude ax.set
    
    step = step + 1
    
plt.show()


