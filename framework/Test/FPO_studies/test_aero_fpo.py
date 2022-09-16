
"""
MDOAirB

Description:
    - This test module performs parametric analysis of the
    aero model

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
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Attributes.Airspeed.airspeed import mach_to_V_tas

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================

# Aircraft initialization
vehicle = initialize_aircraft_parameters()
wing = vehicle['wing']
operations = vehicle['operations']
wing_surface = wing['area']
delta_ISA = operations['flight_planning_delta_ISA']

knots_to_feet_minute = 101.268
knots_to_meters_second = 0.514444
ft_to_m = 0.3048
GRAVITY = 9.80665

switch_neural_network = 0
ft_to_m = 0.3048
mass = 100000

# =============================================================================
# parametric analysis 
"""Plot Thrust force [N] vs fuel flow [kg/hr] as 
function of altitude and mach number"""

altitude = 35000
mach = np.linspace(0.72, 0.82,3)
alpha = np.linspace(-10,30,50)

step = 1
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

fig = plt.figure(figsize=(10, 9))
plt.suptitle('Altitude = {} ft'.format(altitude), fontsize=20)

# As a function of angle of attack
# for j in mach:
#     CL_vec = []
#     CL_in_vec = []
#     CD_vec = []
    
#     for i in alpha:
#         V_tas = mach_to_V_tas(j, altitude, delta_ISA)
#         _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)
#         CL_input = (2*mass*GRAVITY) / (rho_ISA*((V_tas*knots_to_meters_second)**2)*wing_surface)

#         CD, CL = aerodynamic_coefficients_ANN(vehicle, altitude*ft_to_m, j, CL_input, i, switch_neural_network) 
#         #sfc_vec.append(ff/engine_thrust)
#         CL_vec.append(CL)
#         CD_vec.append(CD)
#         CL_in_vec.append(CL_input)
        
#     ax = fig.add_subplot(1, 3, step)
#     ax.plot(CL_vec,CD_vec, 'o', color='k',)
#     ax.set_xlabel('CL')
#     ax.set_ylabel('CD')
#     ax.title.set_text('Mach = {}'.format(j))
#     # add title with mach number and altitude ax.set
    
#     step = step + 1


CL_in_vec = np.linspace(0, 1.2, 50)
alpha_deg = np.linspace(0, 2, 3)
colours = ['r', 'g', 'b']

for k in alpha_deg:
    step2 = 0
    ax = fig.add_subplot(1, 3, step)
    ax.legend(loc="upper left")
    ax.set_xlabel('CL', fontsize=20)
    ax.set_ylabel('CD', fontsize=20)
    ax.set_title('Alpha = {} deg'.format(k), fontsize=20)
    
    for j in mach:
        CD_vec = []
        CL_vec = []
        

        for i in CL_in_vec:

            CD, CL = aerodynamic_coefficients_ANN(
            vehicle, altitude*ft_to_m, j, i, k, switch_neural_network)
            CL_vec.append(CL)
            CD_vec.append(CD)
            
        ax.plot(CL_vec, CD_vec,
                color=colours[step2], label="Mach = {}".format(j))
        step2 = step2 + 1

    step = step + 1
    
plt.show()


