"""
MDOAirB

Description:
    - This module obtain the cruise altutude considering buffeting constraints

Reference:
    - Reference: Ruijgrok, Elements of airplane performance
    - Chapter 10, pag 261

TODO's:
    -

| Authors: Alejandro Rios
| Email: aarc.88@gmail.com
| Creation: January 2021
| Last modification: July 2021
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
import numpy as np

from framework.Attributes.Airspeed.airspeed import mach_to_V_tas
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665

def buffet_altitude(vehicle, mass, altitude, limit_altitude, mach_climb):
    '''
    Description:
        - This function performs the evaluation of the buffet altitude

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - mass - aircraft maximum mass [kg]
        - altitude - [ft]
        - limit_altitude - [ft]
        - mach - mach number_climb

    Outputs:
        - altitude - [ft]
    '''
    airport_departure = vehicle['airport_departure']
    wing = vehicle['wing']
    operations = vehicle['operations']
    
    wing_surface = wing['area']
    step = 100
    load_factor = 1.3
    gamma = 1.4
    delta_ISA = operations['flight_planning_delta_ISA']

    # Typical values of wing loading for jet airplanes around 5749 [Pascal]
    wing_loading_constraint = 6000
    _, _, _, _, P_ISA, _, _, _ = atmosphere_ISA_deviation(
        limit_altitude, delta_ISA)
    CL_constraint = ((2)/(gamma*P_ISA*mach_climb**2))*wing_loading_constraint

    CL = 0.1
    max_count = 10000
    count = 0
    while (CL < CL_constraint) and (count < max_count):
        theta, delta, sigma, T_ISA, P_ISA, rho_ISA, _, a = atmosphere_ISA_deviation(
            altitude, delta_ISA)
        CL = ((2*load_factor)/(gamma*P_ISA*mach_climb*mach_climb)) * \
            (mass*GRAVITY/wing_surface)
        altitude = altitude+step

        count = count + 1

    return altitude

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# weight = 43112
# altitude = 10000
# limit_altitude = 41000
# mach_climb = 0.78
# print(buffet_altitude(weight, altitude, limit_altitude, mach_climb))
