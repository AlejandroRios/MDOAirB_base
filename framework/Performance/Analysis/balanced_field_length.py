"""
MDOAirB

Description:
    - Balanced length field module

Reference:
    - Reference: Torenbeek. 1982 and Gudmunsson 2014
    - Chapter 5, page 169, equation 5-91 and Chapter 17 equation 17-1

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def balanced_field_length(vehicle, airport_departure, weight_takeoff,gamma_2):
    '''
    Description:
        - This function performs the evaluation of the balanced field length

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - weight_takeoff - takeoff weight [N]
        - gamma_2 - second segment climb gradient

    Outputs:
        - balanced fiel length [m]
        
    Note: for project design the case of delta_gamma2 = 0 presents most
    interest, as the corresponding weight is limited by the second segment
    climb requirement (Torenbeek, 1982)
    '''

    wing = vehicle['wing']
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']


    # Aircraft data import
    CL_max_takeoff = aircraft['CL_maximum_takeoff']
    wing_surface = wing['area']  # [m2]
    T_avg = aircraft['average_thrust']  # [N]
    engines_number = aircraft['number_of_engines']

    # Airport data import
    airfield_elevation = airport_departure['elevation']  # [ft]
    delta_ISA = airport_departure['tref']  # [deg C]

    # horizontal distance from airfield surface requirement according to FAR 25 - [m]
    h_takeoff = 10.7
    if engines_number == 2:
        gamma2_min = 0.024
    elif engines_number ==3:
        gamma2_min = 0.027
    elif engines_number == 4:
        gamma2_min = 0.03

    delta_gamma2 = gamma_2-gamma2_min

    delta_S_takeoff = 200  # [m]
    g = 9.807  # [m/s2]
    CL_2 = 0.694*CL_max_takeoff
    mu = 0.01*CL_max_takeoff + 0.02
    _, _, sigma, _, _, rho, _, _ = atmosphere_ISA_deviation(
        airfield_elevation, delta_ISA)  # [kg/m3]

    aux1 = 0.863/(1 + (2.3*delta_gamma2))
    aux2 = ((weight_takeoff/wing_surface)/(rho*g*CL_2)) + h_takeoff

    # To high takeoff weight will made this coeficient less than mu resulting in a negative
    # landing field, when that happend this code will use twice the takeoff weight as coefficient
    # to continue with the iteration
    if T_avg/weight_takeoff > mu:
        aux3 = (1/((T_avg/weight_takeoff)-mu)) + 2.7
    else:
        aux3 = weight_takeoff*2

    aux4 = delta_S_takeoff/np.sqrt(sigma)

    return aux1*aux2*aux3 + aux4  # [m]
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
