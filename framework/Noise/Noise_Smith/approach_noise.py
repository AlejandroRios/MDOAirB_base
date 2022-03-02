"""
MDOAirB

Description:
    - This function calculates the noise during the approach

Reference:
    - SMITH, M.J.T - Aircraft Noise (1989)
    - ESDU77022 - Atmospheric properties

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
from framework.Noise.Noise_Smith.noise_airframe import noise_airframe

import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
deg_to_rad = np.pi/180
def approach_noise(time_vec,velocity_vec,distance_vec,altitude_vec,landing_parameters,noise_parameters,aircraft_geometry,vehicle):
    """
    Description:
        -This function calculates the noise during approach

    Inputs:
        - time_vec - vector containing time [s]
        - velocity_vec - vector containing speed [m/s]
        - distance_vec - vector containing distances [m]
        - altitude_vec - vector containing altitude [m]
        - landing_parameters - landing constant parameters
        - noise_parameters - noise constant parameters
        - aircraft_geometry - dictionary containing aircraft constant parameters
        - vehicle - dictionary containing aircraft parameters

    Outputs:
        - f - frequencies
        - OASPLhistory - noise history [dB]
        - tetaout - noise direction related to the receptor [deg]
        - time_vec - vectort containing time [s]
        - distance_vec - vector containing distances [m]
        - altitude_vec - vector containing altitude [m]
    """
    aircraft = vehicle['aircraft']

    XA = noise_parameters['takeoff_longitudinal_distance_mic']
    dlat = noise_parameters['takeoff_lateral_distance_mic'] 

    tetaout = []
    airframe_noise = []
    engine_noise = []
    SPL = []

    gamma = landing_parameters['gamma']


    for i in range(len(time_vec)-1):
        altitude = altitude_vec[i]
        XB = distance_vec[i]
        L1 = np.abs(XB-XA)
        R = np.sqrt(altitude**2 + L1**2 + dlat**2)
        termo1 = np.sqrt(((altitude - L1*np.tan(np.abs(gamma*deg_to_rad)))**2 + dlat**2)/(R**2))
        if XB > XA:
            theta = np.arcsin(termo1)/deg_to_rad
        elif XB == XA:
            theta = 90
        else:
            theta = 180-(np.arcsin(termo1)/deg_to_rad)
        
        vairp = velocity_vec[i]

        fi = np.arctan(altitude/dlat)/deg_to_rad
        if vairp == 0:
            vairp = 0.1
        
        Fphase = 2
        if altitude >= 100:
            aircraft_geometry['main_landing_gear_position'] = 2
            aircraft_geometry['main_landing_gear_position'] = 2
        else:
            aircraft_geometry['main_landing_gear_position'] = 1
            aircraft_geometry['main_landing_gear_position'] = 1

        f, SPLAC = noise_airframe(noise_parameters, aircraft_geometry, altitude, 0, theta, fi, R, Fphase, vairp, vehicle)

        f               = f
        tetaout.append(theta)
        airframe_noise.append(SPLAC)


    OASPLhistory        = np.asarray(airframe_noise)

    return f, OASPLhistory, tetaout, time_vec, distance_vec, altitude_vec
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
