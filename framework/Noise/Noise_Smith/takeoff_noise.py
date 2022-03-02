"""
MDOAirB

Description:
    - This function calculates the noise during takeoff

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
from framework.Noise.Noise_Smith.noise_engine import noise_engine
import numpy as np

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
deg_to_rad = np.pi/180
def takeoff_noise(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec, throttle_position, takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle):
    """
    Description:
        -This function calculates the noise during takeoff

    Inputs:
        - time_vec - vector containing time [s]
        - velocity_vec - vector containing speed [m/s]
        - distance_vec - vector containing distances [m]
        - velocity_horizontal_vec - vector containing horizontal speed [m/s]
        - altitude_vec - vector containing altitude [m]
        - velocity_vertical_vec - vector containing horizontal speed [m/s]
        - trajectory_angle_vec - vector containing the trajectory angle [deg]
        - fan_rotation_vec - vector containing the fan rotation [rpm]
        - compressor_rotation_vec - vector containing the compressor rotation speed [rpm]
        - throttle_position - throttle position [1.0 = 100%]
        - takeoff_parameters - takeoff constant parameters
        - noise_parameters - noise constant parameters
        - aircraft_geometry - dictionary containing aircraft constant parameters
        - engine_parameters - engine constant parameters
        - vehicle - dictionary containing aircraft parameters

    Outputs:
        - f
        - OASPLhistory
        - tetaout
        - time_vec - vector containing time [s]
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

    for i in range(len(time_vec)):
        altitude = altitude_vec[i]
        XB = distance_vec[i]
        gamma = trajectory_angle_vec[i]
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
        N1 = fan_rotation_vec[i]
        N2 = compressor_rotation_vec[i]

        fi = np.arctan(altitude/dlat)/deg_to_rad
        if vairp == 0:
            vairp = 0.1
        
        Fphase = 1
        if altitude >= 100:
            aircraft_geometry['main_landing_gear_position'] = 2
            aircraft_geometry['main_landing_gear_position'] = 2
        else:
            aircraft_geometry['main_landing_gear_position'] = 1
            aircraft_geometry['main_landing_gear_position'] = 1

        f, SPLAC = noise_airframe(noise_parameters, aircraft_geometry, altitude, 0, theta, fi, R, Fphase, vairp, vehicle)
        ft, OASPLENG = noise_engine(noise_parameters,aircraft_geometry,altitude,0,theta,fi,R,1.0,N1,N2,vairp,vehicle)

        f               = f
        tetaout.append(theta)
        airframe_noise.append(SPLAC)
        engine_noise.append(OASPLENG)

        # print(airframe[i-1])
        SPL_aux       = 10*np.log10(10**(0.1*airframe_noise[i-1])+aircraft['number_of_engines']*10**(0.1*engine_noise[i-1]))
        SPL.append(SPL_aux.T)

    OASPLhistory        = np.asarray(SPL)

    return f, OASPLhistory, tetaout, time_vec, distance_vec, altitude_vec
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
