"""
MDOAirB

Description:
    - This module performs the simulation of takeoff profile which outputs are used
    to evaluate noise

Reference:
    - 

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
import numpy as np

from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Engine.engine_performance import turbofan
from framework.Noise.Noise_Smith.takeoff_integration import takeoff_integration
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.81
kt_to_ms = 0.514444

def thrust_equation_coefficients(V,T):
    """
    Description:
        - This function calculates the engine coefficientes to evaluate the thrust force as function of speed
    Inputs:
        - V
        - T
    Outputs:
        - T0
        - T1
        - T2
    """

    A = T[0]
    B = (T[1] - T[0])/(V[1]-V[0])
    C = ((T[2]-T[1])/(V[2]-V[1])-(T[1]-T[0])/(V[1]-V[0]))/(V[2]-V[0])
    T0 = T[0] - B*V[0] + C*V[0]*V[1]
    T1 = B - C*(V[0]+V[1])
    T2 = C
    return T0, T1, T2

def takeoff_profile(takeoff_parameters,landing_parameters,aircraft_parameters,runaway_parameters,engine_parameters,vehicle):
    """
    Description:
        - Performs the simulation of the takeoff and provides vectors containing the output information
    Inputs:
        - takeoff_parameters - takeoff constant parameters
        - landing_parameters
        - aircraft_parameters
        - runaway_parameters
        - engine_parameters - engine constant parameters
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - time_vec - vector containing time [s]
        - velocity_vec - vector containing speed [m/s]
        - distance_vec - vector containing distances [m]
        - velocity_horizontal_vec - vector containing horizontal speed [m/s]
        - altitude_vec - vector containing altitude [m]
        - velocity_vertical_vec - vector containing horizontal speed [m/s]
        - trajectory_angle_vec - vector containing the trajectory angle [deg]
        - fan_rotation_vec - vector containing the fan rotation [rpm]
        - compressor_rotation_vec - vector containing the compressor rotation speed [rpm]
    """
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']
    _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(0, 0)

    # Initial calculations:
    V_rotation = takeoff_parameters['k1'] * np.sqrt((2*takeoff_parameters['takeoff_weight'])/(rho_ISA*wing['area']*aircraft['CL_maximum_takeoff']))
    V_2 = takeoff_parameters['k2'] * np.sqrt((2*takeoff_parameters['takeoff_weight'])/(rho_ISA*wing['area']*aircraft['CL_maximum_takeoff']))
    V_35 = V_2 + 10/1.9438
    V_vector = np.array([0, (V_2+20)/2, V_2+20])
    mach_vector = V_vector/(a*kt_to_ms)

    # thrust_vector,_ = zip(aircraft['number_of_engines']*turbofan(0, mach_vector[i], 1, vehicle) for i in mach_vector)
    thrust_vector = []
    for i in mach_vector:
        thrust,_ , vehicle= turbofan(0, i, 1, vehicle)
        thrust = aircraft['number_of_engines']*thrust
        thrust_vector.append(thrust)
    
    T0,T1,T2 = thrust_equation_coefficients(V_vector,thrust_vector)

    engine['T0'] = T0
    engine['T1'] = T1
    engine['T2'] = T2

    # Calculating the All Engines Operative takeoff

    time_vector = []
    altitude_vector = []

    # Runaway run

    # ----- Run until VR -----
    initial_block_altitude = 0 
    initial_block_distance = 0
    initial_block_trajectory_angle = 0
    initial_block_time = 0
    initial_block_velocity = 0
    initial_block_horizontal_velocity = 0
    initial_block_vertical_velocity = 0
    initial_fan_rotation = engine['fan_rotation']
    initial_compressor_rotation = engine['compressor_rotation']

    stop_criteria = V_rotation
    i2 = 1
    error = 1e-6


    phase = 'ground'
    (final_block_altitude,
    final_block_distance,
    final_block_trajectory_angle,
    final_block_time,
    final_block_velocity,
    final_block_horizontal_velocity,
    final_block_vertical_velocity,
    final_fan_rotation,
    final_compressor_rotation,
    time_vec1,
    velocity_vec1,
    distance_vec1,
    velocity_horizontal_vec1,
    altitude_vec1,
    velocity_vertical_vec1,
    trajectory_angle_vec1,
    fan_rotation_vec1,
    compressor_rotation_vec1) = takeoff_integration(
        initial_block_altitude,
        initial_block_distance,
        initial_block_trajectory_angle,
        initial_block_time,
        initial_block_velocity,
        initial_block_horizontal_velocity,
        initial_block_vertical_velocity,
        initial_fan_rotation,
        initial_compressor_rotation,
        aircraft_parameters,
        takeoff_parameters,
        runaway_parameters,
        landing_parameters,
        vehicle,
        rho_ISA,
        stop_criteria,
        phase
        )

    # ----- Run until V35 -----
    initial_block_altitude = 0 
    initial_block_distance = final_block_distance
    initial_block_trajectory_angle = 0
    initial_block_time = final_block_time
    initial_block_velocity = final_block_velocity
    initial_block_horizontal_velocity = 0
    initial_block_vertical_velocity = 0
    initial_fan_rotation = engine['fan_rotation']
    initial_compressor_rotation = engine['compressor_rotation']
    
    stop_criteria = V_35

    phase = 'ground'

    (final_block_altitude,
    final_block_distance,
    final_block_trajectory_angle,
    final_block_time,
    final_block_velocity,
    final_block_horizontal_velocity,
    final_block_vertical_velocity,
    final_fan_rotation,
    final_compressor_rotation,
    time_vec2,
    velocity_vec2,
    distance_vec2,
    velocity_horizontal_vec2,
    altitude_vec2,
    velocity_vertical_vec2,
    trajectory_angle_vec2,
    fan_rotation_vec2,
    compressor_rotation_vec2) = takeoff_integration(
        initial_block_altitude,
        initial_block_distance,
        initial_block_trajectory_angle,
        initial_block_time,
        initial_block_velocity,
        initial_block_horizontal_velocity,
        initial_block_vertical_velocity,
        initial_fan_rotation,
        initial_compressor_rotation,
        aircraft_parameters,
        takeoff_parameters,
        runaway_parameters,
        landing_parameters,
        vehicle,
        rho_ISA,
        stop_criteria,
        phase
        )

    # ----- Flare to 35 ft -----
    initial_block_altitude = 0 
    initial_block_distance = final_block_distance
    initial_block_trajectory_angle = 0
    initial_block_time = final_block_time
    initial_block_velocity = final_block_velocity
    initial_block_horizontal_velocity = 0
    initial_block_vertical_velocity = 0
    initial_fan_rotation = engine['fan_rotation']
    initial_compressor_rotation = engine['compressor_rotation']

    stop_criteria = 35

    phase = 'flare'

    (final_block_altitude,
    final_block_distance,
    final_block_trajectory_angle,
    final_block_time,
    final_block_velocity,
    final_block_horizontal_velocity,
    final_block_vertical_velocity,
    final_fan_rotation,
    final_compressor_rotation,
    time_vec3,
    velocity_vec3,
    distance_vec3,
    velocity_horizontal_vec3,
    altitude_vec3,
    velocity_vertical_vec3,
    trajectory_angle_vec3,
    fan_rotation_vec3,
    compressor_rotation_vec3) = takeoff_integration(
        initial_block_altitude,
        initial_block_distance,
        initial_block_trajectory_angle,
        initial_block_time,
        initial_block_velocity,
        initial_block_horizontal_velocity,
        initial_block_vertical_velocity,
        initial_fan_rotation,
        initial_compressor_rotation,
        aircraft_parameters,
        takeoff_parameters,
        runaway_parameters,
        landing_parameters,
        vehicle,
        rho_ISA,
        stop_criteria,
        phase
        )

    # ----- Flare to 2000 ft -----

    initial_block_altitude = final_block_altitude 
    initial_block_distance = final_block_distance
    initial_block_trajectory_angle = final_block_trajectory_angle
    initial_block_time = final_block_time
    initial_block_velocity = final_block_velocity
    initial_block_horizontal_velocity = final_block_horizontal_velocity
    initial_block_vertical_velocity = final_block_vertical_velocity
    initial_fan_rotation =final_fan_rotation
    initial_compressor_rotation = final_compressor_rotation

    stop_criteria = 2000

    phase = 'climb'

    (final_block_altitude,
    final_block_distance,
    final_block_trajectory_angle,
    final_block_time,
    final_block_velocity,
    final_block_horizontal_velocity,
    final_block_vertical_velocity,
    final_fan_rotation,
    final_compressor_rotation,
    time_vec4,
    velocity_vec4,
    distance_vec4,
    velocity_horizontal_vec4,
    altitude_vec4,
    velocity_vertical_vec4,
    trajectory_angle_vec4,
    fan_rotation_vec4,
    compressor_rotation_vec4) = takeoff_integration(
        initial_block_altitude,
        initial_block_distance,
        initial_block_trajectory_angle,
        initial_block_time,
        initial_block_velocity,
        initial_block_horizontal_velocity,
        initial_block_vertical_velocity,
        initial_fan_rotation,
        initial_compressor_rotation,
        aircraft_parameters,
        takeoff_parameters,
        runaway_parameters,
        landing_parameters,
        vehicle,
        rho_ISA,
        stop_criteria,
        phase
        )

    time_vec = np.concatenate((time_vec1,time_vec2,time_vec3,time_vec4),axis=0)
    velocity_vec = np.concatenate((velocity_vec1,velocity_vec2,velocity_vec3,velocity_vec4),axis=0)
    distance_vec = np.concatenate((distance_vec1,distance_vec2,distance_vec3,distance_vec4),axis=0)
    velocity_horizontal_vec = np.concatenate((velocity_horizontal_vec1,velocity_horizontal_vec2,velocity_horizontal_vec3,velocity_horizontal_vec4),axis=0)
    altitude_vec = np.concatenate((altitude_vec1,altitude_vec2,altitude_vec3,altitude_vec4),axis=0)
    velocity_vertical_vec = np.concatenate((velocity_vertical_vec1,velocity_vertical_vec2,velocity_vertical_vec3,velocity_vertical_vec4),axis=0)
    trajectory_angle_vec = np.concatenate((trajectory_angle_vec1,trajectory_angle_vec2,trajectory_angle_vec3,trajectory_angle_vec4),axis=0)
    fan_rotation_vec = np.concatenate((fan_rotation_vec1,fan_rotation_vec2,fan_rotation_vec3,fan_rotation_vec4),axis=0)
    compressor_rotation_vec = np.concatenate((compressor_rotation_vec1,compressor_rotation_vec2,compressor_rotation_vec3,compressor_rotation_vec4),axis=0)

    return time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
