"""
File name : Descent to altitude function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : November/2020
Last edit : November/2020
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    - This module calculates the aircraft performance during descent by integrating
    in time the point mass equations of movement. 
Inputs:
    - initial mass [kg]
    - mach - mach number_climb
    - climb_V_cas - calibrated airspeed during climb [kt]
    - delta_ISA - ISA temperature deviation [deg C] [C deg]
    - final_altitude [ft]
    - initial_altitude [ft]
    - vehicle - dictionary containing aircraft parameters dictionary
Outputs:
    - final_distance [ft]
    - total_climb_time [min]
    - total_burned_fuel [kg]
    - final_altitude [ft]
TODO's:
    - Include a better description of this module

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Airspeed.airspeed import V_cas_to_mach, mach_to_V_cas, crossover_altitude
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Engine.engine_performance import turbofan
from framework.Performance.Analysis.descent_deceleration import decelaration_to_250
from framework.Performance.Analysis.descent_to_altitude import rate_of_descent_calculation

import numpy as np
# from scipy.integrate import odeint
# from scipy.integrate import ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.8067
kghr_to_kgmin = 0.01667
kghr_to_kgsec = 0.000277778


def descent_integration(mass, mach_descent, descent_V_cas, delta_ISA, altitude_vec, speed_vec, mach_vec, initial_altitude, vehicle):
    
    rate_of_descent = -500

    transition_altitude = crossover_altitude(
    mach_descent, descent_V_cas, delta_ISA)


    initial_block_distance = 0
    initial_block_altitude = initial_altitude
    initial_block_mass = mass
    initial_block_time = 0
    
    distance_vec = []
    time_vec = []
    mass_vec = []
    
    for i in range(len(altitude_vec)-1):

        descent_V_cas = (speed_vec[i+1] + speed_vec[i])/2
        mach_descent = (mach_vec[i+1] + mach_vec[i])/2
        if mach_descent <= 0.3:
            mach_descent = 0.78


        initial_block_altitude = altitude_vec[i]
        final_block_altitude = altitude_vec[i+1]

        distance_vec.append(initial_block_distance)
        mass_vec.append(initial_block_mass)
        time_vec.append(initial_block_time)


        if initial_block_altitude <= transition_altitude:
            final_block_distance, final_block_altitude, final_block_mass, final_block_time = climb_integrator(
                        initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, descent_V_cas, 0, delta_ISA, vehicle)
        else:
            final_block_distance, final_block_altitude, final_block_mass, final_block_time = climb_integrator(
                initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, 0, mach_descent, delta_ISA, vehicle)

        initial_block_distance = final_block_distance
        initial_block_altitude = final_block_altitude
        initial_block_mass = final_block_mass
        initial_block_time = final_block_time

    final_distance = distance_vec[-1] 
    total_descent_time = time_vec[-1]
    total_burned_fuel = mass_vec[0] - mass_vec[-1]
    final_altitude = altitude_vec[-1]

    return final_distance, total_descent_time, total_burned_fuel, final_altitude


def climb_integrator(initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, climb_V_cas, mach_climb, delta_ISA, vehicle):

    Tsim = initial_block_time + 40
    stop_condition.terminal = True

    stop_criteria = final_block_altitude

    sol = solve_ivp(climb, [initial_block_time, Tsim], [initial_block_distance, initial_block_altitude, initial_block_mass],
            events = stop_condition,method='LSODA',args = (climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria))

    distance = sol.y[0]
    altitude = sol.y[1]
    mass = sol.y[2]
    time = sol.t

    final_block_distance = distance[-1]
    final_block_altitude = altitude[-1]
    final_block_mass = mass[-1]
    final_block_time = time[-1]
    return final_block_distance, final_block_altitude, final_block_mass, final_block_time

def stop_condition(time, state, climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria):
    H = state[1]
    return 0 if H<stop_criteria else 1


def climb(time, state, climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria):

    aircraft = vehicle['aircraft']
    distance = state[0]
    altitude = state[1]
    mass = state[2]

    # print('t',time)
    # print('alt',altitude)

    # print('current:',altitude)
    # print('limit altitude:',final_block_altitude )
    # if altitude < final_block_altitude:
    #     return

    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)
    throttle_position = 0.3

    if climb_V_cas > 0:
        mach = V_cas_to_mach(climb_V_cas, altitude, delta_ISA)
    else:
        mach = mach_climb

    thrust_force, fuel_flow , vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]

    total_thrust_force = thrust_force*aircraft['number_of_engines']
    total_fuel_flow = fuel_flow*aircraft['number_of_engines']
    step_throttle = 0.01

    while (total_fuel_flow < 0 and throttle_position <= 1):
        thrust_force, fuel_flow , vehicle = turbofan(
            altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
        TSFC = (fuel_flow*GRAVITY)/thrust_force
        total_fuel_flow = aircraft['number_of_engines'] * fuel_flow
        throttle_position = throttle_position+step_throttle

    
    thrust_to_weight = aircraft['number_of_engines'] * \
        thrust_force/(mass*GRAVITY)
    
    # if thrust_to_weight < 0.001:
    #     return

    # print('T to W:',thrust_to_weight)
    rate_of_climb, V_tas, climb_path_angle = rate_of_descent_calculation(
        thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)

    x_dot = (V_tas*101.269)*np.cos(climb_path_angle)  # ft/min
    h_dot = (V_tas*101.269)*np.sin(climb_path_angle)  # ft/min
    # if (altitude < 10000 and altitude > 1500):
    #     h_dot = -500
    W_dot = -2*fuel_flow*kghr_to_kgmin  # kg/min
    # time_dot =  h_dot
    dout = [x_dot, h_dot, W_dot]

    return dout
# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================
# mass = 43112
# mach_climb = 0.78
# climb_V_cas = 280
# delta_ISA = 0
# final_altitude = 39000
# initial_altitude = 0
# print(climb_integration(mass, mach_climb, climb_V_cas, delta_ISA, final_altitude, initial_altitude))
# print(state)
