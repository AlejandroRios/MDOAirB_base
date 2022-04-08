"""
MDOAirB

Description:
    - This module perform a numerical integration to simulate the takeoff phase

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
import math
import warnings

import numpy as np
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import \
    atmosphere_ISA_deviation
from framework.Performance.Engine.engine_performance import turbofan
from framework.Performance.Analysis.climb_to_altitude import rate_of_climb_calculation
from framework.Attributes.Airspeed.airspeed import V_cas_to_mach, mach_to_V_cas, crossover_altitude

from joblib import dump, load
from scipy.integrate import ode, solve_ivp

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.81
kt_to_ms = 0.514444



def takeoff_integration(
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
    ):
    """
    Description:
        - This function calculates the airplane sideline effective percibed noise during takeoff

    Inputs:
        - initial_block_altitude
        - initial_block_distance
        - initial_block_trajectory_angle
        - initial_block_time
        - initial_block_velocity
        - initial_block_horizontal_velocity
        - initial_block_vertical_velocity
        - initial_fan_rotation
        - initial_compressor_rotation
        - aircraft_parameters
        - takeoff_parameters - takeoff constant parameters
        - runaway_parameters
        - landing_parameters
        - vehicle - dictionary containing aircraft parameters
        - rho_ISA
        - stop_criteria
        - phase

    Outputs:
        - final_block_altitude
        - final_block_distance
        - final_block_trajectory_angle
        - final_block_time
        - final_block_velocity
        - final_block_horizontal_velocity
        - final_block_vertical_velocity
        - final_fan_rotation
        - final_compressor_rotation
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
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')

    if phase == 'ground':
        Tsim = initial_block_time + 200
        stop_condition_ground.terminal = True
        sol = solve_ivp(ground, [initial_block_time, Tsim], [initial_block_distance, initial_block_velocity],
                events = stop_condition_ground,rtol = 1e-10, method='LSODA',args = (takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria), dense_output=True, min_step = 1e-10)
    

    if phase == 'flare':
        Tsim = initial_block_time + 200
        stop_condition_flare.terminal = True
        sol = solve_ivp(flare, [initial_block_time, Tsim], [initial_block_distance, initial_block_velocity, initial_block_altitude, initial_block_vertical_velocity, initial_block_trajectory_angle],
                events = stop_condition_flare,rtol = 1e-10, method='LSODA',args = (aircraft_parameters,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria), dense_output=True,min_step = 1e-10)
    

    if phase == 'ground':

        time_vec = sol.t
        N = len(time_vec)
        # sol_var = np.zeros((N,6))

        distance_vec = sol.y[0]
        velocity_vec = sol.y[1]
        velocity_horizontal_vec = sol.y[1]
        altitude_vec = np.zeros(N)
        velocity_vertical_vec = np.zeros(N)
        trajectory_angle_vec = np.zeros(N)

        fan_rotation = np.zeros(N)
        compressor_rotation = np.zeros(N)
        for i in range(N):
            _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(0, 0)
            mach_aux = velocity_vec[i]/a*kt_to_ms

            if engine['type'] == 0:
                thrust_force, fuel_flow, vehicle = turbofan(
                    0, mach_aux, 1, vehicle)

                engine = vehicle['engine']
                fan_rotation[i] = engine['fan_rotation']
                compressor_rotation[i] = engine['compressor_rotation']
            else:
                thrust_force = nn_unit_F.predict(scaler_F.transform([(0, mach_aux, 1)]))
                fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(0, mach_aux, 1)]))

        fan_rotation_vec = fan_rotation
        compressor_rotation_vec = compressor_rotation


        final_block_time = time_vec[-1]
        final_block_velocity = velocity_vec[-1]
        final_block_distance = distance_vec[-1]
        final_block_horizontal_velocity = velocity_vec[-1]
        final_block_altitude = altitude_vec[-1]
        final_block_vertical_velocity =  velocity_vertical_vec[-1]
        final_block_trajectory_angle = trajectory_angle_vec[-1]
        final_fan_rotation = fan_rotation_vec[-1]
        final_compressor_rotation = compressor_rotation_vec[-1]
    
    elif phase == 'flare':
        
        time_vec = sol.t
        N = len(time_vec)

        velocity_vec = np.sqrt(sol.y[1]**2 + sol.y[3]**2)
        distance_vec = sol.y[0]
        velocity_horizontal_vec = sol.y[1]
        altitude_vec = sol.y[2]
        velocity_vertical_vec = sol.y[3]
        trajectory_angle_vec = sol.y[4]

        fan_rotation = np.zeros(N)
        compressor_rotation = np.zeros(N)
        for i in range(N):
            _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(0, 0)
            mach_aux = velocity_vec[i]/a*kt_to_ms

            if engine['type'] == 0:
                thrust_force, fuel_flow, vehicle = turbofan(
                    0, mach_aux, 1, vehicle)

                engine = vehicle['engine']
                fan_rotation[i] = engine['fan_rotation']
                compressor_rotation[i] = engine['compressor_rotation']
            else:
                thrust_force = nn_unit_F.predict(scaler_F.transform([(0, mach_aux, 1)]))
                fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(0, mach_aux, 1)]))

        fan_rotation_vec = fan_rotation
        compressor_rotation_vec = compressor_rotation


        final_block_time = time_vec[-1]
        final_block_velocity = velocity_vec[-1]
        final_block_distance = distance_vec[-1]
        final_block_horizontal_velocity = velocity_vec[-1]
        final_block_altitude = altitude_vec[-1]
        final_block_vertical_velocity =  velocity_vertical_vec[-1]
        final_block_trajectory_angle = trajectory_angle_vec[-1]
        final_fan_rotation = fan_rotation_vec[-1]
        final_compressor_rotation = compressor_rotation_vec[-1]


    elif phase == 'climb':

        if math.floor(initial_block_time) == math.ceil(initial_block_time + takeoff_parameters['time_step']):
            time = math.floor(initial_block_time) + takeoff_parameters['time_step']
        else:
            time = math.ceil(initial_block_time)

        V_35 = np.sqrt(initial_block_horizontal_velocity**2 + initial_block_vertical_velocity**2)
        _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(initial_block_altitude, 0)
        mach =V_35/(a*kt_to_ms)

        if engine['type'] == 0:
            thrust_force, fuel_flow, vehicle = turbofan(
                initial_block_altitude, mach, 1, vehicle)
        else:
            thrust_force = nn_unit_F.predict(scaler_F.transform([(initial_block_altitude, mach, 1)]))
            fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(initial_block_altitude, mach, 1)]))

        
        engine = vehicle['engine']

        total_thrust_force = thrust_force*aircraft['number_of_engines']

        if initial_block_altitude <= 100:
            gamma = np.arctan(total_thrust_force/((aircraft['maximum_takeoff_weight']*GRAVITY) - (aircraft_parameters['CD_air_LG_down']/aircraft_parameters['CL_air'])))
        else:
            gamma = np.arctan(total_thrust_force/((aircraft['maximum_takeoff_weight']*GRAVITY) - (aircraft_parameters['CD_air_LG_up']/aircraft_parameters['CL_air'])))
        
        rate_of_climb = V_35*np.sin(gamma)
        delta_altitude = rate_of_climb*takeoff_parameters['time_step']
        delta_distance = V_35*takeoff_parameters['time_step']*np.cos(gamma)
        

        time_vec = [time]
        velocity_vec = [V_35]
        distance_vec = [float(initial_block_distance+delta_distance)]
        velocity_horizontal_vec = [float(V_35*np.cos(gamma))]
        altitude_vec = [float(initial_block_altitude+delta_altitude)]
        velocity_vertical_vec = [float(rate_of_climb)]
        trajectory_angle_vec = [float(gamma*(180/np.pi))]
        fan_rotation_vec = [float(engine['fan_rotation'])]
        compressor_rotation_vec = [float(engine['compressor_rotation'])]
 
        iteration = 0
        for i in range(1,1000):

            time = time + takeoff_parameters['time_step']
            distance = distance_vec[-1]
            altitude = altitude_vec[-1]

            iteration = iteration + 1

            _, _, _, _, _, _, _, a = atmosphere_ISA_deviation(altitude, 0)

            mach = velocity_vec[-1]/(a*kt_to_ms)


            if engine['type'] == 0:
                thrust_force, fuel_flow, vehicle = turbofan(
                altitude, mach, 1, vehicle)
            else:
                thrust_force = nn_unit_F.predict(scaler_F.transform([(altitude, mach, 1)]))
                fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(altitude, mach, 1)]))

            engine = vehicle['engine']

            total_thrust_force = thrust_force*aircraft['number_of_engines']
            
            N1 = engine['fan_rotation']
            N2 = engine['compressor_rotation']


            if initial_block_altitude <= 100:
                gamma = np.arctan(total_thrust_force/((aircraft['maximum_takeoff_weight']*GRAVITY) - (aircraft_parameters['CD_air_LG_down']/aircraft_parameters['CL_air'])))
            else:
                gamma = np.arctan(total_thrust_force/((aircraft['maximum_takeoff_weight']*GRAVITY) - (aircraft_parameters['CD_air_LG_up']/aircraft_parameters['CL_air'])))
            rate_of_climb = V_35*np.sin(gamma)
            
            delta_altitude = rate_of_climb*takeoff_parameters['time_step']
            delta_distance = V_35*takeoff_parameters['time_step']*np.cos(gamma)

            time_vec.append(time)
            velocity_vec.append(V_35)
            distance_vec.append(float(distance+delta_distance))
            velocity_horizontal_vec.append(float(V_35*np.cos(gamma)))
            altitude_vec.append(float(altitude+delta_altitude))
            velocity_vertical_vec.append(float(rate_of_climb))
            trajectory_angle_vec.append(float(gamma*(180/np.pi)))
            fan_rotation_vec.append(float(N1))
            compressor_rotation_vec.append(float(N2))

            if distance_vec[-1] >= 10000:
                break
        
        time_vec = time_vec

        final_block_time = time_vec[-1]
        final_block_velocity = velocity_vec[-1]
        final_block_distance = distance_vec[-1]
        final_block_horizontal_velocity = velocity_vec[-1]
        final_block_altitude = altitude_vec[-1]
        final_block_vertical_velocity =  velocity_vertical_vec[-1]
        final_block_trajectory_angle = trajectory_angle_vec[-1]
        final_fan_rotation = fan_rotation_vec[-1]
        final_compressor_rotation = compressor_rotation_vec[-1]


    return (final_block_altitude,
    final_block_distance,
    final_block_trajectory_angle,
    final_block_time,
    final_block_velocity,
    final_block_horizontal_velocity,
    final_block_vertical_velocity,
    final_fan_rotation,
    final_compressor_rotation,
    time_vec,
    velocity_vec,
    distance_vec,
    velocity_horizontal_vec,
    altitude_vec,
    velocity_vertical_vec,
    trajectory_angle_vec,
    fan_rotation_vec,
    compressor_rotation_vec)
    
def ground(time,state,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria):
    """
    Description:
        - This function sets the integration for the ground run

    Inputs:
        - time
        - state
        - takeoff_parameters - takeoff constant parameters
        - runaway_parameters
        - landing_parameters
        - rho_ISA
        - vehicle - dictionary containing aircraft parameters
        - stop_criteria

    Outputs:
        - dout
    """
    warnings.filterwarnings('ignore')
    wing = vehicle['wing']
    engine = vehicle['engine']

    distance = state[0]
    velocity = state[1]

    constant = np.cos(takeoff_parameters['lambda']) + runaway_parameters['mu_roll']*np.sin(takeoff_parameters['lambda'])
    K0 = engine['T0']*constant - runaway_parameters['mu_roll']*takeoff_parameters['takeoff_weight']
    K1 = engine['T1']*constant
    K2 = engine['T2']*constant + 0.5*rho_ISA*wing['area']*(runaway_parameters['mu_roll']*landing_parameters['CL_3P'] - landing_parameters['CD_3P'])
    
    # Cálculo das derivadas
    x_dot = velocity
    V_dot = (GRAVITY/takeoff_parameters['takeoff_weight']) * (K0 + K1*velocity + K2*velocity*velocity)
    

    dout = np.asarray([x_dot, V_dot])
    dout = dout.reshape(2, )
    return dout

def stop_condition_ground(time,state,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria):
    V = state[1]
    return 0 if V>stop_criteria else 1

def flare(time,state,aircraft_parameters,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria):
    """
    Description:
        - This function sets the integration for the flare

    Inputs:
        - time
        - state
        - aircraft_parameters
        - takeoff_parameters - takeoff constant parameters
        - runaway_parameters
        - landing_parameters
        - rho_ISA
        - vehicle - dictionary containing aircraft parameters
        - stop_criteria

    Outputs:
        - dout
    """
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')

    distance = state[0]
    velocity_horizontal = state[1]
    altitude = state[2]
    velocity_vertical = state[3]
    trajectory_angle = state[4]

    V_resultant = np.sqrt(velocity_vertical**2 + velocity_horizontal**2)
    gamma = trajectory_angle

    _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(altitude, 0)

    mach = V_resultant/(a*kt_to_ms)
    throttle_position = 1.0

    if engine['type'] == 0:
        thrust_force, fuel_flow, vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    else:
        thrust_force = nn_unit_F.predict(scaler_F.transform([(altitude, mach, throttle_position)]))
        fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(altitude, mach, throttle_position)]))
    
    total_thrust_force = thrust_force*aircraft['number_of_engines']

    drag = 0.5*rho_ISA*(V_resultant**2)*wing['area']*aircraft_parameters['CD_air_LG_down']
    lift = 0.5*rho_ISA*(V_resultant**2)*wing['area']*aircraft_parameters['CL_air']

    velocity_horizontal = V_resultant*np.cos(gamma)
    parameter2_dot = GRAVITY/(aircraft['maximum_takeoff_weight']*GRAVITY) * (total_thrust_force - drag - aircraft['maximum_takeoff_weight']*np.sin(gamma))*np.cos(gamma) - GRAVITY*(1.3 - np.cos(gamma))*np.sin(gamma)
    velocity_vertical = V_resultant*np.sin(gamma)
    parameter4_dot = GRAVITY*(1.3-np.cos(gamma))*np.cos(gamma) + (GRAVITY/aircraft['maximum_takeoff_weight'])*(total_thrust_force - drag - aircraft['maximum_takeoff_weight']*np.sin(gamma))*np.sin(gamma)
    gamma_dot = (GRAVITY/V_resultant)*(1.3 - np.cos(gamma))


    dout = np.asarray([velocity_horizontal, parameter2_dot, velocity_vertical,  parameter4_dot, gamma_dot])
    dout = dout.reshape(5, )
    return dout
def stop_condition_flare(time,state,aircraft_parameters,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle,stop_criteria):
    H = state[2]
    return 0 if H>stop_criteria else 1

def climb(time, state, climb_V_cas, mach_climb, delta_ISA, final_block_altitude, vehicle):
    """
    Description:
        - This function sets the integration for the climb phase

    Inputs:
        - time
        - state
        - climb_V_cas - calibrated airspeed during climb [kt]
        - mach - mach number_climb
        - delta_ISA - ISA temperature deviation [deg C]
        - final_block_altitude
        - vehicle - dictionary containing aircraft parameters

    Outputs:
        - dout
    """
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')

    distance = state[0]
    altitude = state[1]

    mass = aircraft['maximum_takeoff_weight']

    if altitude > final_block_altitude:
        return
        
    _, _, _, _, _, rho_ISA, _, _  = atmosphere_ISA_deviation(altitude, delta_ISA)
    throttle_position = 1.0

    if climb_V_cas > 0:
        mach = V_cas_to_mach(climb_V_cas, altitude, delta_ISA)
    else:
        mach = mach_climb

    if engine['type'] == 0:
        thrust_force, fuel_flow , vehicle = turbofan(
            altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    else:
        thrust_force = nn_unit_F.predict(scaler_F.transform([(altitude, mach, throttle_position)]))
        fuel_flow = nn_unit_FC.predict(scaler_FC.transform([(altitude, mach, throttle_position)]))
    
    thrust_to_weight = aircraft['number_of_engines'] * \
        thrust_force/(mass*GRAVITY)
    rate_of_climb, V_tas, climb_path_angle = rate_of_climb_calculation(
        thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)
    # if rate_of_climb < 300:
    #     print('rate of climb violated!')

    V_tas = V_tas*kt_to_ms

    rate_of_climb = rate_of_climb*0.0166667

    x_dot = (V_tas*101.269)*np.cos(climb_path_angle)  # ft/min
    h_dot = (V_tas*101.269)*np.sin(climb_path_angle)  # ft/min
    # W_dot = -2*fuel_flow*kghr_to_kgmin  # kg/min
    time_dot = h_dot*0.0166667
    # dout = np.asarray([x_dot, h_dot, W_dot])
    dout = np.asarray([x_dot, h_dot])
    dout = dout.reshape(2, )

    return dout
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
