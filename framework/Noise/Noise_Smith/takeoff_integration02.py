"""
File name :
Authors   : 
Email     : aarc.88@gmail.com
Date      : 
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
Inputs:
    -
Outputs:
    -
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Engine.engine_performance import turbofan

from scipy.integrate import ode
import numpy as np
import math
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

    aircraft = vehicle['aircraft']

    if phase == 'ground':
        t0 = initial_block_time
        z0 = [initial_block_distance, initial_block_velocity]
        solver = ode(ground)
        solver.set_integrator('vode', nsteps=1000)
        solver.set_f_params(takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle)
        solver.set_initial_value(z0, t0)
        t1 = initial_block_time + 200
        # N = 50
        t = np.linspace(t0, t1,100)
        N = len(t)
        sol = np.empty((N,2))
        sol_var = np.empty((N,6))
        sol[0] = z0
        times = np.empty((N, 1))


    if phase == 'flare':
        t0 = initial_block_time
        z0 = [initial_block_distance, initial_block_velocity, initial_block_altitude, initial_block_vertical_velocity, initial_block_trajectory_angle]
        solver = ode(flare)
        solver.set_integrator('vode', nsteps=1000)
        solver.set_f_params(aircraft_parameters, takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle)
        solver.set_initial_value(z0, t0)
        t1 = initial_block_time + 200
        # N = 50
        t = np.linspace(t0, t1,100)
        N = len(t)
        sol = np.empty((N,5))
        sol_var = np.empty((N,3))
        sol[0] = z0
        times = np.empty((N, 1))


    k = 1

    if phase == 'ground':
        while solver.successful() and solver.y[1] <= stop_criteria:
            solver.integrate(t[k])
            sol[k] = solver.y
            times[k] = solver.t
            
            _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(0, 0)
            mach = solver.y[1]/(a*kt_to_ms)
            thrust_force, fuel_flow, vehicle = turbofan(
                0, mach, 1, vehicle)

            engine = vehicle['engine']

            sol_var[k,0] = sol[k,1]
            sol_var[k,1] = 0
            sol_var[k,2] = 0
            sol_var[k,3] = 0
            sol_var[k,4] = engine['fan_rotation']
            sol_var[k,5] = engine['compressor_rotation']

            k += 1

        time_vec = times[1:k]
        velocity_vec = sol[1:k, 1]
        distance_vec = sol[1:k, 0]
        velocity_horizontal_vec = sol_var[1:k,0]
        altitude_vec = sol_var[1:k,1]
        velocity_vertical_vec = sol_var[1:k,2]
        trajectory_angle_vec = sol_var[1:k,3]
        fan_rotation_vec = sol_var[1:k,4]
        compressor_rotation_vec = sol_var[1:k,5]


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
        while solver.successful() and solver.y[2] <= stop_criteria:
            solver.integrate(t[k])
            sol[k] = solver.y
            times[k] = solver.t

            _, _, _, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(0, 0)
            mach = solver.y[1]/(a*kt_to_ms)
            thrust_force, fuel_flow, vehicle = turbofan(
                0, mach, 1, vehicle)
            engine = vehicle['engine']

            sol_var[k,0] = np.sqrt(sol[k,1]**2 + sol[k,3]**2)
            sol_var[k,1] = engine['fan_rotation']
            sol_var[k,2] = engine['compressor_rotation']

            k += 1

        time_vec = times[1:k]
        velocity_vec = sol_var[1:k,0]
        distance_vec = sol[1:k, 0]
        velocity_horizontal_vec = sol[1:k, 1]
        altitude_vec = sol[1:k, 2]
        velocity_vertical_vec = sol[1:k, 3]
        trajectory_angle_vec = sol[1:k, 4]
        fan_rotation_vec = sol_var[1:k,1]
        compressor_rotation_vec = sol_var[1:k,2]


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

        thrust_force, fuel_flow, vehicle = turbofan(
            initial_block_altitude, mach, 1, vehicle)
        
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
            thrust_force, fuel_flow, vehicle = turbofan(
            altitude, mach, 1, vehicle)

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
        
        time_vec = np.asarray([time_vec]).T

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
    
def ground(time,state,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle):
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

def flare(time,state,aircraft_parameters,takeoff_parameters,runaway_parameters,landing_parameters,rho_ISA,vehicle):
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

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
    thrust_force, fuel_flow, vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    
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

def climb(time, state, climb_V_cas, mach_climb, delta_ISA, final_block_altitude, vehicle):
    aircraft = vehicle['aircraft']

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

    thrust_force, fuel_flow , vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    thrust_to_weight = aircraft['number_of_engines'] * \
        thrust_force/(mass*GRAVITY)
    rate_of_climb, V_tas, climb_path_angle = rate_of_climb_calculation(
        thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)
    # if rate_of_climb < 300:
    #     print('rate of climb violated!')

    V_tas = V_tas*kt_to_ms

    rate_of_climb = rate_of_climb*0.0166667

    x_dot = (V_tas*101.269)*np.cos(climb_path_angle)  # ft/min
    h_dot = rate_of_climb  # ft/min
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
