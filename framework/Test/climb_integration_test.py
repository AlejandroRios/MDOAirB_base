"""
MDOAirB

Description:
    - This module calculates the aircraft performance during climb by integrating
        in time the point mass equations of movement. 

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
# from scipy.integrate import odeint
# from scipy.integrate import ode
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from framework.Attributes.Airspeed.airspeed import V_cas_to_mach, V_tas_to_V_cas, mach_to_V_cas, crossover_altitude
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Analysis.climb_acceleration import acceleration_to_250
from framework.Performance.Engine.engine_performance import turbofan
from framework.Performance.Analysis.climb_to_altitude import rate_of_climb_calculation

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
feet_to_nautical_miles = 0.000164579

def climb_integration(mass, mach_climb, climb_V_cas, delta_ISA, final_altitude, initial_altitude, vehicle):
    """
    Description:
        - This function calculates the aircraft performance during climb by integrating
        in time the point mass equations of movement. 
    Inputs:
        - initial mass [kg]
        - mach - mach number_climb
        - climb_V_cas - calibrated airspeed during climb [kt] [knots]
        - delta_ISA - ISA temperature deviation [deg C] [C deg]
        - final_altitude [ft]
        - initial_altitude [ft]
        - vehicle - dictionary containing aircraft parameters dictionary
    Outputs:
        - final_distance [ft]
        - total_climb_time [min]
        - total_burned_fuel [kg]
        - final_altitude [ft]
    """
    rate_of_climb = 500

    time_climb1 = 0
    time_climb2 = 0
    time_climb3 = 0

    transition_altitude = crossover_altitude(
        mach_climb, climb_V_cas, delta_ISA)

    time = 0
    distance = 0
    fuel1 = 0
    fuel2 = 0
    fuel3 = 0

    if final_altitude >= transition_altitude:
        flag1 = 1
        flag2 = 1
        flag3 = 1

    if (final_altitude >= 10000 and final_altitude < transition_altitude):
        flag1 = 1
        flag2 = 1
        flag3 = 0

    if final_altitude < 10000:
        flag1 = 1
        flag2 = 0
        flag3 = 0

    total_burned_fuel = []
    total_climb_time = []
    
    distance_vec = []
    altitude_vec = []
    mass_vec = []
    time_vec = []
    sfc_vec = []
    thrust_vec = []
    mach_vec = []
    CL_vec = []
    CD_vec = []
    LoD_vec = []
    throttle_vec = []
    vcas_vec = []

    throttle_position = 0.95

    fig, axs = plt.subplots(7)
    fig.suptitle('Vertically stacked subplots')

    fig1, axs1 = plt.subplots(3)
    fig1.suptitle('Vertically stacked subplots')

    axs[0].set_xlim(float(initial_altitude),float(final_altitude))
    axs[1].set_xlim(float(initial_altitude),float(final_altitude))
    axs[2].set_xlim(float(initial_altitude),float(final_altitude))
    axs[3].set_xlim(float(initial_altitude),float(final_altitude))
    axs[4].set_xlim(float(initial_altitude),float(final_altitude))
    axs[5].set_xlim(float(initial_altitude),float(final_altitude))
    axs[6].set_xlim(float(initial_altitude),float(final_altitude))

    axs1[0].set_xlim(initial_altitude,final_altitude)
    axs1[1].set_xlim(initial_altitude,final_altitude)
    axs1[2].set_xlim(initial_altitude,final_altitude)

    if flag1 == 1:

        # Climb to 10000 ft with 250 KCAS
        if final_altitude <= 11000:
            final_block_altitude = final_altitude
        else:
            final_block_altitude = 11000

        initial_block_distance = 0
        initial_block_altitude = initial_altitude
        initial_block_mass = mass
        initial_block_time = 0

        final_block_distance, final_block_altitude, final_block_mass, final_block_time, distance, altitude, mass, time, sfcv, thrustv, machv, CLv, CDv, LoDv, throttlev, vcasv = climb_integrator(
            initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, climb_V_cas, mach_climb, delta_ISA, vehicle,axs,axs1)

        distance_vec = np.append(distance_vec, distance)
        altitude_vec = np.append(altitude_vec, altitude) 
        mass_vec = np.append(mass_vec, mass) 
        time_vec = np.append(time_vec, time)
        
        sfc_vec = np.append(sfc_vec, sfcv)
        thrust_vec = np.append(thrust_vec, thrustv)
        mach_vec = np.append(mach_vec, machv)
        CL_vec = np.append(CL_vec, CLv)
        CD_vec = np.append(CD_vec, CDv)
        LoD_vec = np.append(LoD_vec, LoDv)
        throttle_vec = np.append(throttle_vec, throttlev)
        vcas_vec = np.append(vcas_vec, vcasv)

        # _, _, delta_altitude, _ = acceleration_to_250(
        #     rate_of_climb, climb_V_cas, delta_ISA, vehicle)
        # final_block_altitude = final_block_altitude + delta_altitude

        burned_fuel = initial_block_mass - final_block_mass
        climb_time = final_block_time - initial_block_time
        total_burned_fuel.append(burned_fuel)
        total_climb_time.append(climb_time)

    if flag2 == 1:

        initial_block_distance = final_block_distance
        initial_block_altitude = final_block_altitude
        initial_block_mass = final_block_mass
        initial_block_time = final_block_time

        if final_altitude <= transition_altitude:
            final_block_altitude = final_altitude
        else:
            final_block_altitude = transition_altitude

        final_block_distance, final_block_altitude, final_block_mass, final_block_time, distance, altitude, mass, time, sfcv, thrustv, machv, CLv, CDv, LoDv, throttlev, vcasv = climb_integrator(
            initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, climb_V_cas, mach_climb, delta_ISA, vehicle,axs,axs1)
        
        distance_vec = np.append(distance_vec, distance)
        altitude_vec = np.append(altitude_vec, altitude)
        mass_vec = np.append(mass_vec, mass)
        time_vec = np.append(time_vec, time)

        sfc_vec = np.append(sfc_vec, sfcv)
        thrust_vec = np.append(thrust_vec, thrustv)
        mach_vec = np.append(mach_vec, machv)
        CL_vec = np.append(CL_vec, CLv)
        CD_vec = np.append(CD_vec, CDv)
        LoD_vec = np.append(LoD_vec, LoDv)
        throttle_vec = np.append(throttle_vec, throttlev)
        vcas_vec = np.append(vcas_vec, vcasv)

        burned_fuel = initial_block_mass - final_block_mass
        climb_time = final_block_time - initial_block_time
        total_burned_fuel.append(burned_fuel)
        total_climb_time.append(climb_time)
        # plt.plot(time_interval, state[:, 1])

    if flag3 == 1:

        initial_block_distance = final_block_distance
        initial_block_altitude = final_block_altitude
        initial_block_mass = final_block_mass
        initial_block_time = final_block_time

        final_block_altitude = final_altitude

        final_block_distance, final_block_altitude, final_block_mass, final_block_time, distance, altitude, mass, time, sfcv, thrustv, machv, CLv, CDv, LoDv, throttlev, vcasv = climb_integrator(
            initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, 0, mach_climb, delta_ISA, vehicle,axs,axs1)
        
        distance_vec = np.append(distance_vec, distance)
        altitude_vec = np.append(altitude_vec, altitude)
        mass_vec = np.append(mass_vec, mass)
        time_vec = np.append(time_vec, time)

        sfc_vec = np.append(sfc_vec, sfcv)
        thrust_vec = np.append(thrust_vec, thrustv)
        mach_vec = np.append(mach_vec, machv)
        CL_vec = np.append(CL_vec, CLv)
        CD_vec = np.append(CD_vec, CDv)
        LoD_vec = np.append(LoD_vec, LoDv)
        throttle_vec = np.append(throttle_vec, throttlev)
        vcas_vec = np.append(vcas_vec, vcasv)
        
        burned_fuel = initial_block_mass - final_block_mass

        climb_time = final_block_time - initial_block_time
        total_burned_fuel.append(burned_fuel)
        total_climb_time.append(climb_time)

    final_altitude = final_block_altitude

    final_distance = final_block_distance
    total_burned_fuel = sum(total_burned_fuel)
    total_climb_time = sum(total_climb_time)
    
    fig.tight_layout()
    fig1.tight_layout()
    plt.show()

    plt.show()

    

    

    return final_distance, total_climb_time, total_burned_fuel, final_altitude, distance_vec, altitude_vec, mass_vec, time_vec, sfc_vec, thrust_vec, mach_vec, CL_vec, CD_vec, LoD_vec, throttle_vec, vcas_vec

def climb_integrator(initial_block_distance, initial_block_altitude, initial_block_mass, initial_block_time, final_block_altitude, climb_V_cas, mach_climb, delta_ISA, vehicle,axs,axs1):
    """
    Description:
        - This function sets the integration parameters. 
    Inputs:
        - initial_block_distance
        - initial_block_altitude
        - initial_block_mass
        - initial_block_time
        - final_block_altitude
        - climb_V_cas - calibrated airspeed during climb [kt]
        - mach - mach number_climb
        - delta_ISA - ISA temperature deviation [deg C]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - final_block_distance
        - final_block_altitude
        - final_block_mass
        - final_block_time
    """

    print('initial altitude i:',initial_block_altitude)
    print('final altitude i:',final_block_altitude)

    print('initial speed i:',climb_V_cas)
    print('initial mach i:',mach_climb)
    print('initial distance i:',initial_block_distance*feet_to_nautical_miles)
    print('initial mass i:',initial_block_mass)
    print('initial time i:',initial_block_time)

    print('------------------------------------------------------------')
    

    

    Tsim = initial_block_time + 100000
    stop_condition.terminal = True
    stop_criteria = final_block_altitude
    t_span = [initial_block_time, Tsim]
    t = np.arange(initial_block_time, Tsim, 0.01)
    sol = solve_ivp(climb, t_span, [initial_block_distance, initial_block_altitude, initial_block_mass],
            events = stop_condition, method='LSODA',args = (climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria),dense_output=True, rtol=1e-12,atol=1e-8,t_eval=t)


    distance0 = sol.y[0]
    altitude0= sol.y[1]
    mass0 = sol.y[2]
    time0 = sol.t

    y_event_value = sol.y_events[0][0][1]


    print('event altitude:',sol.y_events[0][0][1])

    distance  = distance0[(altitude0<=y_event_value)]
    altitude = altitude0[(altitude0<=y_event_value)]
    mass = mass0[(altitude0<=y_event_value)]
    time = time0[(altitude0<=y_event_value)]

    final_block_distance = distance[-1]
    final_block_altitude = altitude[-1]
    final_block_mass = mass[-1]
    final_block_time = time[-1]
    
    sfc_vec = []
    thrust_vec = []
    mach_vec = []
    CL_vec = []
    CD_vec = []
    LoD_vec = []
    throttle_vec = [] 
    vcas_vec = [] 
    RoC_vec = []
    
    for i in range(len(altitude)):
        sfc, thrust_force, mach, CL, CD, LoD, throttle, vcas, RoC = compute_flight_data(
            altitude[i], mass[i], climb_V_cas, mach_climb, delta_ISA, vehicle)
        sfc_vec = np.append(sfc_vec, sfc)
        thrust_vec = np.append(thrust_vec, thrust_force)
        mach_vec = np.append(mach_vec, mach)
        CL_vec = np.append(CL_vec, CL)
        CD_vec = np.append(CD_vec, CD)
        LoD_vec = np.append(LoD_vec, LoD)
        throttle_vec = np.append(throttle_vec, throttle)
        vcas_vec = np.append(vcas_vec, vcas)
        RoC_vec = np.append(RoC_vec,RoC)

    
    step_size =np.diff(altitude)

    time_step = np.zeros(len(altitude))
    time_step[1:] = np.cumsum(step_size/RoC_vec[1:]) + initial_block_time
    time_step[0] = initial_block_time

    # step_size =np.diff(altitude)

    # print(altitude)

    # print(step_size)
    # time_step = time_step-time_step[0]


    axs[0].plot(altitude, vcas_vec,alpha=0.5)
    axs[0].set_ylabel('Vcas')
    axs[1].plot(altitude, mach_vec,alpha=0.5)
    axs[1].set_ylabel('Mach')
    axs[2].plot(altitude, thrust_vec,alpha=0.5)
    axs[2].set_ylabel('Thrust')
    axs[3].plot(altitude, CL_vec,alpha=0.5)
    axs[3].set_ylabel('CL')
    axs[4].plot(altitude, CD_vec,alpha=0.5)
    axs[4].set_ylabel('CD')
    axs[5].plot(altitude, throttle_vec,alpha=0.5)
    axs[5].set_ylabel('Throt')
    axs[6].plot(altitude, RoC_vec,alpha=0.5)
    axs[6].set_ylabel('Roc')
    axs[6].set_xlabel('altitude')


    axs1[0].plot(altitude, distance*0.000164579,alpha=0.5)
    axs1[0].set_ylabel('distance')
    axs1[1].plot(altitude, mass,alpha=0.5)
    axs1[1].set_ylabel('mass')
    axs1[2].plot(altitude, time,alpha=0.5)
    axs1[2].set_ylabel('time')
    axs1[2].set_xlabel('altitude')

    plt.draw()
    
    print('initial altitude f:',initial_block_altitude)
    print('final altitude f:',final_block_altitude)

    print('final speed i:',climb_V_cas)
    print('final mach i:',mach_climb)
    print('final distance i:',final_block_distance*feet_to_nautical_miles)
    print('final mass i:',final_block_mass)
    print('final time i:',final_block_time)

    print('------------------------------------------------------------')



    
    return final_block_distance, final_block_altitude, final_block_mass, final_block_time, distance, altitude, mass, time, sfc_vec, thrust_vec, mach_vec, CL_vec, CD_vec, LoD_vec, throttle_vec, vcas_vec

def stop_condition(time, state, climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria):
    H = state[1]
    return 0 if H>stop_criteria else 1

def climb(time, state, climb_V_cas, mach_climb, delta_ISA, vehicle,stop_criteria):
    """
    Description:
        - This function uses the mass-point equations of motion to evaluate the sates in time. 
    Inputs:
        - time
        - state
        - climb_V_cas - calibrated airspeed during climb [kt]
        - mach - mach number_climb
        - delta_ISA - ISA temperature deviation [deg C]
        - vehicle - dictionary containing aircraft parameters
        - stop_criteria
    Outputs:
        - dout
    """
    aircraft = vehicle['aircraft']

    distance = state[0]
    altitude = state[1]
    mass = state[2]

    # if altitude > final_block_altitude:
    #     return
    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)
    throttle_position = 0.95

    if climb_V_cas > 0:
        mach = V_cas_to_mach(climb_V_cas, altitude, delta_ISA)
    else:
        mach = mach_climb

    thrust_force, fuel_flow, vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]

    thrust_to_weight = aircraft['number_of_engines'] * \
        thrust_force/(mass*GRAVITY)
    rate_of_climb, V_tas, climb_path_angle, _, _, _ = rate_of_climb_calculation(
        thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)
    # if rate_of_climb < 300:
    #     print('rate of climb violated!')

    x_dot = (V_tas*101.269)*np.cos(climb_path_angle)  # ft/min
    h_dot = (V_tas*101.269)*np.sin(climb_path_angle)  # ft/min
    W_dot = -2*fuel_flow*kghr_to_kgmin  # kg/min
    # time_dot = h_dot
    dout = [x_dot, h_dot, W_dot]
    return dout


def compute_flight_data(altitude, mass, climb_V_cas, mach_climb, delta_ISA, vehicle):
    """
    Description:
        - This function computes some date once the trajectory is solved 
    Inputs:
        - time
        - state
        - climb_V_cas - calibrated airspeed during climb [kt]
        - mach - mach number_climb
        - delta_ISA - ISA temperature deviation [deg C]
        - vehicle - dictionary containing aircraft parameters
        - stop_criteria
    Outputs:
        - dout
    """
    # if altitude > final_block_altitude:
    #     return
    aircraft = vehicle['aircraft']
    
    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(
        altitude, delta_ISA)
    throttle_position = 0.95

    if climb_V_cas > 0:
        mach = V_cas_to_mach(climb_V_cas, altitude, delta_ISA)
    else:
        mach = mach_climb

    thrust_force, fuel_flow, vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
    thrust_to_weight = aircraft['number_of_engines'] * \
        thrust_force/(mass*GRAVITY)
    rate_of_climb, V_tas, climb_path_angle, CL, CD, LoD = rate_of_climb_calculation(
        thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)  # thrust_to_weight = aircraft['number_of_engines'] * \ thrust_force/(mass*GRAVITY)
    #rate_of_climb, V_tas, climb_path_angle = rate_of_climb_calculation(
    #    thrust_to_weight, altitude, delta_ISA, mach, mass, vehicle)
    # if rate_of_climb < 300:
    #     print('rate of climb violated!')
    # time_dot = h_dot
    vcas = V_tas_to_V_cas(V_tas, altitude, delta_ISA)
    sfc = fuel_flow/(thrust_force/10) # sfc in kg/h/daN
    if sfc < 0:
        print('Climb phase')
        print('SFC = {}'.format(sfc))
        print('Fuel Flow = {}'.format(fuel_flow))
        print('Thrust = {}'.format(thrust_force))
        print('Altitude = {}, Mach = {}, Throttle position = {}'.format(altitude, mach, throttle_position))
    return sfc, thrust_force, mach, CL, CD, LoD, throttle_position, vcas, rate_of_climb
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================


from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
vehicle = initialize_aircraft_parameters()

mass = 90718
mach_climb = 0.78
climb_V_cas = 250
delta_ISA = 0
final_altitude = 40000
initial_altitude = 1500

aircraft = vehicle['aircraft']
wing = vehicle['wing']
winglet = vehicle['winglet']
horizontal_tail = vehicle['horizontal_tail']
vertical_tail = vehicle['vertical_tail']
fuselage = vehicle['fuselage']
engine = vehicle['engine']
pylon = vehicle['pylon']
nose_landing_gear = vehicle['nose_landing_gear']
main_landing_gear = vehicle['main_landing_gear']
performance = vehicle['performance']
operations = vehicle['operations']
airport_departure = vehicle['airport_departure']
airport_destination = vehicle['airport_destination']

x = [
    185,  #WingArea - 0
    78.2,  #AspectRatio x 10 - 1
    24.3,  #TaperRatio - 2
    25,  #sweep_c4 - 3
    -2.25,  #twist - 4
    38.5,  #semi_span_kink - 5
    43,  #BPR x 10 - 6
    18.82,  #FanDiameter X 10 - 7
    25.8,  #Compressor pressure ratio - 8
    1500,  #turbine inlet temperature - 9 
    15,  #FPR x 10 - 10
    239,  #PAX number - 11
    6,  #seat abreast - 12
    4000,  #range - 13
    37000,  #design point pressure - 14
    78,  #design point mach x 10 - 15
    1,
    1,
    1,
    1
]

wing['area'] = x[0]
wing['aspect_ratio'] = x[1]/10
wing['taper_ratio'] = x[2]/100
wing['sweep_c_4'] = x[3]
wing['twist'] = x[4]
wing['semi_span_kink'] = x[5]/100
aircraft['passenger_capacity'] = x[11]
fuselage['seat_abreast_number'] = x[12]
performance['range'] = x[13]
# aircraft['winglet_presence'] = x[17]
aircraft['winglet_presence'] = 1
# aircraft['slat_presence'] = x[18]
aircraft['slat_presence'] = 1
# horizontal_tail['position'] = x[19]
horizontal_tail['position'] = 1

engine['bypass'] = x[6]/10
engine['diameter'] = x[7]/10
engine['compressor_pressure_ratio'] = x[8]
engine['turbine_inlet_temperature'] = x[9]
engine['fan_pressure_ratio'] = x[10]/10

# engine['position'] = x[16]
engine['position'] = 1
engine['fan_diameter'] = engine['diameter']*0.98  # [m]


climb_integration(mass, mach_climb, climb_V_cas, delta_ISA, final_altitude, initial_altitude, vehicle)

