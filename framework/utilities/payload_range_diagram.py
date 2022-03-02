"""
File name : Cruise performance function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : November/2020
Last edit : November/2020
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil
Description:
    - This module calculates the cruise performance using the Breguet equations
Inputs:
    - Cruise altitude [ft]
    - Delta ISA [C deg]
    - Mach number
    - Mass at top of climb
    - Cruise distance [mn]
    - Vehicle dictionary
Outputs:
    - Cruise time [min]
    - Mass at top of descent [kg]
TODO's:
    - Rename variables 
"""
# =============================================================================
# IMPORTS
# =============================================================================
from inspect import isfunction
import numpy as np
# from scipy.optimize import fsolve
# from scipy.optimize import minimize
from scipy import optimize
from scipy.optimize import root
from framework.Performance.Engine.engine_performance import turbofan
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Attributes.Airspeed.airspeed import V_cas_to_mach, mach_to_V_cas, mach_to_V_tas, crossover_altitude
# from framework.Aerodynamics.aerodynamic_coefficients import zero_fidelity_drag_coefficient
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665
ft_to_m = 0.3048
km_to_nm = 0.539957
def payload_range(mass,mass_fuel, vehicle):

    
    aircraft = vehicle['aircraft']
    operations = vehicle['operations']
    wing = vehicle['wing']
    Sw = wing['area']
    W0 = mass

    Wf = mass_fuel

    delta_ISA = operations['flight_planning_delta_ISA']

    altitude = operations['optimal_altitude_cruise']

    mach = operations['mach_cruise']

    # Initialize product of all phases
    Mf = 1 - Wf/W0
    Mf_cruise = Mf
        
    ### Landing and Taxi
    Mf_cruise = Mf_cruise/0.992
        
    ### Loiter
    # Loiter at max L/D
    TSFC, L_over_D, fuel_flow, throttle_position = specific_fuel_consumption(
            vehicle, mach, altitude, delta_ISA, mass)
    #print(LDmax)

    TSFC = TSFC*1/3600

    # Factor to fuel comsumption
    # C_loiter = C_cruise*0.4/0.5
    C_loiter = TSFC*0.4/0.5
    #print(C_loiter)
    
    ### Cruise 2 (already known value)
    Mf_cruise = Mf_cruise/0.9669584006325017

    loiter_time = 60 * 45
    
    # Continue Loiter
    Mf_cruise = Mf_cruise/np.exp(-loiter_time*C_loiter/L_over_D)

    ### Descent
    Mf_cruise = Mf_cruise/0.99

    ### Start and warm-up
    Mf_cruise = Mf_cruise/0.99

    ### Taxi
    Mf_cruise = Mf_cruise/0.99

    ### Take-off
    Mf_cruise = Mf_cruise/0.995

    ### Climb
    Mf_cruise = Mf_cruise/0.98
    
    ### Cruise
    # Atmospheric conditions at cruise altitude
    # T,p,rho,mi = dt.atmosphere(altitude_cruise, 288.15)
    _, _, _, T, _, rho, _, a = atmosphere_ISA_deviation(altitude, delta_ISA)

    # Cruise speed
    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)


    range_cruise = -(np.log(Mf_cruise)*V_tas*L_over_D)/TSFC



    return (range_cruise/1000)*km_to_nm


def specific_fuel_consumption(vehicle, mach, altitude, delta_ISA, mass):

    knots_to_meters_second = 0.514444

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    wing_surface = wing['area']

    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)
    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)

    CL_required = (2*mass*GRAVITY) / \
        (rho_ISA*((knots_to_meters_second*V_tas)**2)*wing_surface)
    # print('CL',CL_required)
    phase = 'cruise'
    # CD = zero_fidelity_drag_coefficient(aircraft_data, CL_required, phase)

    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, _ = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL_required, alpha_deg, switch_neural_network)

    friction_coefficient = 0.003
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD = CD_wing + CD_ubrige

    
    L_over_D = CL_required/CD
    throttle_position = 0.6

    thrust_force, fuel_flow , vehicle = turbofan(
        altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]

    FnR = mass*GRAVITY/L_over_D

    step_throttle = 0.01
    throttle_position = 0.6
    total_thrust_force = 0

    while (total_thrust_force < FnR and throttle_position <= 1):
        thrust_force, fuel_flow , vehicle = turbofan(
            altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
        TSFC = (fuel_flow*GRAVITY)/thrust_force
        total_thrust_force = aircraft['number_of_engines'] * thrust_force
        throttle_position = throttle_position+step_throttle

    L_over_D = CL_required/CD

    return TSFC, L_over_D, fuel_flow, throttle_position

# from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
# vehicle = initialize_aircraft_parameters()
# altitude = 40000
# delta_ISA = 0
# mass = 50000



# print(ranges(altitude, delta_ISA, mass, vehicle))