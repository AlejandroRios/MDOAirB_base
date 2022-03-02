"""
MDOAirB

Description:
    - This module calculates the cruise performance using the Breguet equations

Reference:
    - 

TODO's:
    - Review and rename variables

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

def cruise_performance(altitude, delta_ISA, mach, mass, distance_cruise, vehicle):
    """
    Description:
        - This function sets the division of segments to perform the cruise calculation
    Inputs:
        - altitude
        - delta_ISA - ISA temperature deviation [deg C]
        - mach - mach number
        - mass - aircraft mass [kg]
        - distance_cruise - cruise distance [nm]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - time_cruise - time spent during cruise phase [min]
        - final_mass - aircraft mass at the end of cruise [kg]
    """
    n = 10
    step_cruise = distance_cruise/n
    distance = 0
    time_cruise = 0
    mass_fuel_cruise = 0

    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)

    for i in range(n):

        TSFC, L_over_D, fuel_flow, throttle_position = specific_fuel_consumption(
            vehicle, mach, altitude, delta_ISA, mass)

        mass_fuel, time = mission_segment(
            mass, step_cruise, L_over_D, TSFC, V_tas)

        time_cruise = time_cruise + time

        mass_fuel_cruise = mass_fuel_cruise + mass_fuel
        # print(mass_fuel)

    final_mass = mass - mass_fuel_cruise
    # print(final_mass)

    return time_cruise, final_mass

def cruise_performance_simple(altitude, delta_ISA, mach, mass, distance_cruise, vehicle):
    """
    Description:
        - This function calculates the cruise performance based on Breguet equations. 
    Inputs:
        - altitude - [ft]
        - delta_ISA - ISA temperature deviation [deg C]
        - mach - mach number
        - mass - aircraft mass [kg]
        - distance_cruise - cruise distance [nm]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - time_cruise - time spent during cruise phase [min]
        - final_mass - aircraft mass at the end of cruise [kg]
    """
    n = 4
    step_cruise = distance_cruise/n
    distance = 0
    time_cruise = []
    mass_fuel_cruise = []
    d = 0

    # if distance_cruise < 0:
    #     print('duhh')
    
    V_tas = mach_to_V_tas(mach, altitude, delta_ISA)

    for i in range(n):

        TSFC, L_over_D, fuel_flow, throttle_position = specific_fuel_consumption(
            vehicle, mach, altitude, delta_ISA, mass)

        mass_fuel, time = breguet_simple(altitude, delta_ISA, mach, L_over_D, TSFC,step_cruise,mass)
        # if time <0 or mass_fuel < 0:
        #     print(mass_fuel)
        #     print(time)
        mass_fuel_cruise.append(mass_fuel)
        time_cruise.append(time)
        d = d+step_cruise

    final_mass = mass - sum(mass_fuel_cruise)
    time_cruise = sum(time_cruise)

    return time_cruise, final_mass


def specific_fuel_consumption(vehicle, mach, altitude, delta_ISA, mass):
    """
    Description:
        - This function calculates the specific fuel consumption during cruise segment
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - mach - mach number
        - altitude - [ft]
        - delta_ISA - ISA temperature deviation [deg C]
        - mass - aircraft mass [kg]
    Outputs:
        - TSFC - thrust specific fuel consumption [(kg/hr)/N] 
        - L_over_D - lift to drag ratio
        - fuel_flow - engine fuel flow [kg/hr]
        - throttle_position - throttle position [1.0 = 100%]
    """
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


def mission_segment(mass_0, step_cruise, L_over_D, TSFC, V_tas):
    """
    Description:
        - This function calculates the mission segment fuel burn
    Inputs:
        - mass_0 - initial mass [kg]
        - step_cruise - number of segments in which the cruise phase is divided
        - L_over_D - lift over drag ratio 
        - TSFC - thrust specific fuel consumption [(kg/hr)/N] 
        - V_tas - true airspeed [kt]
    Outputs:
        - mass_fuel - fuel mass [kg]
        - time - time [min]
    """
    knots_to_meters_second = 0.514444
    second_to_miniute = 0.01667
    fixedW = mass_0  # [kg]
    R = step_cruise*1852  # convert 600 nmi to m [m]
    TSFC = TSFC*(1/3600)  # 1/s

    V = V_tas*knots_to_meters_second  # [m/s]
    segments = [breguet('jet', 'cruise', R, L_over_D, TSFC, V, False)]
    fuel_safety_margin = 0.06
    FF = (1+fuel_safety_margin)*missionfuelburn(segments)

    def EWfunc(w0): 
        return 3.03*w0**-0.235

    mass_0 = fuelfractionsizing(EWfunc, fixedW, FF, False, False)

    # print(FF)
    # print(mass_0)
    mass_fuel = FF*mass_0
    time = 1/TSFC * L_over_D * np.log(1/segments[0])
    return mass_fuel, time*second_to_miniute


def breguet(type, task, E_R_or_frac, LD, SFC, V, eta_p):
    """
    Description:
        - This function calculates the Breguet equation
    Inputs:
        - type - type of analysis
        - task - task analysis
        - E_R_or_frac
        - LD - lift over drag ratio
        - SFC - specific fuel consumption [(kg/hr)/N]
        - V - aircraft speed
        - eta_p
    Outputs:
        - varargout
    """
    if V == "False":
        V = 'NaN'

    varargout = [0]*2

    if type == 'jet' and task == 'loiter':
        varargout = np.exp(-E_R_or_frac*SFC/(LD))
    elif type == 'jet' and task == 'cruise':
        varargout = np.exp(-E_R_or_frac*SFC/(V*LD))
    elif type == 'prop' and task == 'loiter':
        varargout = np.exp(-E_R_or_frac*SFC*V/(LD*eta_p))
    elif type == 'prop' and task == 'cruise':
        varargout = np.exp(-E_R_or_frac*SFC/(LD*eta_p))
    elif type == 'jet' and task == 'range':
        varargout[0] = -LD*np.log(E_R_or_frac)/SFC
        varargout[1] = varargout[0]*V
    elif type == 'prop' and task == 'range':
        varargout[0] = -LD*eta_p*np.log(E_R_or_frac)/SFC
        varargout[1] = varargout[0]/V
    else:
        print('Unknown mission segment type and/or task string')
    
    return(varargout)

def breguet_simple(altitude, delta_ISA, mach, LD, SFC,step_cruise,W0):
    """
    Description:
        - This function calculates the simplified Breguet equation
    Inputs:
        - altitude - [ft]
        - delta_ISA - ISA temperature deviation [deg C]
        - mach - mach number
        - LD - lift over drag ratio
        - SFC - specific fuel consumption [(kg/hr)/N]
        - step_cruise 
        - W0 - initial weight [N]
    Outputs:
        - mass_fuel - fuel mass [kg]
        - time - time [min]
    """
    second_to_miniute = 0.01667
    _, _, _, _, _, _, _, a = atmosphere_ISA_deviation(altitude, delta_ISA)
    V=mach*a
    a1=SFC*(1/LD)
    a2=step_cruise*a1/V;
    f=np.exp(a2)
    Wf=W0/f
    mass_fuel=W0-Wf
    time=(1/a1)*np.log(W0/Wf)*60

    # time = LD/SFC *np.log(W0/Wf)
    # R = step_cruise*1852
    return float(mass_fuel), float(time)

def fuelfractionsizing(sf, fixedW, FF, tol, maxW):
    """
    Description:
        - This function calculates the fuel fraction
    Inputs:
        - sf
        - fixedW
        - FF
        - tol
        - maxW
    Outputs:
        - W0
    """

    if isfunction(sf) is not True:
        if len(sf) == 1:
            W0 = fixedW/(1-FF-sf[0])
            return
        elif len(sf) == 2:
            def sf(w0): return sf[0]*w0**(sf[1])
        elif len(sf) == 3:
            def sf(w0): return sf[2]*sf[0]*w0**(sf[1])
        else:
            print('invalid empty weight function sf')

    if isinstance(sf, (np.ndarray)) is True:
        FF = missionfuelburn(FF)

    minW = fixedW/(1-FF)

    # # default maxW represents an aircraft with a terrible fixedW fraction
    if not maxW:
        maxW = 800000/GRAVITY

    if not tol:
        tol = 1e-5*minW

    def f(W): 
        
        return 1 - sf(W) - FF - fixedW/W

    W0_init = np.mean([minW, maxW])
    # (0.015,0.2)
    # bounds = minW
    if minW > maxW:
        return
    

    bnds = ((minW,maxW))
    # res = minimize(f, W0_init, bounds=((minW,maxW),), tol=tol, options={'disp': False})
    # W0 = res.x[0]
    W0 =  optimize.newton(f, W0_init,tol=tol)
    # W0 = optimize.bisect(f, minW,maxW)

    # W0 = root(f,W0_init)
    # W0 = res

    # bounds = np.array([minW, maxW])
    # bounds = minW
    # W0 = fsolve(f, W0_init, xtol=tol)

    # print(W0)

    return(W0)


def missionfuelburn(varargin):
    """
    Description:
        - This function calculates misison fuel burn
    Inputs:
        - varargin
    Outputs:
        - FF
    """
    n = len(varargin)
    fracs = [0]*(n+1)
    fracs[0] = 1
    for ii in range(n):
        fracs[ii+1] = fracs[ii]*varargin[ii]
    FF = 1 - fracs[-1]
    return(FF)

# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================

# from framework.Database.Aircrafts.baseline_aircraft_parameters import *
# altitude = 39000
# delta_ISA = 30
# mach = 0.8
# mass = 50000
# distance_cruise = 1500
# # results = cruise_performance(altitude, delta_ISA, mach, mass, distance_cruise,vehicle)
# # print(results)

# # res = []
# # enes = []
# # for n in range(1,50):
# #     print(n)
# #     _,aux = cruise_performance(altitude, delta_ISA, mach, mass, distance_cruise,vehicle,n)
# #     res.append(float(aux))
# #     enes.append(n)

# # import matplotlib.pyplot as plt


# # print(res)
# # print(n)
# # plt.plot(enes, res)
# # plt.show()
# from datetime import datetime
# start_time = datetime.now()
# _,aux = cruise_performance(altitude, delta_ISA, mach, mass, distance_cruise,vehicle)
# end_time = datetime.now()
# print('Mission sizing execution time: {}'.format(end_time - start_time))



# from framework.Database.Aircrafts.baseline_aircraft_parameters import *

# performance = vehicle['performance']

# tolerance = 100

# aircraft = vehicle['aircraft']
# engine = vehicle['engine']
# wing = vehicle['wing']

# airport_departure = vehicle['airport_departure']
# airport_destination = vehicle['airport_destination']

# operations = vehicle['operations']
# performance = vehicle['performance']

# mission_range = 300
# aircraft['maximum_zero_fuel_weight'] = 31700
# aircraft['operational_empty_weight'] = 21800
# aircraft['passenger_capacity'] = 77
# aircraft['number_of_engines'] = 2
# engine['maximum_thrust'] = 63173
# operations['reference_load_factor'] = 0.85
# aircraft['maximum_takeoff_weight'] = 38790
# aircraft['maximum_landing_weight'] = 34100


# altitude = 33000
# delta_ISA = 0
# mach = 0.6133
# mass = 37569
# distance_cruise = 110
# from datetime import datetime
# start_time = datetime.now()

# time_cruise, final_mass = cruise_performance_simple(altitude, delta_ISA, mach, mass, distance_cruise, vehicle)
# end_time = datetime.now()
# print('objective function time: {}'.format(end_time - start_time))
# print(time_cruise)
# print(mass-final_mass)
