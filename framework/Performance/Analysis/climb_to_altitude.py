"""
MDOAirB

Description:
    - This function performs the calculation process to obtain the time, fuel
    anddistance for one altitude step og the step integration process

Reference:
    - Reference: Blake, BOEING CO. Flight Operations Engineering -
    Jet Transport Performance Methods. 7th ed. Boeing Co., Everett,
    Estados Unidos, 1989
    - Chapter 30, page 30-11

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
# from framework.Attributes.Atmosphere.atmosphere import atmosphere
from framework.Attributes.Airspeed.airspeed import mach_to_V_tas
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

def rate_of_climb_calculation(thrust_to_weight, h, delta_ISA, mach, mass, vehicle):
    """
    Description:
        - This function calculates the aircraft performance during climb by integrating
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
    """
    aircraft = vehicle['aircraft']  
    wing = vehicle['wing']   
    wing_surface = wing['area']

    knots_to_feet_minute = 101.268
    knots_to_meters_second = 0.514444
    ft_to_m = 0.3048

    phase = "climb"

    V_tas = mach_to_V_tas(mach, h, delta_ISA)

    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(h, delta_ISA)

    CL = (2*mass*GRAVITY) / \
        (rho_ISA*((V_tas*knots_to_meters_second)**2)*wing_surface)
    CL = float(CL)

    # CD = zero_fidelity_drag_coefficient(aircraft_data, CL, phase)
    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, _ = aerodynamic_coefficients_ANN(vehicle, h*ft_to_m, mach, CL,alpha_deg,switch_neural_network)
    
    friction_coefficient = wing['friction_coefficient']
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD = CD_wing + CD_ubrige

    L_to_D = CL/CD

    if mach > 0:
        acceleration_factor, _ = acceleration_factor_calculation(
            h, delta_ISA, mach)
        climb_path_angle = np.arcsin(
            (thrust_to_weight - 1/(L_to_D))/(1 + acceleration_factor))
    else:
        _, acceleration_factor = acceleration_factor_calculation(
            h, delta_ISA, mach)
        climb_path_angle = np.arcsin(
            (thrust_to_weight - 1/(L_to_D))/(1 + acceleration_factor))
    rate_of_climb = knots_to_feet_minute * V_tas * np.sin(climb_path_angle)
    return rate_of_climb, V_tas, climb_path_angle


def acceleration_factor_calculation(h, delta_ISA, mach):
    """
    Description:
        - This function calculates the acceleration factor
    Inputs:
        - h
        - delta_ISA - ISA temperature deviation [deg C]
        - mach - mach number
    Outputs:
        - acceleration factor
    """
    lambda_rate = 0.0019812
    tropopause = (71.5 + delta_ISA)/lambda_rate

    _, _, _, T, _, _, _, _ = atmosphere_ISA_deviation(0,0)
    _, _, _, T_ISA, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)

    if h < tropopause:
        # For constant calibrated airspeed below the tropopause:
        acceleration_factor_V_CAS = (
            0.7*mach**2)*(phi_factor(mach) - 0.190263*(T_ISA/T))
        # For constant Mach number below the tropopause:
        acceleration_factor_mach = (-0.13318*mach**2)*(T_ISA/T)
    elif h > tropopause:
        # For constant calibrated airspeed above the tropopause:
        acceleration_factor_V_CAS = (0.7*mach**2)*phi_factor(mach)
        # For constant Mach number above the tropopause:
        acceleration_factor_mach = 0

    return acceleration_factor_V_CAS, acceleration_factor_mach


def phi_factor(mach):
    aux1 = (1 + 0.2*mach**2)**3.5 - 1
    aux2 = (0.7*mach**2) * (1 + 0.2*mach**2)**2.5
    return aux1/aux2
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# aircraft_data = baseline_aircraft()
