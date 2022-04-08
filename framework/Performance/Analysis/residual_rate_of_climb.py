"""
MDOAirB

Description:
    - This module calculates the residual rate of climb

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
# from framework.Aerodynamics.aerodynamic_coefficients import zero_fidelity_drag_coefficient
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
ft_to_m = 0.3048
kt_to_ms = 0.514444
def residual_rate_of_climb(vehicle, airport_departure, weight_takeoff,engine_cruise_thrust):
    """
    Description:
        - This function calculates the residual rate of climb
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - weight_takeoff - takeoff weight [N]
        - engine_cruise_thrust - [N]
    Outputs:
        - thrust_to_weight_residual
    """

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']

    CL_maximum_takeoff = aircraft['CL_maximum_takeoff']/(1.2**2)
    wing_surface = wing['area']
    maximum_takeoff_weight = weight_takeoff  # [N]

    airfield_elevation = airport_departure['elevation']
    airfield_delta_ISA = airport_departure['tref']
    
    thrust_takeoff = engine['maximum_thrust']

    _, _, _, _, _, rho, _, a = atmosphere_ISA_deviation(
        airfield_elevation, airfield_delta_ISA)  # [kg/m3]

    V = 1.2*np.sqrt(2*maximum_takeoff_weight /
                    (CL_maximum_takeoff*wing_surface*rho))
    mach = V/a*kt_to_ms
    phase = 'cruise'

    # CD_takeoff = zero_fidelity_drag_coefficient(aircraft_data, CL_maximum_takeoff, phase)
    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, _ = aerodynamic_coefficients_ANN(
        vehicle, airfield_elevation*ft_to_m, mach, CL_maximum_takeoff,alpha_deg,switch_neural_network)

    friction_coefficient = wing['friction_coefficient']
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD_takeoff = CD_wing + CD_ubrige

    L_to_D = CL_maximum_takeoff/CD_takeoff
    if aircraft['number_of_engines']  == 2:
        steady_gradient_of_climb = 0.024  # 2.4% for two engines airplane
    elif aircraft['number_of_engines']  == 3:
        steady_gradient_of_climb = 0.027  # 2.4% for two engines airplane
    elif aircraft['number_of_engines']  == 4:
        steady_gradient_of_climb = 0.03  # 2.4% for two engines airplane


    thrust_to_weight_residual = 1/((engine_cruise_thrust/thrust_takeoff)*L_to_D)

    return thrust_to_weight_residual

def residual_rate_of_climb2(vehicle, airport_departure, weight_takeoff,engine_TO_thrust):
    """
    Description:
        - This function calculates the residual rate of climb
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - weight_takeoff - takeoff weight [N]
        - engine_cruise_thrust - [N]
    Outputs:
        - thrust_to_weight_residual
    """

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']
    
    V_Vmd = np.sqrt(np.sqrt(3))
    CL_CLmd = 1/(V_Vmd**2)

    M_cruise = 0.85
    f_taper = 0.005*(1 + 1.5*(wing['taper_ratio']- 0.6)**2)
    e = 1/(1 + 0.12*M_cruise**6)/(1 + (0.142 + wing['aspect_ratio']*(10*wing['mean_thickness'])**0.33*f_taper)/np.cos(wing['sweep_c_4']*np.pi/180)**2 + 0.1*(3*aircraft['number_of_engines'] + 1)/(4 + wing['aspect_ratio'])**0.8)

    # CD_takeoff = zero_fidelity_drag_coefficient(aircraft_data, CL_maximum_takeoff, phase)
    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, CL_wing = aerodynamic_coefficients_ANN(
        vehicle, 30000*ft_to_m, M_cruise , 0, alpha_deg,switch_neural_network)

    CL_md = np.sqrt(CD_wing*np.pi*wing['aspect_ratio']*e)

    CL = CL_CLmd/CL_md

    k_E = 15.8

    Swet_Sw = 6.1
    Emax =  k_E *np.sqrt(wing['aspect_ratio']/Swet_Sw)
    E = Emax *2/(1/CL_CLmd+CL_CLmd)

    Tcr_Tto = weight_takeoff/(engine_TO_thrust*E)

    thrust_to_weight_residual = 1/(Tcr_Tto*E)    


    return thrust_to_weight_residual
    
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
