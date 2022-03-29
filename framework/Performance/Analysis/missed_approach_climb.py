
"""
MDOAirB

Description:
    - This function calculates the thrust to weight ratio following the requiremnts
      of missed climb approach with one-engine-inoperative accoring to FAR 25.121.
      For this case the climb gradient expressed as a percentage takes a value of 0.021 (for two engine aircraft).
      The lading gear is up and takeoff flaps are deployed

Reference:
    - References: FAR 25.121 and ROSKAM 1997 - Part 1, pag. 142

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


def missed_approach_climb_OEI(vehicle, airport_destination, maximum_takeoff_weight, weight_landing):
    """
    Description:
        - This function calculates the missed approach thrust to weight ratio with one enfine inoperative
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        -airport_destination
        - maximum_takeoff_weight - maximum takeoff weight[kg]
        - weight_landing - maximum landing weight [kg]
    Outputs:
        - thrust_to_weight_landing
    """
    ft_to_m = 0.3048
    kt_to_ms = 0.514444
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    maximum_landing_weight = weight_landing
    CL_maximum_landing = aircraft['CL_maximum_landing']/(1.3**2)
    wing_surface = wing['area']
    airfield_elevation = airport_destination['elevation']
    airfield_delta_ISA = airport_destination['tref']
    phase = 'climb'

    _, _, _, _, _, rho, _, a = atmosphere_ISA_deviation(
        airfield_elevation, airfield_delta_ISA)  # [kg/m3]

    V = 1.3*np.sqrt(2*maximum_landing_weight /
                    (CL_maximum_landing*wing_surface*rho))
    mach = V/a*kt_to_ms

    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1

    CD_wing, _ = aerodynamic_coefficients_ANN(
        vehicle, airfield_elevation*ft_to_m, mach, CL_maximum_landing,alpha_deg,switch_neural_network)

    friction_coefficient = wing['friction_coefficient']
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD_landing = CD_wing + CD_ubrige
    # CD_landing = zero_fidelity_drag_coefficient(aircraft_data, CL_maximum_landing, phase)

    L_to_D = CL_maximum_landing/CD_landing
    if aircraft['number_of_engines']  == 2:
        steady_gradient_of_climb = 0.021  # 2.4% for two engines airplane
    elif aircraft['number_of_engines']  == 3:
        steady_gradient_of_climb = 0.024  # 2.4% for two engines airplane
    elif aircraft['number_of_engines']  == 4:
        steady_gradient_of_climb = 0.027  # 2.4% for two engines airplane

    aux1 = (aircraft['number_of_engines'] /(aircraft['number_of_engines'] -1))
    aux2 = (1/L_to_D) + steady_gradient_of_climb
    aux3 = maximum_landing_weight/maximum_takeoff_weight
    thrust_to_weight_landing = aux1*aux2*aux3
    return thrust_to_weight_landing


def missed_approach_climb_AEO(vehicle, airport_destination, maximum_takeoff_weight, weight_landing):
    """
    Description:
        - This function calculates the missed approach thrust to weight ratio with all engines operative
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - maximum_takeoff_weight - maximum landing weight [kg]
        - weight_landing - landing weight [kg]
    Outputs:
        - thrust_to_weight_landing
    """
    ft_to_m = 0.3048
    kt_to_ms = 0.514444
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    maximum_landing_weight = weight_landing

    CL_maximum_landing = aircraft['CL_maximum_landing']
    wing_surface = wing['area']

    airfield_elevation = airport_destination ['elevation']
    airfield_delta_ISA = airport_destination ['tref']
    phase = 'descent'

    _, _, _, _, _, rho, _, a = atmosphere_ISA_deviation(
        airfield_elevation, airfield_delta_ISA)  # [kg/m3]
    V = 1.3*np.sqrt(2*maximum_landing_weight /
                    (CL_maximum_landing*wing_surface*rho))
    mach = V/a*kt_to_ms

    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 1
    CD_wing, _ = aerodynamic_coefficients_ANN(
        vehicle, airfield_elevation*ft_to_m, mach, CL_maximum_landing,alpha_deg,switch_neural_network)
    friction_coefficient = wing['friction_coefficient']
    CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

    CD_landing = CD_wing + CD_ubrige

    L_to_D = CL_maximum_landing/CD_landing

    steady_gradient_of_climb = 0.032  # 2.4% for two engines airplane

    aux1 = (1/L_to_D) + steady_gradient_of_climb
    aux2 = maximum_landing_weight/maximum_takeoff_weight
    thrust_to_weight_landing = aux1 * aux2
    return thrust_to_weight_landing
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
