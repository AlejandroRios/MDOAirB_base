"""
MDOAirB

Description:
    - This module computes the reserve fuel simplified calculation.


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
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def reserve_fuel(landing_weight, alternative_airport_distance, holding_time, delta_ISA):
    """
    Description:
        - This function computes the reserve fuel simplified calculation.
 
    Inputs:
        - landing_weight - [kg]
        - alternative_airport_distance - [nm]
        - holding_time - [min]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - fuel - [kg]
    """

    reference_fuel_fraction = 1000
    reference_weight = 22000
    reference_mach = 0.78  # Mach 0.78 @ FL330

    _, _, _, _, _, rho, _, a = atmosphere_ISA_deviation(
        33000, delta_ISA)  # [kg/m3]
    reference_V_tas = reference_mach*a
    fuel_fraction_holding = 0.8*(reference_fuel_fraction/60)*(landing_weight/reference_weight)
    sr_alternative = (reference_V_tas/reference_fuel_fraction)*(reference_weight/landing_weight)
    alternative_fuel = alternative_airport_distance/sr_alternative
    holding_time = fuel_fraction_holding*holding_time
    fuel = alternative_fuel+holding_time
    return fuel
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
