"""
MDOAirB

Description:
    - International Standard Atmosphere (ISA) based in:

Reference: 
    - Blake, BOEING CO. Flight Operations Engineering - Jet
    Transport Performance Methods. 7th ed. Boeing Co., Everett,
    Estados Unidos, 1989
    - Chapter 4, page 4-1

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def atmosphere_ISA_deviation(h, delta_ISA):
    """
    Description:
        - Obtains the atmosphere properties considering ISA deviation
    Inputs:
        - mach - mach number
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - Calibated airspeed [knots]
    """

    h = h  # ft
    h1 = 11  # Troposphere max altitude[km]
    L0 = -6.5e-3
    T0 = 288.15  # Reference altitude at sea level [K]
    p0 = 1.01325e5  # Reference pressure at sea level [Pa]
    rho0 = 1.2250  # Reference density at sea level[kg/m3]
    mi0 = 18.27E-06
    Tzero=291.15
    Ceh= 120 # C = Sutherland's constant for the gaseous material in question
    T1 = T0+L0*h1*1e3  # Temperature at troposphere limit

    lambda_rate = 0.0019812  # Temperature lapse rate - decrease of deg C for increasing 1 ft
    C1 = 5.25588
    C2 = 0.22336
    C3 = 36089.24
    C4 = 20805.7

    # Troposphere altitude correction considering delta ISA
    tropopause = 36089.24


    if h <= tropopause:
        # at or below Troposphere:
        theta = (T0 - (lambda_rate*h) + delta_ISA)/T0  # Temperature ratio
        delta = ((T0 - lambda_rate*h)/T0)**C1  # Pressure ratio
    elif h > tropopause:
        # above Troposphere:
        theta = (T1 + delta_ISA)/T0  # Temperature ratio
        delta = C2*np.exp((C3 - h)/C4)  # Pressure ratio

    sigma = delta/theta  # desity ratio

    a = 661.4786*np.sqrt(theta)  # [kts]

    T_ISA = theta*T0  # Temperature ISA [K]
    P_ISA = delta*p0  # Pressure ISA [Pa]
    rho_ISA = sigma*rho0  # Desnsity ISA [Kg/m^3]

    viscosity_ISA = mi0*((T_ISA+Ceh)/(Tzero+Ceh))*((T_ISA/Tzero)**1.5)

    return theta, delta, sigma, T_ISA, P_ISA, rho_ISA, viscosity_ISA, a
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# h = 32808.4
# delta_ISA = 0
# # delta_ISA = airport_departure['tref']
# print(atmosphere_ISA_deviation(h, delta_ISA))
