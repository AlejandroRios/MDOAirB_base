"""
MDOAirB

Description:
    - This module performs speed transformations.

Reference: 
    - Reference: Gudmundsson, General Aviation Aircraft Design: Applied Methods
    and Procedures, 2013
    - pag 770
    - Blake, BOEING CO. Flight Operations Engineering -
    Jet Transport Performance Methods. 7th ed. Boeing Co., Everett,
    Estados Unidos, 1989
    - Chapter 6, page 6-12
    - Chapter 30, page 30-2

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def mach_to_V_cas(mach, h, delta_ISA):
    """
    Description:
        - Converts mach number to Calibrated AirSpeed
    Inputs:
        - mach - mach number
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - Calibated airspeed [knots]
    """
    _, delta, _, _, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)

    speed_of_sound = 661.4786  # sea level [knots]
    aux1 = ((0.2 * (mach**2) + 1)**3.5) - 1
    aux2 = (delta*aux1 + 1)**(1/3.5)
    return speed_of_sound * np.sqrt(5*(aux2 - 1))


def mach_to_V_tas(mach, h, delta_ISA):
    """
    Description:
        -  Converts mach number to True Air Speed
    Inputs:
        - mach - mach number
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - True airspeed [knots]
    """
    theta, _, _, _, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)
    speed_of_sound = 661.4786  # sea level [knots]
    return speed_of_sound * mach * np.sqrt(theta)


def V_cas_to_V_tas(V_cas, h, delta_ISA):
    """
    Description:
        -  Converts Calibrated Air Speed to True Air Speed
    Inputs:
        - V_cas - calibrated airspeed [kt]
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - True airspeed [kt]
    """
    speed_of_sound = 661.4786  # sea level [knots]
    theta, delta, _, _, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)
    aux1 = (1 + 0.2 * (V_cas/speed_of_sound)**2)**3.5
    aux2 = ((1/delta)*(aux1 - 1) + 1)**(1/3.5)
    aux3 = np.sqrt(theta*(aux2 - 1))
    return 1479.1 * aux3

def V_tas_to_V_cas(V_tas, h, delta_ISA):
    """
    Description:
        -  Converts True Air Speed to Calibrated Air Speed
    Inputs:
        - V_tas - true airspeed [kt]
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - Calibrated airspeed [kt]
    """
    speed_of_sound = 661.4786  # sea level [knots]
    theta, delta, _, _, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)

    aux1 = (1 + (1/theta)*(V_tas/1479.1)**2)**3.5
    aux2 = (delta*(aux1-1)+1)**(1/3.5)
    aux3 = 1479.1*np.sqrt(aux2-1)

    return aux3


def V_cas_to_mach(V_cas, h, delta_ISA):
    """
    Description:
        - Converts calibrated air speed to mach
    Inputs:
        - V_cas - calibrated airspeed [kt]
        - h - altitude [ft]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - mach - mach number
    """
    _, delta, _, _, _, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)
    speed_of_sound = 661.4786  # sea level [knots]
    aux1 = ((1 + 0.2*((V_cas/speed_of_sound)**2))**3.5) - 1
    aux2 = ((1/delta)*aux1 + 1)**((1.4-1)/1.4)
    return np.sqrt(5 * (aux2-1))


def crossover_altitude(mach, V_cas, delta_ISA):
    """
    Description:
        - Calculates the transition or crossosver altitude. The
        altitude at which a specified CAS and Mach value represent the
        same TAS value.
        The curves for constant CAS and constant Mach intersect at this
        point. Above this altitude the Mach number is used to reference speeds.
    Inputs:
        - mach - mach number
        - V_cas - calibrated airspeed [kt]
        - delta_ISA - ISA temperature deviation [deg C]
    Outputs:
        - crossover_altitude [ft]
    """
    flag = 0
    h = 0
    while flag <= 0:
        M1 = V_cas_to_mach(V_cas, h, delta_ISA)
        if M1 >= mach:
            flag = 1
            crossover_altitude = h
        h = (h + 100)
    return crossover_altitude
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# print(V_cas_to_V_tas(200, 10000, 0))
# print(crossover_altitude(0.75, 340, 0))

# print(V_tas_to_V_cas(328.19232653, 10900.10822041, 0))