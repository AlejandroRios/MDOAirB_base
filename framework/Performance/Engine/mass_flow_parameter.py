"""
MDOAirB

Description:
    - This module calculates the mass flow parameters. 

Reference:
    - Aircraft Engine Design, 2003,  Jack Mattingly, William H. Heiser, David T. Pratt

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
from framework.Attributes.Atmosphere.temperature_dependent_air_properties import FAIR
from scipy import optimize
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def MASSFP(Tt=None, f=None, M=None, ht=None, Prt=None, gammat=None, at=None):
    """
    Description:
        - This function calculates the mass flow parameters.
    Inputs:
        - Tt - Total temperature
        - f - fuel/air ratio
        - M - Mach number
        - ht - static enthalpy [Jkg]
        - Prt - relative pressure [pa]
        - gammat - ratio of specific heats
        - at - speed of sound [m/s]
    Outputs:
        - TtdT - temperature ratio
        - PtdP - pressure ratio
        - MFP - mass flow parameter
    """

    list_variables = [Tt, f, M, ht, Prt, gammat, at]
    nargin = sum(x is not None for x in list_variables)

    if nargin < 7:
        # in: T0 [K] f | out: ht [J/kg] Prt [Pa] R0 [J/KgK]
        _, ht, Prt, _, _, _, gammat, at = FAIR(1, f=f, T=Tt)

    global h_t, g_c, f_g, M_g
    h_t = ht
    g_c = 1             # [m/s]
    f_g = f
    M_g = M

    Vguess = (M*at)/(1+((gammat-1)/2)*M**2)

    optimize.fsolve(vals, Vguess)
    TtdT = Tt/T_g
    PtdP = Prt/Pr_g
    MFP = (M/PtdP)*np.sqrt((gamman_g*g_c)/R_g*TtdT)

    return TtdT, PtdP, MFP


def vals(V):

    global T_g, Pr_g, R_g, gamman_g, a_g
    h = h_t - V**2/(2*g_c)
    T_g, _, Pr_g, _, _, R_g, gamman_g, a_g = FAIR(2, f=f_g, h=h)

    Vn = M_g*a_g
    if V != 0:
        Verror = np.abs((V-Vn)/V)
    else:
        Verror = np.abs(V-Vn)
    return Verror


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# import time
# start_time = time.time()
# print(MASSFP(Tt=1.4662e3, f=0.0222, M=1))
# print("--- %s seconds ---" % (time.time() - start_time))
