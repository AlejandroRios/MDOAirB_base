"""
MDOAirB

Description:
    - This module calculates the low pressure turbine parameters. 

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
from framework.Performance.Engine.mass_flow_parameter import MASSFP
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def TURB(Tti, f, AidAe, Mi, Me, eta_ti, TteR):
    """
    Description:
        - This function calculates the low pressure turbine parameters. 
    Inputs:
        - Tti - total temperature at station i
        - f - fuel/air ratio
        - AidAe - in/exit area ratio
        - Mi - entry mach number
        - Me - exist mach number 
        - eta_ti - engine termal efficiency
        - TteR - exit reference total temperature [K]
    Outputs:
        - pi_t - total pressure ratio low pressure turbine
        - Tau_t - total enthalpy ration low pressure turbine
        - T_te - total temperature at turbine exit
    """

    _, hti, Prti, phiti, cpti, Rti, gammati, ati = FAIR(item=1, f=f, T=Tti)
    Ti, _, MFPi = MASSFP(Tt=Tti, f=f, M=Mi)
    T_te = TteR
    while True:  # Label 1
        Te, Pe, MFPe = MASSFP(Tt=T_te, f=f, M=Me)
        _, hte, Prte, phite, cpte, Rte, gammate, ate = FAIR(
            item=1, f=f, T=T_te)

        # T, h, Pr, phi, Cp, R, gamma, a

        pi_t = (MFPi/MFPe)*AidAe*np.sqrt(T_te/Tti)
        Prtei = pi_t*Prti
        Ttei, htei, _, phitei, cptei, Rtei, gammatei, atei = FAIR(
            item=3, f=f, Pr=Prtei)
        hte = hti - eta_ti*(hti - htei)
        Tau_t = (hte/hti)
        Tten, _, Prte, phite, cpte, Rte, gammate, ate = FAIR(
            item=2, f=f, h=hte)
        Tte_error = np.abs(T_te - Tten)
        if Tte_error > 0.01:
            T_te = Tten
            continue  # Go to 1
        else:
            break
    return pi_t, Tau_t, T_te
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# print(TURB(1.0404e3, 0.0189, 0.4365, 1, 1, 0.8525, 984.0251))
