"""
MDOAirB

Description:
    - This module calculates the compressible flow parameters. 

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


def RGCOMPR(item=None, Tt=None, M=None, f=None, TtdT=None, PtdP=None, MFP=None):
    """
    Description:
        - This function calculates the compressible flow parameters. 
    Inputs:
        - item - computation mode
        - Tt - Total temperature [K]
        - M - Mach number
        - f - fuel/air ratio
        - TtdT - temperature ratio
        - PtdP - pressure ratio
        - MFP - mass flow parameter
    Outputs:
        - M - Mach number
        - TtdT - temperature ratio
        - PtdP - pressure ratio
        - MFP - mass flow parameter
    """

    BTU_to_ft_lb = 780
    g_c = 1

    if item == 1:  # Mach known
        TtdT, PtdP, MFP = MASSFP(Tt, f, M)
    elif item == 2 or item == 3:  # Tt/T or Pt/P known
        Tt, ht, Prt, phi_t, cpt, Rt, gamma_t, at = FAIR(item=1, f=f, T=Tt)
        if item == 2:
            T = Tt/TtdT
            T, h, Pr, phi, cp, R, gamma, a = FAIR(item=2, f=f, T=T)
        else:
            Pr = Prt/PtdP
            T, h, Pr, phi, cp, R, gamma, a = FAIR(item=3, f=f, Pr=Pr)

        Vp2 = 2*(ht - h)*g_c
        if Vp2 < 0:
            M = 0
            T = Tt
        else:
            M = np.sqrt(Vp2)/a

        TtdT, PtdP, MFP = MASSFP(Tt=Tt, f=f, M=M)
    elif item == 4 or item == 5:  # MFP known
        if item == 4:
            M = 2
        else:
            M = 0.5

        dM = 0.1
        TtdT, PtdP, MFP_0 = MASSFP(Tt=Tt, f=f, M=M)
        while True:  # Label 4
            M = M + dM
            TtdT, PtdP, MFP_n = MASSFP(Tt=Tt, f=f, M=M)
            MFP_error = np.abs(MFP_n - MFP_0)
            if MFP_error > 0.00001:
                dM = (MFP - MFP_n)/(MFP_n - MFP_0)*dM
                MFP_0 = MFP_n
                continue  # go to 4

            else:
                break

    return M, TtdT, PtdP, MFP
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# M, TtdT, PtdP, MFP = RGCOMPR(item=1, Tt=1.4662e3, M=1, f=0.0222, TtdT=None, PtdP=None, MFP=None)
# print(M, TtdT, PtdP, MFP)
