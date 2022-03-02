"""
MDOAirB

Description:
    - Calculates the temperature dependent properties, the specific heat Cp, 
    the gas constant R, the ratio of specific heatsγand the speed of sound a

Reference:
    - Gordon 1976

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
from scipy import optimize
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def FAIR(item=None, f=None, T=None, h=None, Pr=None, phi=None):
    """
    Description:
        Calculates the temperature dependent properties
    Inputs:
        - item - computation mode
        - f - fuel/air ratio
        - T - temperature 
        - h - static enthalpy
        - Pr - relative pressure
        - phi - temperature dependent portion of entropy
    Outputs:
        - T - temperature [K]
        - h - static enthalpy [Jkg]
        - Pr - relative pressure [pa]
        - phi - temperature dependent portion of entropy [JK/gK]
        - Cp - specific heat at constant pressure [JK/gK]
        - R - gas constant [JK/gK]
        - gamma - ratio of specific heats
        - a - speed of sound [m/s]
    """
    
    list_variables = [item, f, T, h, Pr, phi]
    nargin = sum(x is not None for x in list_variables)

    # Derived from equations
    # Convertion factors
    R2K = 0.5556
    BTUlbm2Jkg = 2326
    psia2pa = 6895
    BTUlbR2JKgK = 4184
    fts2ms = 0.3048

    nargin = 2 + nargin
    if f > 0.0676:
        print('f cannot be greater than 0.0676')

    if item == 1:  # T is known
        if nargin > 2:
            T = T/R2K
            h, Pr, phi, Cp, R, gamma, a = unFAIR(T, f)
        else:
            print('T must be defined for case 1')

    elif item == 2:  # h is known
        if nargin > 3:
            h = h/BTUlbm2Jkg

            T = optimize.fminbound(lambda T: np.abs(
                h-find_h(f, T)), 166.667, 2222.222)
            h, Pr, phi, Cp, R, gamma, a = unFAIR(T, f)
        else:
            print('h must be defined for case 2')

    elif item == 3:  # Pr is known
        if nargin > 4:
            Pr = Pr/psia2pa
            T = optimize.fminbound(lambda T: np.abs(
                Pr-find_Pr(f, T)), 166.667, 2222.222)
            # T = fminbnd(@(T)abs(Pr-findPr(f, T)), 166, 2222.222)
#                 T = fminsearch(@(T)abs(Pr-findPr(f, T)), 200)
            h, Pr, phi, Cp, R, gamma, a = unFAIR(T, f)
        else:
            print('Pr must be defined for case 2')

    elif item == 4:  # phi is known
        if nargin > 5:
            phi = phi/BTUlbR2JKgK
            T = optimize.fminbound(lambda T: np.abs(
                phi-find_phi(f, T)), 166.667, 2222.222)
            # T = fminbnd(@(T)abs(phi-findphi(f, T)), 166.667, 2222.222)
#                 T = fminsearch(@(T)abs(phi-findphi(f, T)), 200)
            h, Pr, phi, Cp, R, gamma, a = unFAIR(T, f)
        else:
            print(' must be defined for case 2')

    T = T*R2K
    h = h*BTUlbm2Jkg
    Pr = Pr*psia2pa
    phi = phi*BTUlbR2JKgK
    Cp = Cp*BTUlbR2JKgK
    R = R*BTUlbR2JKgK
    a = a*fts2ms

    return T, h, Pr, phi, Cp, R, gamma, a


def find_h(f, T):
    h, _, _, _, _, _, _ = unFAIR(T, f)
    return h


def find_Pr(f, T):
    _, Pr, _, _, _, _, _ = unFAIR(T, f)
    return Pr


def find_phi(f, T):
    _, _, phi, _, _, _, _ = unFAIR(T, f)
    return(phi)


def unFAIR(T, FAR):

    BTU_lbm_to_ft2_s2 = 25037.00

    [Cp_a, h_a, phi_a] = AFPROP_A(T)
    [Cp_p, h_p, phi_p] = AFPROP_P(T)

    # ============ Equation 4.26 a, b, c, d ===================
    R = 1.9857117/(28.97-FAR*0.946186)  # BTU ./( lbm R)
    Cp = (Cp_a+FAR*Cp_p)/(1+FAR)
    h = (h_a+FAR*h_p)/(1+FAR)
    phi = (phi_a+FAR*phi_p)/(1+FAR)
    # ============ Equation 2.55 - " reduced pressure " =======
    phi_ref = 1.578420959  # BTU ./( lbm R) phi@492 .00 R

    Pr = np.exp((phi-phi_ref)/R)

    gamma = Cp/(Cp-R)
    a = np.sqrt(gamma*R*BTU_lbm_to_ft2_s2*T)

    return h, Pr, phi, Cp, R, gamma, a


def AFPROP_A(T):
    # ===== Define coeficients from Table 2.2 for air alone ======
    A0 = 2.5020051E-01
    A1 = -5.1536879E-05
    A2 = 6.5519486E-08
    A3 = -6.7178376E-12
    A4 = -1.5128259E-14
    A5 = 7.6215767E-18
    A6 = -1.4526770E-21
    A7 = 1.0115540E-25
    h_ref = -1.7558886  # BTU ./lbm
    phi_ref = 0.0454323  # BTU ./(lbm R)
    # ====== Equations 2.60 , 2.61 , 2.62 for air alone ===========
    Cp_a, h_a, phi_a = AFPROP(T, A0, A1, A2, A3, A4,
                              A5, A6, A7, h_ref, phi_ref)

    return Cp_a, h_a, phi_a


def AFPROP_P(T):
    # ==== Now change coefficients for the products of combustion.
    A0 = 7.3816638E-02
    A1 = 1.2258630E-03
    A2 = -1.3771901E-06
    A3 = 9.9686793E-10
    A4 = -4.2051104E-13
    A5 = 1.0212913E-16
    A6 = -1.3335668E-20
    A7 = 7.2678710E-25
    h_ref = 30.58153  # BTU ./lbm
    phi_ref = 0.6483398  # BTU ./( lbm R)
    Cp_p, h_p, phi_p = AFPROP(T, A0, A1, A2, A3, A4,
                              A5, A6, A7, h_ref, phi_ref)

    return Cp_p, h_p, phi_p


def AFPROP(T, A0, A1, A2, A3, A4, A5, A6, A7, h_ref, phi_ref):
    Cp = (A0
          + A1*T
          + A2*T**2
          + A3*T**3
          + A4*T**4
          + A5*T**5
          + A6*T**6
          + A7*T**7)

    h = (h_ref
         + A0*T
         + (A1/2)*T**2
         + (A2/3)*T**3
         + (A3/4)*T**4
         + (A4/5)*T**5
         + (A5/6)*T**6
         + (A6/7)*T**7
         + (A7/8)*T**8)

    phi = (phi_ref
           + A0*np.log(T)
           + A1*T
           + A2/2*T**2
           + A3/3*T**3
           + A4/4*T**4
           + A5/5*T**5
           + A6/6*T**6
           + A7/7*T**7)
    return Cp, h, phi

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# h = 10000
# f = 0
# T = 288.15
# # print(function_h(T))
# # T = optimize.fminbound(function_h, 166.667, 2222.222)

# print(FAIR(1, f, T))

# ht0 = 289029
# print(FAIR(2, 0, h=ht0))


# print(FAIR(3, 0, Pr=36232))


# print(FAIR(1, 0.0241, T=1466))


# # def myFunc(arg1, *args):
# #   print(args)
# #   w = []
# #   w += args
# #   print(w)

# # print(myFunc(1, 3, 4, 5, 6))
