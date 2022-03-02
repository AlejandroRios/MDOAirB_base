"""
Function  :
Title     :
Written by: 
Email     : aarc.88@gmail.com
Date      : 
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
Inputs:
    -
Outputs:
    -
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from framework.baseline_aircraft_GNBA import baseline_aircraft
from framework.Stability.Dynamic.Cmat import Cmat
from framework.Attributes.Atmosphere.atmosphere import atmosphere
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def aero_loads(state, control):

    # state =state.squeeze().T
    # control = control.squeeze().T

    aircraft_data = baseline_aircraft()

    V = state[0]
    alpha_deg = state[1]
    q_deg_s = state[2]
    theta_deg = state[3]
    h = state[4]
    x = state[5]
    beta_deg = state[6]
    phi_deg = state[7]
    p_deg_s = state[8]
    r_deg_s = state[9]
    psi_deg = state[10]
    y = state[11]

    ih = control[2]
    delta_e = control[3]
    delta_a = control[4]
    delta_r = control[5]

    b = wing['span']
    c = aircraft_data['mean_aerodynamic_chord']
    S = wing['area']

    ## ----------------------------Conversões---------------------------------#
    q_rad_s = np.deg2rad(q_deg_s)
    p_rad_s = np.deg2rad(p_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    ## ----------------------Matriz de Transformação--------------------------#
    C_alpha = Cmat(2, np.deg2rad(alpha_deg))
    C_beta = Cmat(3, np.deg2rad(-beta_deg))
    # C_ba   = C_alpha*C_beta
    C_ba = C_alpha.dot(C_beta)

    ## -----------------------------Atmosfera---------------------------------#
    ft_to_m = 0.3048
    _, _, rho, _ = atmosphere(h/ft_to_m)
    q_bar = 0.5*rho*V**2

    ## -----------------------------Sustentação-------------------------------#
    CL0 = 0.308
    CLa = 0.133
    CLq = 16.7
    CLih = 0.0194
    CLde = 0.00895

    CL = CL0+CLa*alpha_deg+CLq*((q_rad_s*c)/(2*V))+CLih*ih+CLde*delta_e
    La = q_bar*S*CL

    ## --------------------------------Arrasto--------------------------------#
    CD0 = 0.02207
    CDa = 0.00271
    CDa2 = 0.000603
    CDq2 = 35.904
    CDb2 = 0.00016
    CDp2 = 0.5167
    CDr2 = 0.5738
    CDih = -0.00042
    CDih2 = 0.000134
    CDde2 = 4.61e-5
    CDda2 = 3e-5
    CDdr2 = 1.81e-5

    CD = CD0+CDa*alpha_deg+CDa2*alpha_deg**2+CDq2*((q_rad_s*c)/(2*V))**2+CDb2*beta_deg**2+CDp2*((p_rad_s*b)/(
        2*V))**2+CDr2*((r_rad_s*b)/(2*V))**2+CDih*ih+CDih2*ih**2+CDde2*delta_e**2+CDda2*delta_a**2+CDdr2*delta_r**2
    D = q_bar*S*CD

    ## --------------------------------Lateral--------------------------------#
    Cyb = 0.0228
    Cyp = 0.084
    Cyr = -1.21
    Cyda = 2.36e-4
    Cydr = -5.75e-3

    CY = Cyb*beta_deg+Cyp*((p_rad_s*b)/(2*V))+Cyr * \
        ((r_rad_s*b)/(2*V))+Cyda*delta_a+Cydr*delta_r
    Y = q_bar*S*CY

    ## --------------------------Momento de Arfagem---------------------------#
    CM0 = 0.017
    CMa = -0.0402
    CMq = -57
    CMih = -0.0935
    CMde = -0.0448

    CM = CM0+CMa*alpha_deg+CMq*((q_rad_s*c)/(2*V))+CMih*ih+CMde*delta_e
    M = q_bar*S*c*CM

    ## -------------------------Momento de Rolamento--------------------------#
    Clb = -3.66e-3
    Clp = -0.661
    Clr = 0.144
    Clda = -2.87e-3
    Cldr = 6.76e-4

    Cl = Clb*beta_deg+Clp*((p_rad_s*b)/(2*V))+Clr * \
        ((r_rad_s*b)/(2*V))+Clda*delta_a+Cldr*delta_r
    L = q_bar*S*b*Cl

    ## ---------------------------Momento de Guinada--------------------------#
    Cnb = 5.06e-3
    Cnp = 0.0219
    Cnr = -0.634
    Cnda = 0
    Cndr = -3.26e-3

    Cn = Cnb*beta_deg+Cnp*((p_rad_s*b)/(2*V))+Cnr * \
        ((r_rad_s*b)/(2*V))+Cnda*delta_a+Cndr*delta_r
    N = q_bar*S*b*Cn

    ## --------------------------Forças Aerodinâmicas-------------------------#
    # Faero_b = C_ba*np.array([[-D], [-Y], [-La]])

    # print(C_ba)
    Faero_b = C_ba.dot(np.array([-D, -Y, -La]))

    ## -------------------------Momentos Aerodinâmicos------------------------#
    Maero_O_b = np.array([[L], [M], [N]])

    ## ---------------------------------Saidas--------------------------------#
    Yaero = np.array([[CL], [CD], [CY], [CM], [Cl], [Cn]])

    return Faero_b, Maero_O_b, Yaero
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# state = [ 265,
#            0,
#            0,
#            0,
#        11582,
#            0,
#            0,
#            -2,
#            0,
#            0,
#            0,
#            0]

# control = [0.0020,    0.0020,    0.1417,         0,    0.0103,   -0.0338]
# Faero_b, Maero_O_b, Yaero = aero_loads(state, control)

# print(Faero_b)
# print(Maero_O_b)
# print(Yaero)
