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
from framework.Stability.Dynamic.Cmat import Cmat
from framework.baseline_aircraft_GNBA import baseline_aircraft

from framework.Stability.Dynamic.skew import skew
from framework.Stability.Dynamic.aero_loads import aero_loads
from framework.Stability.Dynamic.prop_loads import prop_loads
from framework.Attributes.Atmosphere.atmosphere import atmosphere

from scipy.integrate import odeint
import matplotlib.pyplot as plt
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
# def dynamics(state, t, control1, control2, control3, control4, control5, control6):


def dynamics(t, state, control):

    aircraft_data = baseline_aircraft()

    # control = [control1, control2, control3, control4, control5, control6]

    # X       = [V alpha q theta H x beta phi p r psi y].'
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

    ## -----------------------------------------------------------------------#
    omega_b = np.array([p_deg_s, q_deg_s, r_deg_s]).T  # Velocidade ângular
    # omega_b  = omega_b.transpose()
    v = V*np.sin(np.deg2rad(beta_deg))
    u = V*np.cos(np.deg2rad(beta_deg))*np.cos(np.deg2rad(alpha_deg))
    w = V*np.cos(np.deg2rad(beta_deg))*np.sin(np.deg2rad(alpha_deg))
    V_b = np.array([u, v, w]).T  # Velocidade linear

    ## ----------------------Matriz de Transformação--------------------------#
    C_phi = Cmat(1, np.deg2rad(phi_deg))
    C_theta = Cmat(2, np.deg2rad(theta_deg))
    C_psi = Cmat(3, np.deg2rad(psi_deg))
    C_bv = C_phi.dot(C_theta).dot(C_psi)

    ## ------------------------Matriz de Gravidade----------------------------#
    g_b = C_bv.dot(np.array([0, 0, g]).T)

    ## --------------------Matriz de Massa Generalizada-----------------------#
    m = aircraft_data['maximum_takeoff_weight']/g  # Massa da Aeronave
    J_O_b = aircraft_data['inertia_matrix']  # Matriz de Inércia
    # Distância entre origem e centro de massa
    rC_b = aircraft_data['CG_position']

    Mgen = np.zeros((6, 6))

    aux1 = m*np.eye(3, dtype=int)
    aux2 = -m*skew(rC_b)
    aux3 = m*skew(rC_b)
    aux4 = J_O_b

    # print(aux1)
    Mgen[0:0+aux1.shape[0], 0:0+aux1.shape[1]] += aux1
    Mgen[0:0+aux2.shape[0], 3:3+aux2.shape[1]] += aux2
    Mgen[3:3+aux3.shape[0], 0:0+aux3.shape[1]] += aux3
    Mgen[3:3+aux4.shape[0], 3:3+aux4.shape[1]] += aux4

    ## -------------------------Forças e Momentos-----------------------------#
    Faero_b, Maero_O_b, Yaero = aero_loads(state, control)
    Fprop_b, Mprop_O_b, Yprop = prop_loads(state, control)

    ## -----------Termos restantes das equações do movimento------------------#

    eq_F = -(m*skew(omega_b).dot(V_b) - m*skew(omega_b).dot(skew(rC_b)
                                                            ).dot(omega_b)) + Faero_b.T + Fprop_b + m*g_b
    eq_M = -(skew(omega_b).dot(J_O_b).dot(omega_b) + m*skew(rC_b).dot(skew(omega_b)
                                                                      ).dot(V_b)) + Maero_O_b.T + Mprop_O_b + m*skew(rC_b).dot(g_b)

    # ----------------------------Acelerações--------------------------------#
    # edot     = Mgen\[eq_F eq_M]

    V_FM = np.zeros((1, 6))
    V_FM[0:0+eq_F.shape[0], 0:0+eq_F.shape[1]] += eq_F
    V_FM[0:0+eq_M.shape[0], 3:3+eq_M.shape[1]] += eq_M

    V_FM = V_FM.T

    edot = (np.linalg.inv(Mgen)).dot(V_FM)

    u_dot = edot[0]
    v_dot = edot[1]
    w_dot = edot[2]

    Vdot = ((V_b.T).dot(edot[0:3]))/V

    ## ----------------------Cinemática de Rotação----------------------------#

    HPhi_inv = np.zeros((3, 3))
    aux5 = np.array([C_phi[:, 0]]).T
    aux6 = np.array([C_phi[:, 1]]).T
    aux7 = np.array([C_bv[:, 2]]).T
    HPhi_inv[0:0+aux5.shape[0], 0:0+aux5.shape[1]] += aux5
    HPhi_inv[0:0+aux6.shape[0], 1:1+aux6.shape[1]] += aux6
    HPhi_inv[0:0+aux7.shape[0], 2:2+aux7.shape[1]] += aux7
    Phi_dot_rad = (np.linalg.inv(HPhi_inv)).dot(omega_b)

    ## ---------------------Cinemática de Translação--------------------------#
    dReodt = (C_bv.T).dot(V_b)

    # -----------------------------Fator de Carga----------------------------#
    n_C_b = -1/(m*g)*(Faero_b.T + Fprop_b)
    r_pilot_b = aircraft_data['r_pilot_b']
    n_pilot_b = n_C_b + -1/g * \
        (skew(edot[3:6]).dot(r_pilot_b-rC_b) +
         skew(omega_b).dot(r_pilot_b-rC_b))

    ## ---------------------------Pressão Dinâmica----------------------------#
    ft_to_m = 0.3048
    _, _, rho, a = atmosphere(h/ft_to_m)
    Mach = V/a
    q_bar = 0.5*rho*V**2

    ## ------------------------------Saída------------------------------------#
    p_deg_dot = np.rad2deg(edot[3])
    q_deg_dot = np.rad2deg(edot[4])
    r_deg_dot = np.rad2deg(edot[5])
    alpha_dot = np.rad2deg((u*w_dot-w*u_dot)/(u**2+w**2))

    # print((u*w_dot-w*u_dot))
    beta_dot = np.rad2deg((V*v_dot-v*Vdot)/(V*np.sqrt(u**2+w**2)))
    phi_dot = np.rad2deg(Phi_dot_rad[0])
    theta_dot = np.rad2deg(Phi_dot_rad[1])
    psi_dot = np.rad2deg(Phi_dot_rad[2])
    H_dot = -dReodt[2]
    V_dot = Vdot
    x_dot = dReodt[0]
    y_dot = dReodt[1]
    Xdot = [V_dot, alpha_dot, q_deg_dot, theta_dot, H_dot, x_dot,
            beta_dot, phi_dot, p_deg_dot, r_deg_dot, psi_dot, y_dot]
    Y = [n_pilot_b, n_C_b, Mach, q_bar, Yaero, Yprop, beta_deg]
    return Xdot
# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================
global g
g = 9.8067
# state0 = [    265.1970,
#    -0.8608,
#    -0.0556,
#    -2.6389,
#     11581.7862,
#     132.6543,
#    -2.6632,
#     29.1633,
#     3.8677,
#     0.2369,
#     1.9661,
#    -0.0143]


# control = (67307,
#          67307,
#     0.5318,
#          0,
#    -0.0823,
#     0.1392)

# # t = 0.5
# # X_dot, Y = dynamics(state, control, t)

# # print(Y)


# # state0 = [2.0, 0.0]
# t = np.arange(0.0, 5.0, 0.001)

# K = 1
# state = odeint(dynamics, state0, t, args=(control))

# plt.plot(t, state)
# # xlabel('TIME (sec)')
# # ylabel('STATES')
# # title('Mass-Spring System')
# # legend(('$x$ (m)', '$\dot{x}$ (m/sec)'))

# plt.show()
