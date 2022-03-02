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
from framework.Stability.Dynamic.state_vector import state_vector
from framework.Stability.Dynamic.controle_vector import control_vector
from framework.Stability.Dynamic.dynamics import dynamics
from framework.Stability.Dynamic.Cmat import Cmat
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def trim(x, trim_par):
    # dimensão de x è 14 incognitas
    # X = [1   2   3    4   5 6  7    8   9  10  11   12 | 13  14  15 16 17 18]
    # X = [V alpha q  theta H x beta phi  p  r   psi  y  | Tle Tre ih de da dr]
    # x = [1   2   3    4   5 -  6    7   8  9  -    -   | 10  11  12 13 14 15]
    # trim_par parametros de trimagem
    X = state_vector(x, trim_par)
    U = control_vector(x)

    # [Xdot, Y] = dynamics(0, X, U, trim_par.W)
    Xdot = dynamics(0, X, U)

    # Velocidade que està sendo utilizada é a velocidade inercial
    C_tv = (Cmat(2, trim_par['gamma_deg']*np.pi/180)
            ).dot(Cmat(3, trim_par['chi_deg']*np.pi/180))
    V_i = C_tv.dot([trim_par['V'], 0, 0])
    Beta = X[6]

    f = [Xdot[0],
         Xdot[1],
         Xdot[2],
         Xdot[3]-trim_par.theta_dot_deg_s,
         Xdot[4] - V_i[2],
         Xdot[5] - V_i[0],
         Xdot[6],
         Xdot[7]-trim_par.phi_dot_deg_s,
         Xdot[8],
         Xdot[9],
         Xdot[10]-trim_par.psi_dot_deg_s,
         Xdot[11] - V_i[1],
         Beta,
         U[1] - U[2]]
    return f
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
