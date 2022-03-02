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
from framework.Attributes.Atmosphere.atmosphere import atmosphere
from framework.Stability.Dynamic.trim import trim
from scipy import optimize

import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
ft_to_m = 0.3048

H_ft_eq = 10000
_, _, _, a = atmosphere(H_ft_eq)
mach = 0.2
V_eq = mach*a

gamma_deg_eq = 10
phi_dot_deg_s_eq = 0
theta_dot_deg_s_eq = 0
psi_dot_deg_s_eq = 0
beta_deg_eq = 0
chi_deg = 0

trim_par = {}
trim_par = {'V': V_eq,
            'H_m': H_ft_eq,
            'chi_deg': chi_deg,
            'gamma_deg': gamma_deg_eq,
            'phi_dot_deg_s': phi_dot_deg_s_eq,
            'theta_dot_deg': theta_dot_deg_s_eq,
            'psi_dot_deg_s': psi_dot_deg_s_eq,
            'beta_deg_eq': beta_deg_eq,
            'W': [0, 0, 0]}


x_eq_0 = np.zeros((14, 1))
x_eq_0[0] = V_eq


x_eq = optimize.fsolve(trim, x_eq_0, args=(trim_par))
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
