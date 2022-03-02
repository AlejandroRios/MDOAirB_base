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
from numpy import linalg as LA
from framework.Stability.Dynamic.skew import skew
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def Cmat(n, angle_rad):
    nn = n
    # if len(nn)>1:
    #     nn = 4

    if nn == 1:  # Gamma rotation
        C = np.array([[1, 0, 0],
                      [0,  np.cos(angle_rad), np.sin(angle_rad)],
                      [0, -np.sin(angle_rad), np.cos(angle_rad)]])
    elif nn == 2:  # Theta rotations
        C = np.array([[np.cos(angle_rad), 0, -np.sin(angle_rad)],
                      [0, 1, 0],
                      [np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif nn == 3:  # Phi rotation
        C = np.array([[np.cos(angle_rad), np.sin(angle_rad), 0],
                      [-np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])
    else:
        n = n/LA.norm(n)
        C = (1-np.cos(angle_rad))*(np.dot(n, n.T)+np.cos(angle_rad)
                                   * np.eye(3, dtype=int)+-np.sin(angle_rad)*skew(n))

    return C
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# print(Cmat(4, 0))
