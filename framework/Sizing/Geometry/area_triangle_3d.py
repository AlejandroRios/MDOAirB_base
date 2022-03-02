"""
MDOAirB

Description:
    - This module calculate the area triangle for applications in 3D surfaces area calculation.

Reference:
    - https://math.stackexchange.com/questions/128991/how-to-calculate-area-of-3d-triangle

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

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def area_triangle_3d(x, y, z):
    """
    Description:
        - This function calculate the area triangle for applications in 3D surfaces area calculation.
 
    Inputs:
        - x - vector containing x coordinates of the triangle
        - y - vector containing y coordinates of the triangle
        - z - vector containing z coordinates of the triangle
    Outputs:
        - area - triangle area
    """

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    #
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    #
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    # A=sqrt(s*(s-a)*(s-b)*(s-c))

    T1 = (x2*y1 - x3*y1 - x1*y2 + x3*y2 + x1*y3 - x2*y3)**2
    T2 = (x2*z1 - x3*z1 - x1*z2 + x3*z2 + x1*z3 - x2*z3)**2
    T3 = (y2*z1 - y3*z1 - y1*z2 + y3*z2 + y1*z3 - y2*z3)**2
    area = (np.sqrt(T1+T2+T3)/2)
    return(area)

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================