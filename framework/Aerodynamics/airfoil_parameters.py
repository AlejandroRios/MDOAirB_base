"""
MDOAirB

Description:
    - This module calculates airfoil parameters based in Sobieski approach


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
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
# import warnings
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def generate_sobieski_coefficients(rBA, phi, X_tcmax, t_c, theta, epsilon, X_Ycmax, Ycmax, YCtcmax, Hte, EspBF):
    '''
    Description:
        This function solve the system that geneterates the Sobieski coefficients
    Inputs:
        - rBA 
        - phi
        - X_tcmax
        - t_c
        - theta
        - epsilon
        - X_Ycmax
        - Ycmax
        - YCtcmax
        - Hte
        - EspBF
    Outputs:
        - mat A solved
        - mat B solved
        '''
    # These are constant parameters
    # EspBF = 0.0025 # Trailing edge thickness
    # Hte = 0.0 # Trailing edge height (distance to the chord-line)

    #Thickness matrix
    Mt = np.array([[1, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [2*np.sqrt(X_tcmax), 2*X_tcmax, 2*X_tcmax**2, 2*X_tcmax**3, 2*X_tcmax**4],
        [1/2/np.sqrt(X_tcmax), 1, 2*X_tcmax, 3*X_tcmax**2, 4*X_tcmax**3],
        [1, 2, 4, 6, 8]])

    xt = np.array([np.sqrt(2*rBA), EspBF, t_c, 0, phi])

    #Camber matrix
    Mc = np.array([[1, 0, 0, 0, 0, 0],
         [1, 2, 3, 4, 5, 6],
         [X_Ycmax, X_Ycmax**2, X_Ycmax**3, X_Ycmax**4, X_Ycmax**5, X_Ycmax**6],
         [1, 2*X_Ycmax, 3*X_Ycmax**2, 4*X_Ycmax**3, 5*X_Ycmax**4, 6*X_Ycmax**5],
         [X_tcmax, X_tcmax**2, X_tcmax**3, X_tcmax**4, X_tcmax**5, X_tcmax**6],
         [1, 1, 1, 1, 1, 1]])

    xc = np.array([theta, epsilon, Ycmax, 0, YCtcmax, Hte])

    #Solving systems
    A_rev = np.dot(np.linalg.inv(Mt), xt)
    B_rev = np.dot(np.linalg.inv(Mc), xc)

    return A_rev, B_rev

def generate_sobieski_coordinates(a,b,x):
    '''
    Description:
        This function generates the Sobieski coordinates
    Inputs:
        - a
        - b
        - x
    Outputs:
        - yu
        - yl
        - xsing
        - ysing
        - trail
        - slopt
        - radius
        - t_c
        - Hte
        - yc
    '''

    yt  = a[0]*x**0.5 + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4
    yc = b[0]*x + b[1]*x**2 + b[2]*x**3 + b[3]*x**4  + b[4]*x**5 + b[5]*x**6

    yu = yt + yc
    yl = yc - yt
    t_c = max(yu)-min(yl)
    Hte = yu[-1]-yl[-1]

    radius= (a[0]**2)/2
    ysing = yu[0]
    xsing = x[0] + 0.5*radius

    upperangle = (yu[-2]-yu[-1])/(x[-1]-x[-2])
    lowerangle = (yl[-2]-yl[-1])/(x[-1]-x[-2])

    trail = (upperangle - lowerangle) * (180/np.pi)
    slopt = (yc[-2]-yc[-1])/(x[-2]-x[-1])

    return yu, yl, xsing, ysing, trail, slopt, radius, t_c, Hte, yc

def solve_coefficients(airfoil_params,xp,ysup,yinf,Hte,EspBF):
    '''
    Description:
        This function uses the generate_sobieski_coefficients function to estimate the error 
    Inputs:
        - airfoil_params
        - xp 
        - ysup
        - yinf
        - Hte
        - EspBF
    Outputs:
        - error
    '''
    r0      = airfoil_params[0]
    t_c     = airfoil_params[1]
    phi     = airfoil_params[2]
    X_tcmax = airfoil_params[3]
    theta   = airfoil_params[4]
    epsilon = airfoil_params[5]
    Ycmax   = airfoil_params[6]
    YCtcmax = airfoil_params[7]
    X_Ycmax = airfoil_params[8]

    #Generating sobieski coefficients
    A, B=generate_sobieski_coefficients(r0, phi, X_tcmax, t_c, theta, epsilon, X_Ycmax, Ycmax, YCtcmax, Hte, EspBF)

    #Regenerating airfoil coordinates
    ysup_rev, yinf_rev,_, _, _, _, _, _, _, _= generate_sobieski_coordinates(A,B,xp)

    # Compute difference between coordinates
    error = np.sum((ysup-ysup_rev)**2) + np.sum((yinf-yinf_rev)**2)

    return error

def airfoil_sobieski_coefficients(fileToRead1):
    '''
    Description:
        This is the main function for the estimation of sobieski coefficients
    Inputs:
        - fileToRead1 - airfoil file name as specified in Database/Airfoils (withoud extension)
    Outputs:
        - r0 - airfoil leading edge radius [deg]
        - t_c - thick to chord ratio
        - phi - airfoil thickness line angle at trailing edge [deg]
        - X_tcmax - x position of max thick to chord ratio
        - theta - airfoil camber line angle at leading edge [deg]
        - epsilon - airfoil camber line angle at trailing edge [deg]
        - Ycmax - aifoil maximum camber
        - YCtcmax - camber at maximum thickness chordwise position
        - X_Ycmax - airfoil maximum camber position
        - xp - airfoil x coordinate
        - yu - airfoil upper surface coordinates
        - yl - airfoil lower surface coordinates
    '''
    airfoil_names = [fileToRead1]
    # Load airfoil coordinates
    df = pd.read_csv('Database/Airfoils/' +
        airfoil_names[0] + '.dat', sep=',', delimiter=None, header=None, skiprows=[0])
    df.columns = ['x', 'y']
    df_head = df.head()
    n_coordinates = len(df)

    # Compute distance between consecutive points
    dx = []
    dy = []
    ds = []
    ds_vector = []
    ds = np.zeros((n_coordinates, 1))
    ds[0] = 0

    for i in range(1, n_coordinates):
        dx = df.x[i] - df.x[i-1]
        dy = df.y[i] - df.y[i-1]
        ds[i] = ds[i-1] + np.sqrt(dx*dx+dy*dy)

    xa = df.x[0]
    xb = df.x[1]
    ind = 0

    # Find leading edge index
    while xb < xa:
        ind = ind + 1
        xa = df.x[ind]
        xb = df.x[ind+1]

    n_panels_x = 51
    xp = np.linspace(0, 1, n_panels_x)
    xp = np.flip((np.cos(xp*np.pi)/2+0.5))

    # Interpolate upper skin
    dsaux = ds[0:ind+1]
    xaux = df.x[0:ind+1]

    dsaux = np.reshape(dsaux, -1)
    ds = np.reshape(ds, -1)

    dsinterp = interpolate.interp1d(
        xaux, dsaux, kind='slinear', fill_value='extrapolate')(xp)
    yupp_root = interpolate.interp1d(ds, df.y, kind='slinear')(dsinterp)

    # Interpolate lower skin
    dsaux = []
    dsaux = ds[ind:n_coordinates]
    dsinterp = []
    xaux = df.x[ind:n_coordinates]

    dsinterp = interpolate.interp1d(
        xaux, dsaux, kind='slinear', fill_value='extrapolate')(xp)
    ylow_root = interpolate.interp1d(ds, df.y, kind='slinear')(dsinterp)

    xproot = np.array([np.flip(xp), xp])
    xproot = xproot.ravel()
    yproot = np.array([np.flip(yupp_root), ylow_root])
    yproot = yproot.ravel()
    esspraiz = max(yupp_root)-min(ylow_root)

    # Now get airfoil parameters that get closest to the required shape
    # We will do this by minimizing the square error between computed and
    # required coordinates.

    # Define bounds for variables
    LB=np.array([0.015, 0.08,  -0.24, 0.20, -0.2, -0.300, -0.050, -0.050, 0.5])
    UB=np.array([0.20 , 0.16,   0.10, 0.46,  0.1, -0.005, 0.030, 0.025, 0.80])

    bnds = ((0.015,0.2),(0.08,0.16),(-0.24,0.1),(0.2,0.46),(-0.2,0.1),(-0.3,-0.005),(-0.05,0.03),(-0.05,0.25),(0.5,0.8))
    
    airfoil_params0 = 0.50*(UB+LB)

    # These airfoil parameters are constant
    Hte = 0.0 #x(10)
    EspBF = 0.0025 #x(11)

    res = minimize(solve_coefficients, airfoil_params0, bounds = bnds, args = (xp,yupp_root,ylow_root,Hte,EspBF),method='SLSQP',
               options={'disp': False})

    # Split parameters
    r0      = res.x[0]
    t_c     = res.x[1]
    phi     = res.x[2]
    X_tcmax = res.x[3]
    theta   = res.x[4]
    epsilon = res.x[5]
    Ycmax   = res.x[6]
    YCtcmax = res.x[7]
    X_Ycmax = res.x[8]

    # # Get new coordinates with the final results
    A, B=generate_sobieski_coefficients(r0, phi, X_tcmax, t_c, theta, epsilon, X_Ycmax, Ycmax, YCtcmax, Hte, EspBF)

    #Regenerating airfoil coordinates
    yu, yl,_, _, _, _, _, _, _, _= generate_sobieski_coordinates(A,B,xp)

    return r0, t_c, phi, X_tcmax, theta, epsilon, Ycmax, YCtcmax, X_Ycmax, xp, yu, yl

def airfoil_parameters(vehicle):
    '''
    Description:
        This function run the main function for all the airfoils that made part of the wing and 
        update the vehicle dictionary with the outputs
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    '''
    wing = vehicle['wing']

    airfoil_file = np.array(['PR1','PQ1','PT4'])

    r0 = []
    t_c = []
    phi = []
    X_tcmax = []
    theta = []
    epsilon = []
    Ycmax = []
    YCtcmax = []
    X_Ycmax = []
    

    for i in airfoil_file :
        r0_aux, t_c_aux, phi_aux, X_tcmax_aux, theta_aux, epsilon_aux, Ycmax_aux, YCtcmax_aux, X_Ycmax_aux, _, _,_ = airfoil_sobieski_coefficients(i)
        r0.append(r0_aux)
        t_c.append(t_c_aux)
        phi.append(phi_aux)
        X_tcmax.append(X_tcmax_aux)
        theta.append(theta_aux)
        epsilon.append(epsilon_aux)
        Ycmax.append(Ycmax_aux)
        YCtcmax.append(YCtcmax_aux)
        X_Ycmax.append(X_Ycmax_aux)

    wing['leading_edge_radius'] = r0
    wing['thickness_ratio'] = t_c
    wing['thickness_line_angle_trailing_edge'] = phi
    wing['maximum_thickness_chordwise_position'] = X_tcmax
    wing['camber_line_angle_leading_edge'] = theta
    wing['camber_line_angle_trailing_edge'] =  epsilon
    wing['maximum_camber'] = Ycmax
    wing['camber_at_maximum_thickness_chordwise_position'] = YCtcmax
    wing['maximum_camber_chordwise_position'] = X_Ycmax

    return vehicle

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
