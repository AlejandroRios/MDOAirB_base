"""
MDOAirB

Description:
    - This module performs the sizing of the horizontal tail.
Reference:
    -

TODO's:
    - Review final results - the equations are in lb or kg?

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

from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global deg_to_rad
deg_to_rad = np.pi/180
kg_to_lb = 2.20462
kt_to_ms = 0.514444

def sizing_vertical_tail(vehicle,
    mach,
    altitude):
    """
    Description:
        - This function performs the sizing of the vertical tail.
 
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - mach - mach number
        - altitude - [ft]
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    vertical_tail = vehicle['vertical_tail']
    horizontal_tail = vehicle['horizontal_tail']

    # % ## 

    vertical_tail['twist']  = 0
    vertical_tail['dihedral'] = 90

    # print('vt_ar',vertical_tail['aspect_ratio'])
    # print('vt_area',vertical_tail['area'])
    vertical_tail['span'] = np.sqrt(vertical_tail['aspect_ratio']*vertical_tail['area'])
    vertical_tail['center_chord'] = (2*vertical_tail['area'])/(vertical_tail['span']*(1+vertical_tail['taper_ratio']))
    vertical_tail['tip_chord'] = vertical_tail['taper_ratio']*vertical_tail['center_chord']
    vertical_tail['root_chord'] = vertical_tail['tip_chord']/vertical_tail['taper_ratio']
    vertical_tail['mean_geometrical_chord'] = vertical_tail['area']/vertical_tail['span']
    vertical_tail['mean_aerodynamic_chord'] = 2/3 * vertical_tail['center_chord'] * (1 + vertical_tail['taper_ratio'] + vertical_tail['taper_ratio']**2)/(1 + vertical_tail['taper_ratio'])
    vertical_tail['mean_aerodynamic_chord_yposition'] = (2*vertical_tail['span'])/6*(1+2*vertical_tail['taper_ratio'])/(1+vertical_tail['taper_ratio'])
    vertical_tail['sweep_leading_edge'] = 1/deg_to_rad*(np.arctan(np.tan(deg_to_rad*vertical_tail['sweep_c_4']) + 1/vertical_tail['aspect_ratio']*(1 - vertical_tail['taper_ratio'])/(1 + vertical_tail['taper_ratio'])))
    vertical_tail['sweep_c_2'] = 1/deg_to_rad*(np.arctan(np.tan(deg_to_rad*vertical_tail['sweep_c_4']) - 1/vertical_tail['aspect_ratio']*(1 - vertical_tail['taper_ratio'])/(1 + vertical_tail['taper_ratio'])))
    vertical_tail['sweep_trailing_edge'] = 1/deg_to_rad*(np.arctan(np.tan(deg_to_rad*vertical_tail['sweep_c_4']) - 3/vertical_tail['aspect_ratio']*(1 - vertical_tail['taper_ratio'])/(1 + vertical_tail['taper_ratio'])))

    vertical_tail['thickness_ratio'][0] = 0.11
    vertical_tail['thickness_ratio'][1] = 0.11
    vertical_tail['mean_thickness'] = (vertical_tail['root_chord'] + 3*vertical_tail['thickness_ratio'][1])/4
    vertical_tail['wetted_area'] = 2*vertical_tail['area']*(1 + 0.25*vertical_tail['thickness_ratio'][0]*(1 + (vertical_tail['thickness_ratio'][0]/vertical_tail['thickness_ratio'][1])*vertical_tail['taper_ratio'])/(1 + vertical_tail['taper_ratio']))
    # print(vertical_tail['aspect_ratio'])
    # print(vertical_tail['area'])
    vertical_tail['span'] = np.sqrt(vertical_tail['aspect_ratio']*vertical_tail['area'])

    if horizontal_tail['position']  == 1:
        kv = 1  # vertical tail mounted in the fuselage
    else:
        zh = 0.95*vertical_tail['span']
        kv = 1 + 0.15*((horizontal_tail['area']*zh)/(vertical_tail['area']*vertical_tail['span']))

    theta, delta, sigma, T_ISA, P_ISA, rho_ISA, _, a = atmosphere_ISA_deviation(
        altitude, 0)

    V_dive = (mach*a)*kt_to_ms
    aux_1 = (vertical_tail['area']**0.2 * V_dive)/(1000*np.sqrt(np.cos(vertical_tail['sweep_c_2']*deg_to_rad)))
    vertical_tail['weight'] = kv*vertical_tail['area']*(62*aux_1 - 2.5)
    # print('vt_weight:',vertical_tail['weight'] )

    return vehicle
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
