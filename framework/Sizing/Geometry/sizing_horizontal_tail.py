"""
MDOAirB

Description:
    - This module performs the sizing of the horizontal tail.
Reference:
    -

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
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def sizing_horizontal_tail(vehicle, mach, ceiling):
    """
    Description:
        - This function performs the sizing of the horizontal tail.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - mach - mach number
        - ceiling - [ft]
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    deg_to_rad = np.pi/180
    m2_to_ft2 = (1/0.3048)**2
    kt_to_ms = 1/1.943844   # [kt] para [m/s]
    lb_to_kg = 0.4536

    wing = vehicle['wing']
    winglet = vehicle['winglet']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']


    horizontal_tail['sweep_c_4'] = wing['sweep_c_4'] + 4
    horizontal_tail['aspect_ratio'] = horizontal_tail['aspect_ratio']  # alongamento EH
    horizontal_tail['taper_ratio'] = horizontal_tail['taper_ratio']  # Afilamento EH
    horizontal_tail['mean_chord_thickness'] = (horizontal_tail['thickness_root_chord']+horizontal_tail['thickness_tip_chord'])/2  # [#]espessura media
    horizontal_tail['tail_to_wing_area_ratio']  = horizontal_tail['area']/wing['area']  # rela�ao de areas
    horizontal_tail['twist']  = 0  # torcao EH

    if horizontal_tail['position']  == 1:
        # [�] enflechamento bordo de ataque
        horizontal_tail['sweep_leading_edge'] = 1/deg_to_rad * \
            (np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4']) +
                       1/horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))
        horizontal_tail['sweep_c_2'] = 1/deg_to_rad*(np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4'])-1 /
                                         horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))  # [�] enflechamento C/2
        # [�] enflechamento bordo de fuga
        horizontal_tail['sweep_trailing_edge'] = 1/deg_to_rad * \
            (np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4']) -
                       3/horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))
        horizontal_tail['span'] = np.sqrt(horizontal_tail['aspect_ratio']*horizontal_tail['area'])  # evergadura EH
        horizontal_tail['center_chord'] = 2*horizontal_tail['area']/(horizontal_tail['span']*(1+horizontal_tail['taper_ratio']))  # corda de centro
        horizontal_tail['tip_chord'] = horizontal_tail['taper_ratio']*horizontal_tail['center_chord']  # corda na ponta
        horizontal_tail['dihedral'] = 3
    else:
        horizontal_tail['center_chord'] = vertical_tail['tip_chord']
        horizontal_tail['tip_chord'] = horizontal_tail['taper_ratio']*horizontal_tail['center_chord']
        horizontal_tail['span'] = 2*horizontal_tail['area']/(horizontal_tail['tip_chord']+horizontal_tail['center_chord'])
        horizontal_tail['aspect_ratio'] = horizontal_tail['span']**2/horizontal_tail['area']
        # if "T" config a negative dihedral angle to help relaxe  lateral stability
        horizontal_tail['dihedral'] = -2
        # [�] enflechamento bordo de ataque
        horizontal_tail['sweep_leading_edge'] = 1/deg_to_rad * \
            (np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4']) +
                       1/horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))
        horizontal_tail['sweep_c_2'] = 1/deg_to_rad*(np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4'])-1 /
                                         horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))  # [�] enflechamento C/2
        # [�] enflechamento bordo de fuga
        horizontal_tail['sweep_trailing_edge'] = 1/deg_to_rad * \
            (np.arctan(np.tan(deg_to_rad*horizontal_tail['sweep_c_4']) -
                       3/horizontal_tail['aspect_ratio']*(1-horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])))

    # corda da ponta
    horizontal_tail['mean_geometrical_chord'] = horizontal_tail['area']/horizontal_tail['span']  # mgc
    horizontal_tail['mean_aerodynamic_chord'] = 2/3*horizontal_tail['center_chord']*(1+horizontal_tail['taper_ratio']+horizontal_tail['taper_ratio']**2) / \
        (1+horizontal_tail['taper_ratio'])  # mean aerodynamic chord
    horizontal_tail['mean_aerodynamic_chord_yposition']  = horizontal_tail['span']/6*(1+2*horizontal_tail['taper_ratio'])/(1+horizontal_tail['taper_ratio'])
    #
    ######################### HT Wetted area ######################################
    horizontal_tail['tau'] = horizontal_tail['center_chord']/horizontal_tail['tip_chord'] 
    #ht.thicknessavg = horizontal_tail['mean_chord']*0.50*(horizontal_tail['center_chord']+horizontal_tail['tip_chord'])
    horizontal_tail['wetted_area'] = 2.*horizontal_tail['area'] * \
        (1+0.25*horizontal_tail['thickness_root_chord']*(1+(horizontal_tail['tau'] * horizontal_tail['taper_ratio']))/(1+horizontal_tail['taper_ratio']))  # [m2]
    # HT aerodynamic center
    if horizontal_tail['position']  == 1:
        horizontal_tail['aerodynamic_center'] = (0.92*fuselage['length'] - horizontal_tail['center_chord'] + horizontal_tail['mean_aerodynamic_chord_yposition'] *np.tan(deg_to_rad*horizontal_tail['sweep_leading_edge']) +
                     horizontal_tail['aerodynamic_center_ref']*horizontal_tail['mean_aerodynamic_chord'])
    else:
        horizontal_tail['aerodynamic_center'] = 0.95*fuselage['length']-vertical_tail['center_chord']+vertical_tail['span'] * \
            np.tan(deg_to_rad*vertical_tail['sweep_leading_edge'])+horizontal_tail['aerodynamic_center_ref'] * \
            horizontal_tail['mean_aerodynamic_chord']+horizontal_tail['mean_aerodynamic_chord_yposition'] *np.tan(deg_to_rad*horizontal_tail['sweep_leading_edge'])

    # EMPENAGEM HORIZONTAL (HORIZONTAL TAIL)
    theta, delta, sigma, T_ISA, P_ISA, rho_ISA, _, a = atmosphere_ISA_deviation(
        ceiling, 0)                                 # propriedades da atmosfera
    va = a                                         # velocidade do som [m/s]
    sigma = sigma
    # velocidade de cruzeiro meta, verdadeira [m/s]
    vc = mach*va
    # velocidade de cruzeiro meta [KEAS]
    vckeas = (vc*sigma**0.5)
    vdkeas = 1.25*vckeas
    # empenagem horizontal movel
    kh = 1.1
    prod1 = 3.81*(((horizontal_tail['area']*m2_to_ft2)**0.2)*vdkeas)                    # termo 1
    prod2 = (1000*(np.cos(horizontal_tail['sweep_c_2']*deg_to_rad)) **
             0.5)                     # termo 2
    prodf = prod1/prod2                                          # termo 3
    horizontal_tail['weight'] = (1.25*kh*(horizontal_tail['area']*m2_to_ft2)*(prodf-0.287))*lb_to_kg

    # print('ht weight',horizontal_tail['weight'])

    return vehicle

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================