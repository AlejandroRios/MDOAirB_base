"""
MDOAirB

Description:
    - This module performs the landing gear sizing.

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

from framework.Sizing.Geometry.landig_gear_position import landig_gear_position
from framework.Sizing.Geometry.landing_gear_layout import landing_gear_layout
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def sizing_landing_gear(vehicle):
    """
    Description:
        - This function performs the landing gear sizing.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    fuselage = vehicle['fuselage']
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    nose_landing_gear = vehicle['nose_landing_gear']
    main_landing_gear = vehicle['main_landing_gear']
    engine = vehicle['engine']

    wing['root_chord_yposiion'] = 0.85*(fuselage['width']/2)

    if (engine['position'] == 2 and horizontal_tail['position'] == 1):
        horizontal_tail['position'] = 2

    main_landing_gear['tyre_diameter'] = 0.95
    main_landing_gear_tyre_diameter_new = 100
    nose_landing_gear_tyre_diameter_new = 100
    main_landing_gear['piston_length'] = 1

    # Dont know the meaning of these variables:
    # nose_landing_gear['tyre_diameter'] = 0.8
    nose_landing_gear['piston_length'] = 0.8*main_landing_gear['piston_length']

    while (np.abs(main_landing_gear['tyre_diameter'] - main_landing_gear_tyre_diameter_new) > 0.01 or np.abs(nose_landing_gear['tyre_diameter'] - nose_landing_gear_tyre_diameter_new) > 0.01):
        vehicle, yc_trunnion = landig_gear_position(vehicle)

        (vehicle, A_min, nt_m, main_landing_gear_tyre_diameter_new, wm_max, Lpist_m_new, ds_m, Lpist_n, ds_n, wn_max, nose_landing_gear_tyre_diameter_new) = landing_gear_layout(
            vehicle)

        main_landing_gear['piston_length'] = 0.40 * \
            main_landing_gear['piston_length'] + 0.60*Lpist_m_new
        main_landing_gear['tyre_diameter'] = (
            0.40*main_landing_gear['tyre_diameter'] + 0.60*main_landing_gear_tyre_diameter_new)
        nose_landing_gear['piston_length'] = (
            Lpist_n + nose_landing_gear['piston_length'])/2
        nose_landing_gear['tyre_diameter'] = (
            nose_landing_gear_tyre_diameter_new + nose_landing_gear['tyre_diameter'])/2

        nose_landing_gear['tyre_width'] = wn_max
        nose_landing_gear['piston_diameter'] = ds_n

        main_landing_gear['tyre_width'] = wm_max
        main_landing_gear['piston_diameter'] = ds_m

    return vehicle
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
