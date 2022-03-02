"""
MDOAirB

Description:
    - Landing gear contribution to drag coefficient

Reference:
    - Drag Force and Drag Coefficient - Sadraey M., Aircraft Performance Analysis, VDM Verlag Dr. Müller, 2009
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

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def drag_coefficient_landing_gear(vehicle):
    '''
    Description:
        This function estimates the contribution to drag related to landing gear (main and nose)
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - CD_main
        - CD_nose
    '''
    main_landing_gear = vehicle['main_landing_gear']
    nose_landing_gear = vehicle['nose_landing_gear']
    wing = vehicle['wing']

    # Constants
    nstrut = 2
    CDLGw = 0.30 #(no fairing)
    CDLGs = 1.2  # vertical cylinder LD infinity (Table 3.1)
    #
    Swheel_main = main_landing_gear['tyre_diameter']*main_landing_gear['tyre_width']
    Swheel_nose = nose_landing_gear['tyre_diameter']*nose_landing_gear['tyre_width']
    Sstrut_main = main_landing_gear['piston_length']*main_landing_gear['piston_diameter']
    Sstrut_nose = nose_landing_gear['piston_length']*nose_landing_gear['piston_diameter']
    # TDP principal 
    CD_wheel_main = 0
    for i in range(nstrut):
        CD_wheel_main = CD_wheel_main+ CDLGw*main_landing_gear['unit_wheels_number']*Swheel_main/wing['area']

    CD_strut_main = nstrut*CDLGs*Sstrut_main/wing['area']
    CD_main       = CD_strut_main  + CD_wheel_main
    # TDP nariz
    CD_wheel_nose = CDLGw*nose_landing_gear['unit_wheels_number'] *Swheel_nose/wing['area']
    CD_strut_nose = CDLGs*Sstrut_nose/wing['area']
    CD_nose       = CD_strut_nose  + CD_wheel_nose
    # Miscelleaneous drag
    CD_main = 1.10*CD_main
    CD_nose = 1.10*CD_nose

    return CD_main, CD_nose

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
