"""
MDOAirB

Description:
    - Flap contribution to drag coefficient

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
def drag_coefficient_flap(vehicle):
    '''
    Description:
        This function estimates the contribution to drag related to flap deflection
        the options include:
            - Internal flap: double slotted
            - External flap: single slotted
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - cd_flap - drag coefficient associated to flap deflection 
    '''
    wing = vehicle['wing']

    A_int = 0.0011
    B_int = 1
    A_ext = 0.00018
    B_ext = 2
    cflap= 1 -(wing['rear_spar_ref'] +0.02)
    # 
    cdflap_int = cflap*A_int*(wing['flap_deflection_landing']**B_int)
    cdflap_ext = cflap*A_ext*(wing['flap_deflection_landing']**B_ext)
    cd_flap     = cdflap_int + cdflap_ext

    return cd_flap
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
