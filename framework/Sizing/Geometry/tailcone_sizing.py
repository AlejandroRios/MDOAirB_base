"""
MDOAirB

Description:
    - This module performs the tailcone sizing.
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
########################################################################################
"""Importing Modules"""
########################################################################################
########################################################################################
"""Function definition"""
########################################################################################

import numpy as np
def tailcone_sizing(NPax, PEng, fuse_height, fuse_width):
    """
    Description:
        - This function performs the tailcone sizing.
    Inputs:
        - NPax - number of passengers
        - PEng - engine position
        - fuse_height - fuselage height [m]
        - fuse_width - fuselage width [m]
    Outputs:
        - ltail - tail length [m]
    """
    #  Provide a sizing of the tailcone
    fusext = 0
    if NPax <= 50:
        # passenger baggage 200 kg/m3 e 20 kg por pax
        bagvol = NPax*20/200   # m3

    if PEng == 2:
        ltail_df = 2.0
    else:
        ltail_df = 1.8  # relacao coni/diametro Roskam vol 2 pag 110

    ltail = ltail_df*fuse_width+fusext

    return ltail
