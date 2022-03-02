"""
MDOAirB

Description:
    - This module calculates the pax cabine length.

Reference:
    - PreSTO-Cabin - https://www.fzt.haw-hamburg.de/pers/Scholz/PreSTo/PreSTo-Cabin_Documentation_10-11-15.pdf

TODO's:
    - Clean code
    - Rename variables
    - Review this code

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


def pax_cabine_length(vehicle):
    """
    Description:
        - This module calculates the pax cabine length
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - LenFus - fuselage length - [m]
    """
    aircraft = vehicle['aircraft']
    fuselage = vehicle['fuselage']
    cabine = vehicle['cabine']

    GalleyProf = cabine['seat_prof']
    ToilletProf = cabine['toillet_prof']
    SeatProf = cabine['seat_prof']  # [m]

    # ---------------------------------- BEGIN -------------------------
    DeltaSeats = fuselage['seat_pitch'] - SeatProf
    N1 = round(aircraft['passenger_capacity']/fuselage['seat_abreast_number'])
    N2 = aircraft['passenger_capacity']/fuselage['seat_abreast_number'] - N1
    Nrow = N1
    if N2 > 0:
        Nrow = N1+1

    x0 = 1.7  # entrance area
    for j in range(Nrow):
        # seattop_fileira(x0, fuselage, SeatProf,vehicle)
        x0 = x0+SeatProf+DeltaSeats
    # **** Desenha Toillet
    # Descobre lado de maior largura
    Naux1 = round(fuselage['seat_abreast_number']/2)
    Naux2 = fuselage['seat_abreast_number']/2 - Naux1
    if Naux2 > 0:
        NseatG = Naux1 + 1
    else:
        NseatG= Naux1

    x0T = x0 - DeltaSeats + 0.1
    #LenFus = x0T
    x = []
    y = []
    x.append(x0T)
    y.append(Naux1*cabine['seat_width'] + fuselage['aisle_width'])
    x.append(x[0] + ToilletProf)
    y.append(y[0])
    x.append(x[1])
    y.append(y[1] + NseatG*cabine['seat_width'])
    x.append(x[0])
    y.append(y[2])

    x0G = x0T + 1. + ToilletProf  # walking area with 1 m large
    LenFus = x0G

    return(LenFus)


def seattop_fileira(x0, fuselage, SeatProf,vehicle):
    """
    Description:
        - This function calcute the seat top row
    Inputs:
        - x0
        - fuselage
        -  SeatProf
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - ???
    """
    cabine = vehicle['cabine']

    Naux1 = round(fuselage['seat_abreast_number']/2)
    Naux2 = fuselage['seat_abreast_number']/2 - Naux1
    if Naux2 > 0:  # numero impar de fileiras
        y0 = 0
        x = []
        y = []
        for i in range(1, Naux1):
            x.append(x0)
            y.append(y0 + (i-1)*cabine['seat_width'])
            x.append(x[0]+SeatProf)
            y.append(y[0])
            x.append(x[1])
            y.append(y[1]+cabine['seat_width'])
            x.append(x[0])
            y.append(y[2])
            # fill(x,y,'r')
            # hold on

        y0 = Naux1*cabine['seat_width'] + fuselage['aisle_width']
        for i in range(1, (fuselage['seat_abreast_number']-Naux1)):
            x.append(x0)
            y.append(y0 + (i-1)*cabine['seat_width'])
            x.append(x[0]+SeatProf)
            y.append(y[0])
            x.append(x[1])
            y.append(y[1]+cabine['seat_width'])
            x.append(x[0])
            y.append(y[2])
            # fill(x,y,'r')
            # hold on
    else:  # numero par de fileiras
        # fprintf('\n fuselage['seat_abreast_number'] � par \n')
        x = []
        y = []
        y0 = 0
        for i in range(1, int(fuselage['seat_abreast_number']/2)):
            x.append(x0)
            y.append(y0 + (i-1)*cabine['seat_width'])
            x.append(x[0]+SeatProf)
            y.append(y[0])
            x.append(x[1])
            y.append(y[1]+cabine['seat_width'])
            x.append(x[0])
            y.append(y[2])
            # fill(x,y,'r')
            # hold on

        y0 = (fuselage['seat_abreast_number']/2) * \
            cabine['seat_width'] + fuselage['aisle_width']
        for i in range(1, int(fuselage['seat_abreast_number']/2)):
            x.append(x0)

            y.append(y0 + (i-1)*cabine['seat_width'])
            x.append(x[0]+SeatProf)
            y.append(y[0])
            x.append(x[1])
            y.append(y[1]+cabine['seat_width'])
            x.append(x[0])
            y.append(y[2])

    return

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
