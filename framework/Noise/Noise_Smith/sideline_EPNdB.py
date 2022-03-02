"""
MDOAirB

Description:
    - This module computes airplane sideline noise during takeoff

Reference:
    - Smith

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
from framework.Noise.Noise_Smith.takeoff_EPNdB import takeoff_EPNdB
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def sideline_EPNdB(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec,throttle_position,takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle):
    """
    Description:
        - This function calculates the airplane sideline effective percibed noise during takeoff

    Inputs:
        - time_vec - vector containing time [s]
        - velocity_vec - vector containing speed [m/s]
        - distance_vec - vector containing distances [m]
        - velocity_horizontal_vec - vector containing horizontal speed [m/s]
        - altitude_vec - vector containing altitude [m]
        - velocity_vertical_vec - vector containing horizontal speed [m/s]
        - trajectory_angle_vec - vector containing the trajectory angle [deg]
        - fan_rotation_vec - vector containing the fan rotation [rpm]
        - compressor_rotation_vec - vector containing the compressor rotation speed [rpm]
        - throttle_position - throttle position [1.0 = 100%]
        - takeoff_parameters - takeoff constant parameters
        - noise_parameters - noise constant parameters
        - aircraft_geometry - dictionary containing aircraft constant parameters
        - engine_parameters - engine constant parameters
        - vehicle - dictionary containing aircraft parameters

    Outputs:
        - SLEPNdB
        - XApeak
    """

    ## CORPO DA FUNÇÃO ##
    ## Cálculo do ruído lateral por aproximações sucessivas do ponto onde ele é mais intenso
    # 1ª aproximação (milhares de metros)
    Dmax1               = takeoff_parameters['trajectory_max_distance']
    Dmin1               = 0
    DD                  = 1000
    ncount              = int((Dmax1-Dmin1)/DD+1)
    XA                  = np.zeros(ncount)
    SLnoise             = np.zeros(ncount)


    noise_parameters['takeoff_lateral_distance_mic'] = noise_parameters['sideline_lateral_distance_mic']
    for i in range(ncount):
        XA[i] = DD*(i)+Dmin1
        noise_parameters['takeoff_longitudinal_distance_mic']           = XA[i]
        SLnoise[i]      = takeoff_EPNdB(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec, throttle_position, takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle)


    S1max      = max(SLnoise)
    I1max = np.argmax(SLnoise)
    # 2ª aproximação (centenas de metros)
    Dmax2               = DD*(I1max)+Dmin1
    Dmin2               = DD*(I1max-2)+Dmin1
    DD                  = 100
    ncount              = int((Dmax2-Dmin2)/DD+1)
    XA                  = np.zeros(ncount)
    SLnoise             = np.zeros(ncount)

    for i in range(ncount):
        XA[i] = DD*(i)+Dmin2
        noise_parameters['takeoff_longitudinal_distance_mic']           = XA[i]
        SLnoise[i]      = takeoff_EPNdB(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec, throttle_position, takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle)

    S1max      = max(SLnoise)
    I1max = np.argmax(SLnoise)
    # 3ª aproximação (dezenas de metros)
    Dmax3               = DD*(I1max)+Dmin2
    Dmin3               = DD*(I1max-2)+Dmin2
    DD                  = 10
    ncount              = int((Dmax3-Dmin3)/DD+1)
    XA                  = np.zeros(ncount)
    SLnoise             = np.zeros(ncount)
    for i in range(ncount):
        XA[i] = DD*(i)+Dmin3
        noise_parameters['takeoff_longitudinal_distance_mic']           = XA[i]
        SLnoise[i]      = takeoff_EPNdB(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec, throttle_position, takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle)

    S1max      = max(SLnoise)
    I1max = np.argmax(SLnoise)
    # 4ª aproximação (metros)
    Dmax4               = DD*(I1max)+Dmin3
    Dmin4               = DD*(I1max-2)+Dmin3
    DD                  = 1
    ncount              = int((Dmax4-Dmin4)/DD+1)
    XA                  = np.zeros(ncount)
    SLnoise             = np.zeros(ncount)
    for i in range(ncount):
        XA[i] = DD*(i)+Dmin4
        noise_parameters['takeoff_longitudinal_distance_mic']           = XA[i]
        SLnoise[i]      = takeoff_EPNdB(time_vec,velocity_vec,distance_vec,velocity_horizontal_vec,altitude_vec,velocity_vertical_vec,trajectory_angle_vec,fan_rotation_vec,compressor_rotation_vec, throttle_position, takeoff_parameters,noise_parameters,aircraft_geometry,engine_parameters,vehicle)

    S1max      = max(SLnoise)
    I1max = np.argmax(SLnoise)


    ## SAÍDA DOS DADOS ## 
    SLEPNdB             = SLnoise[I1max]
    XApeak              = XA[I1max]

    return SLEPNdB, XApeak


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
