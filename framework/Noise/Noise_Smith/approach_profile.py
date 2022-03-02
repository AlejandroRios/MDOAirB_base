"""
MDOAirB

Description:
    - This module performs the simulation of approacg profile which outputs are used
    to evaluate noise

Reference:
    - 

TODO's:
    - Rename variables

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
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation

import numpy as np

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def approach_profile(takeoff_parameters,landing_parameters,aircraft_parameters,vehicle):
    """
    Description:
        - Performs the simulation of the approach and provides vectors containing the output information
    Inputs:
        - takeoff_parameters - takeoff constant parameters
        - landing_parameters - landing constant parameters
        - aircraft_parameters - dictionary containing aircraft constant parameters
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - t - time in [s]
        - d - distance [m]
        - h - altitude [m]
        - FN - required thrust [N]
        - CD - drag coefficient
        - CL - lift coefficient
        - VT - true airspeed [m/s]
    """

    GRAVITY = 9.81
    m_to_ft = 3.28084

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']


    DD                  = 4000                                                 # distância inicial da cabeceira da pista [m]
    DH                  = DD*np.tan(-landing_parameters['gamma']*np.pi/180)+50*0.3048                # altura inicial da cabeceira da pista [m]
    ## Velocidades ##
    _, _, sigma, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(DH*m_to_ft, 0)     # propriedades da atmosfera

    VSE                 = (sigma)**0.5 * np.sqrt((2*aircraft['maximum_landing_weight']*GRAVITY)/(rho_ISA*aircraft['CL_maximum_landing']*wing['area']))         
                                                                                # velocidade de estol - equivalente [m/s]
    VREFE               = takeoff_parameters['k1']*VSE                                           # velocidade de referência no pouso - equivalente [m/s]
    
    
    ## Cálculo da trajetória ##
    i1 = 0
    dt = 0.5                                                                   # intervalo de tempo para integração [s]

    t = np.array([0.0])
    d = np.array([DD])
    h = np.array([DH])
    # t = np.append(t,0.0)                                                                # tempo inicial [s])
    # d = np.append(d,DD)                                                                 # distância inicial [m]
    # h = np.append(h,DH)                                                                # altura inicial [m]

    CL = np.empty(0)
    CD = np.empty(0)
    L = np.empty(0)
    D = np.empty(0)
    FN = np.empty(0)
    RoC = np.empty(0)
    VT = np.empty(0)


    while h[i1]>(50*0.3048):
        _, _, sigma, _, _, rho_ISA, _, a = atmosphere_ISA_deviation(DH*m_to_ft, 0)     # propriedades da atmosfera
        VT= np.append(VT,VREFE/(sigma**0.5))                                    # velocidade verdadeira [m/s]
        CL= np.append(CL,aircraft['maximum_landing_weight']/(0.5*(VT[i1])**2*wing['area']*rho_ISA))               # coeficiente de sustentação  
        CD = np.append(CD,aircraft_parameters['CD_air_LG_down'])                                         # coeficiente de arrasto
        L = np.append(L,0.5*rho_ISA*wing['area']*CL[i1]*VT[i1]**2)                     # força de sustentação [N]
        D = np.append(D,0.5*rho_ISA*wing['area']*CD[i1]*VT[i1]**2)                     # força de arrasto [N]
        FN = np.append(FN,(1/aircraft['number_of_engines'])*(aircraft['maximum_landing_weight']*GRAVITY)*(np.sin(landing_parameters['gamma']*np.pi/180)+CD[i1]/CL[i1]))                                                                         # tração requerida [N/mot]
        RoC = np.append(RoC,VT[i1]*np.sin(landing_parameters['gamma']*np.pi/180))                          # razão de descida [m/s]
        i1              = i1+1                                                 # aumento do índice
        t = np.append(t, t[i1-1]+dt)                                           # insnp.tante de tempo seguinte [s]
        h = np.append(h, h[i1-1]+RoC[i1-1]*dt)                                 # altura seguinte [m]
        d = np.append(d,d[i1-1]-VT[i1-1]*dt*np.cos(landing_parameters['gamma']*np.pi/180))             # distância seguinte [m]


    return t, d, h, FN, CD, CL, VT
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
