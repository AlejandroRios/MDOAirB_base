"""
MDOAirB

Description:
    - This module creates the landing gear layout.

Reference:
    - ROSKAM, J. Airplane Design vol. IV - Layout of landing gear and systems. DARCorp, 2010.
      TORENBEEK, E. Synthesis of subsonic airplane design. DUP / Kluwer, 1982.

TODO's:
    - Rename variables
    - Update comments
    - Update format to PEP8

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
import math
import numpy as np
from framework.Sizing.Geometry.tire_selection import tire_selection
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665
kgf_to_lbf = 2.204622622
ms_to_mph = 2.236936292
in_to_m = 0.0254
ft_to_m = 0.3048
rad_to_deg = 180/np.pi


def landing_gear_layout(vehicle):
    """
    Description:
        - This function creates the landing gear layout.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vehicle - dictionary containing aircraft parameters
        - min_angle_main_gear_to_cg - [deg]
        - pneu_number_by_strut
        - pneu_main_gear_diameter - [m]
        - pneu_main_gear_length - [m]
        - min_piston_length - [m]
        - main_gear_shock_absorber_diameter - [m]
        - nose_gear_piston_length - [m]
        - nose_gear_shock_absorber_diameter - [m]
        - pneu_nose_gear_length - [m]
        - pneu_nose_gear_diameter - [m]
    """

    nose_landing_gear = vehicle['nose_landing_gear']
    main_landing_gear = vehicle['main_landing_gear']
    aircraft = vehicle['aircraft']
    fuselage = vehicle['fuselage']
    wing = vehicle['wing']

    # CÁLCULOS
    ns = 2  # Número de struts do trem de pouso principal
    pneu_number_by_strut = 2  # Número de pneus por strut do trem principal
    nt_n = 2  # Número de pneus por strut do trem secundario
    # Pressao maxima do pneu do TDP principal
    psi_m = main_landing_gear['pressure']
    # Pressao maxima do pneu do TDP do nariz
    psi_n = nose_landing_gear['pressure']
    # DIMENSÕES PRIMÁRIAS
    ln = aircraft['forward_center_of_gravity_xposition'] - \
        nose_landing_gear['xpostion']  # Distância do tdp dianteiro ao CG [m]
    # Distância do tdp principal ao CG [m]
    lm = main_landing_gear['xpostion'] - \
        aircraft['forward_center_of_gravity_xposition']
    M = main_landing_gear['xpostion'] - \
        aircraft['after_center_of_gravity_xposition']
    F = main_landing_gear['xpostion']-nose_landing_gear['xpostion']
    L = aircraft['forward_center_of_gravity_xposition'] - \
        nose_landing_gear['xpostion']
    Fator = 1.07  # Fator de ajuste 14CFR25

    # CARGAS
    # Cargas estáticas
    OEW = 0.50*aircraft['maximum_takeoff_weight']

    fatrespdyn = 2
    mi = 0.80  # coef de fricção
    J = aircraft['zCG'] + fuselage['width']/2 + \
        main_landing_gear['piston_length'] + \
        main_landing_gear['tyre_diameter']/2
    fatdyn = fatrespdyn*mi*L*J/(F+mi*J)
    Pn_max = OEW*((F-L) + fatdyn)/F
    Pm_max = OEW*(F-M)/F
    # Carga maxima por pneu no trem principal [lbf]
    LoadMax = Fator*(Pm_max*kgf_to_lbf)/(ns*pneu_number_by_strut)
    if LoadMax > 59500:  # Verifica carga máxima por pneu e acerta número de pneus
        # Define número de pneus requerido
        pneu_number_by_strut_req = math.ceil(LoadMax/59500)
        # Verifica se número requerido de pneus é multiplo de 4
        nwtest = pneu_number_by_strut_req % ns
        if nwtest != 0:
            #    disp('passei por aqui como?')
            # Define número de pneus como um número par
            pneu_number_by_strut = (nwtest+1)*2

    # fprintf('\n MLG: Tyre number per strut #2i \n',pneu_number_by_strut)

    # Cargas dinâmicas
    axg = 0.45
    Pn_dyn = aircraft['maximum_takeoff_weight']*((lm+axg*aircraft['zCG'])/(
        lm+ln+axg*aircraft['zCG']))  # Carga no tdp dianteiro [kgf]
    # Carga por pneu
    # Carga estática por pneu no tdp principal [kgf]
    TL_m_static = LoadMax/kgf_to_lbf
    # Carga estática por pneu no tdp dianteiro [kgf]
    TL_n_static = Pn_max/nt_n*Fator
    TL_n_dyn = 0  # Carga dinâmica por pneu no tdp dianteiro [kgf]
    #
    MLW = aircraft['maximum_takeoff_weight']*0.9
    v_tfo = 1.1*np.sqrt(2*aircraft['maximum_takeoff_weight']*GRAVITY/(
        1.225*aircraft['CL_maximum_takeoff']*wing['area']))  # Velocidade máxima de lift-off [m/s]
    # Velocidade máxima de toque no pouso [m/s]
    v_land = 1.2*np.sqrt(2*MLW*GRAVITY /
                         (1.225*aircraft['CL_maximum_landing']*wing['area']))
    # Velocidade usada para dimensionamento dos pneus [m/s]
    v_qualified = max(v_tfo, v_land)
    # PNEUS
    vqualmph = v_qualified*ms_to_mph
    # Pneu tdp principal
    loadM = TL_m_static*kgf_to_lbf
    #fprintf('\n Tyre sizing for main landing gear ... ')
    TDia, TWid = tire_selection(loadM, vqualmph, psi_m, 'weight')
    # print('\n Done! \n ')
    pneu_main_gear_diameter = TDia  # Diâmetro máximo pneu tdp principal [m]
    # fprintf('\n MLF tyre diameter: #5.2f m \n', pneu_main_gear_diameter)
    D0m_min = 0.96*TDia  # Diâmetro mínimo pneu tdp principal [m]
    pneu_main_gear_length = TWid  # Largura máxima pneu tdp principal [m]
    mstatic_load = 0.4*TDia  # Raio do pneu, carregado [m]
    st_m = (0.25*(pneu_main_gear_diameter+D0m_min) -
            (mstatic_load))  # Deflexão permitida do pneu [m]
    # Pneu tdp dianteiro (Tipo VII)
    loadN = max([TL_n_static, TL_n_dyn])*kgf_to_lbf
    #fprintf('\n Tyre sizing for nose landing gear ... ')
    TDia, TWid = t = tire_selection(loadN, vqualmph, psi_n, 'size')
    #fprintf('\n Done! \n ')
    pneu_nose_gear_diameter = TDia  # Diâmetro máximo pneu tdp dianteiro [m]
    D0n_min = 0.96*TDia  # Diâmetro mínimo pneu tdp dianteiro [m]
    pneu_nose_gear_length = TWid  # Largura máxima pneu tdp dianteiro [m]
    nstatic_load = 0.4*TDia  # Raio do pneu, carregado [m]
    st_n = ((pneu_nose_gear_diameter+D0n_min)/2/2-(nstatic_load)) * \
        in_to_m          # Deflexão permitida do pneu [m]
    # SHOCK ABSORBERS (STRUTS)  - Mason
    # Eficiência para absorção de energia, pneu
    ni_t = 0.47
    # Eficiência para absorção de energia, shock absorber (considerado como oleo-pneumático)
    ni_s = 0.8
    # Fator de carga 14CFR25 [g]
    Ng = 1.5

    # TDP dianteiro
    vsink = 10*ft_to_m
    Ss_n = (0.5*Pn_max/GRAVITY*vsink**2/(1*(Pn_dyn)*Ng)-ni_t*st_n) / \
        ni_s  # Comprimento, shock absorber, tdp dianteiro [m]
    # Comprimento de projeto, shock absorber, tdp dianteiro [m]
    Ss_n = Ss_n+1/12*in_to_m
    # Diâmetro, shock absorber, tdp dianteiro [m]
    nose_gear_shock_absorber_diameter = (
        0.041+0.0025*(Pn_dyn/0.4535)**0.5)/3.28

    nose_gear_piston_length = Ss_n + 2.75*nose_gear_shock_absorber_diameter
    # MIL-L-8552 - Minimum piston length
    nose_landing_gear['piston_length'] = nose_gear_piston_length
    # fprintf('\n Nose landing gear Minimum piston length: #5.2f m \n',Lmain_landing_gear['piston_length'])
    # - TDP Principal

    # Energy calculation
    #  At takeoff
    vsink = 6*ft_to_m  # Velocidade de descida (FAR)
    Sst_ma = (0.50*vsink**2)/(ni_s*Ng*GRAVITY) - (ni_t*st_m) / \
        ni_s  # Comprimento, shock absorber, tdp principal [m]
    Sst_ma = Sst_ma+1/12*in_to_m  # Comprimento de projeto, shock absorber
    #  At landing
    vsink = 10*ft_to_m
    Ssl_ma = (0.50*vsink**2/(ni_s*Ng*GRAVITY) - ni_t*st_m) / \
        ni_s  # Comprimento, shock absorber, tdp principal [m]
    Ssl_ma = Ssl_ma+1/12*in_to_m  # Comprimento de projeto, shock absorber [m]

    Ss_m = max(Sst_ma, Ssl_ma)
    Pm = Pm_max/ns
    # Diâmetro, shock absorber, tdp principal [m]
    main_gear_shock_absorber_diameter = (0.041+0.0025*(Pm/0.4535)**0.5)/3.28
    # MIL-L-8552 - Minimum piston length [m]
    min_piston_length = Ss_m + 2.75*main_gear_shock_absorber_diameter
    main_landing_gear['piston_length'] = min_piston_length
    # fprintf('\n Main landing gear Minimum piston length: #5.2f m \n',min_piston_length)
    # ÂNGULOS
    # A (entre a vertical do tdp principal e o cg)
    # valor para A mínimo [deg]
    min_angle_main_gear_to_cg = (np.arctan(M/aircraft['zCG']))*rad_to_deg
    # valor para A máximo [deg]
    max_angle_main_gear_to_cg = (np.arctan(lm/aircraft['zCG']))*rad_to_deg
    # Beta (clearance da cauda com o solo)
    B = np.arctan((aircraft['zCG']-aircraft['zCG'])/(fuselage['length'] -
                                                     main_landing_gear['xpostion']))*rad_to_deg  # valor de beta

    main_landing_gear['unit_wheels_number'] = pneu_number_by_strut

    return vehicle, min_angle_main_gear_to_cg, pneu_number_by_strut, pneu_main_gear_diameter, pneu_main_gear_length, min_piston_length, main_gear_shock_absorber_diameter, nose_gear_piston_length, nose_gear_shock_absorber_diameter, pneu_nose_gear_length, pneu_nose_gear_diameter
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
