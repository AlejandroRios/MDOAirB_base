"""
MDOAirB

Description:
    - This module caculates the turbofan performance based in EngineSim from NASA

Reference:
    - EngineSim, NASA

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
from scipy import optimize
import math

# from framework.Attributes.Atmosphere.atmosphere import atmosphere
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Attributes.Atmosphere.temperature_dependent_air_properties import FAIR
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def turbofan(altitude, mach, throttle_position, vehicle):
    """
    Description:
        - This functioncaculates the turbofan performance based in EngineSim from NASA
    Inputs:
        - altitude - [ft]
        - mach - mach number
        - throttle_position - throttle position [1.0 = 100%]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - force - [N]
        - fuel_flow - [kg/hr]
        - vehicle - dictionary containing aircraft parameters
    """
    engine = vehicle['engine']

    fan_pressure_ratio = engine['fan_pressure_ratio']
    compressor_pressure_ratio = engine['compressor_pressure_ratio']
    bypass_ratio = engine['bypass']
    fan_diameter = engine['fan_diameter']
    turbine_inlet_temperature = engine['turbine_inlet_temperature']

    # ----- MOTOR DATA INPUT --------------------------------------------------
    fan_disk_area = np.pi*(fan_diameter**2)/4
    compressor_compression_rate = compressor_pressure_ratio / \
        fan_pressure_ratio  # taxa de compressão fornecida pelo compressor
    combustor_compression_ratio = 0.99     # razão de pressão do combustor
    # temperatura na entrada da turbina
    inlet_turbine_temperature = turbine_inlet_temperature
    # at takeoff 10# increase in turbine temperatute for 5 min
    inlet_turbine_temperature_takeoff = turbine_inlet_temperature
    thermal_energy = 43260000   # Poder calorífico (J/kg)

    # ----- DESIGN POINT ------------------------------------------------------

    design_mach = 0
    design_altitude = 0
    design_throttle_position = 1.0

    # ------ EFICIÊNCIAS ------------------------------------------------------

    inlet_efficiency = 0.99
    compressor_efficiency = 0.95
    combustion_chamber_efficiency = 0.99
    turbine_efficiency = 0.98
    nozzle_efficiency = 0.99
    fan_efficiency = 0.90

    # Atualizado tabela acima em setembro de 2013 de acordo com as anotacoes
    # de aula do Ney
    # ------ TEMPERATURE RATIOS -----------------------------------------------
    temperature_ratio = 1

    # #########################################################################
    # #########  G E T  T H E R M O - DESIGN MODE #############################

    # ------ PRESSURE RATIOS --------------------------------------------------

    inlet_pressure_ratio = inlet_efficiency
    compressor_pressure_ratio = compressor_compression_rate
    combustor_pressure_ratio = combustor_compression_ratio
    fan_pressure_ratio = fan_pressure_ratio

    # ------ FREE STREAM ------------------------------------------------------
    gamma = 1.4                             # gamma do programa
    R = 287.2933                           # R do programa

    _,_,_, T_0, P_0,_, _,_= atmosphere_ISA_deviation(design_altitude, 0)

    T0_0 = (1 + (gamma-1)/2*design_mach**2)*T_0     # temperatura total
    P0_0 = P_0*(T0_0/T_0)**(gamma/(gamma-1))  # pressão total
    a0 = np.sqrt(gamma*R*T_0)                  # velocidade do som
    u0 = design_mach*a0                           # velocidade de vôo

    # ------ TEMPERATURE RATIOS -----------------------------------------------
    # ----- ENTRADA DE AR -----------------------------------------------------
    P0_1 = P0_0
    T0_1 = T0_0

    # ----- INLET -------------------------------------------------------------
    T0_2 = T0_1
    P0_2 = P0_1*inlet_pressure_ratio
    T0_2_T0_1 = T0_2/T0_1
    # ----- FAN ---------------------------------------------------------------
    _, _, _, _, Cp_2, _, gamma_2, _ = FAIR(item=1, f=0, T=T_0)

    del_h_fan = Cp_2*T0_2/fan_efficiency * \
        ((fan_pressure_ratio)**((gamma_2-1)/gamma_2)-1)
    del_t_fan = del_h_fan/Cp_2
    T0_13 = T0_2 + del_t_fan
    P0_13 = P0_2 * fan_pressure_ratio
    T0_13_T0_2 = T0_13 / T0_2

    # ----- COMPRESSOR --------------------------------------------------------
    _, _, _, _, Cp_4, _, gamma_4, _ = FAIR(item=1, f=0, T=T0_13)
    del_h_c = Cp_4*T0_13/compressor_efficiency * \
        (compressor_pressure_ratio**((gamma_4-1)/gamma_4)-1)
    tau_r_R = T0_0
    pi_cl_R = fan_pressure_ratio
    T_0R = T_0
    del_t_c = del_h_c/Cp_4
    T0_3 = T0_13 + del_t_c
    P0_3 = P0_13 * compressor_pressure_ratio
    T0_3_T0_13 = T0_3 / T0_13

    _, _, _, _, Cp_3, _, _, _ = FAIR(item=1, f=0, T=T0_3)

    tau_cl_R = T0_13_T0_2
    pi_ch_R = compressor_compression_rate

    # ----- COMBUSTOR ---------------------------------------------------------
    T0_4 = design_throttle_position * \
        inlet_turbine_temperature_takeoff    # ponto de projeto
    P0_4 = P0_3 * combustor_pressure_ratio
    T0_4_T0_3 = T0_4 / T0_3

    # ----- HIGHT TURBINE -----------------------------------------------------
    _, _, _, _, Cp_4, _, gamma_4, _ = FAIR(item=1, f=0, T=T0_4)
    del_h_ht = del_h_c
    del_t_ht = del_h_ht / Cp_4

    T0_5 = T0_4 - del_t_ht
    high_pressure_turbine_pressure_ratio = (
        1-del_h_ht/(Cp_4*T0_4*turbine_efficiency))**(gamma_4/(gamma_4-1))
    P0_5 = P0_4 * high_pressure_turbine_pressure_ratio
    T0_5_T0_4 = T0_5 / T0_4

    # ----- LOWER TURBINE -----------------------------------------------------
    _, _, _, _, Cp_5, _, gamma_5, _ = FAIR(item=1, f=0, T=T0_5)
    del_h_lt = (1 + bypass_ratio)*del_h_fan
    del_t_lt = del_h_lt / Cp_5

    T0_15 = T0_5 - del_t_lt
    low_pressure_turbine_pressure_ratio = (
        1-del_h_lt/(Cp_5*T0_5*turbine_efficiency))**(gamma_5/(gamma_5-1))
    P0_15 = P0_5 * low_pressure_turbine_pressure_ratio
    T0_15_T0_5 = T0_15 / T0_5

    epr = inlet_pressure_ratio*compressor_pressure_ratio*combustor_pressure_ratio * \
        high_pressure_turbine_pressure_ratio * \
        fan_pressure_ratio*low_pressure_turbine_pressure_ratio
    etr = T0_2_T0_1 * T0_3_T0_13*T0_4_T0_3*T0_5_T0_4*T0_13_T0_2*T0_15_T0_5
    # #########################################################################

    # ########### G E T    G E O M E T R Y  ###################################
    acore = fan_disk_area/(bypass_ratio+1)                   # área do núcleo

    #  ---- a8rat = a8 / acore ----
    a8rat = min(0.75*np.sqrt(etr/T0_2_T0_1)/epr*inlet_pressure_ratio, 1.0)
    # OBS: divide por inlet_pressure_ratio pois ele não é considerado no cálculo do EPR e
    # ETR acima no applet original

    a8 = a8rat * acore
    a4 = a8 * high_pressure_turbine_pressure_ratio * \
        low_pressure_turbine_pressure_ratio/np.sqrt(T0_5_T0_4*T0_15_T0_5)
    a4p = a8 * low_pressure_turbine_pressure_ratio/np.sqrt(T0_15_T0_5)
    a8d = a8

    if ((design_mach == mach) and (design_altitude == altitude) and (design_throttle_position == throttle_position)):
        designpoint = 1
    else:
        designpoint = 0

    T_0A = T_0
    tau_r_A = T0_0
    pi_cl_A = fan_pressure_ratio
    tau_cl_A = T0_13_T0_2
    pi_ch_A = compressor_pressure_ratio

    if not designpoint:
        # #########################################################################
        # #########  G E T  T H E R M O - WIND TUNNEL TEST ########################
        # disp('passei aqui designpoint')
        Mach = mach
        design_throttle_position = throttle_position

        # ------ FREE STREAM ------------------------------------------------------
        gamma = 1.4                             # gamma do programa
        R = 287.2933                           # R do programa

        _,_,_, T_0, P_0,rho_0, _,a0= atmosphere_ISA_deviation(altitude, 0)
        T0_0 = (1 + (gamma-1)/2*Mach**2)*T_0     # temperatura total
        P0_0 = P_0*(T0_0/T_0)**(gamma/(gamma-1))  # pressão total
        a0 = np.sqrt(gamma*R*T_0)                  # velocidade do som
        u0 = Mach*a0                           # velocidade de vôo

        # ----- ENTRADA DE AR -----------------------------------------------------
        P0_1 = P0_0
        T0_1 = T0_0

        # ----- INLET -------------------------------------------------------------
        T0_2 = T0_1
        P0_2 = P0_1*inlet_pressure_ratio

        # ----- COMBUSTOR ---------------------------------------------------------
        T0_4 = design_throttle_position*inlet_turbine_temperature
        _, _, _, _, Cp_4, _, gamma_4, _ = FAIR(item=1, f=0, T=T0_4)

        # ----- HIGHT TURBINE -----------------------------------------------------
        T0_5_T0_4 = optimize.fsolve(lambda x: find_turbine_temperature_ratio(
            x, a4p/a4, turbine_efficiency, -gamma_4/(gamma_4-1)), 0.5)[0]
        T0_5 = T0_4*T0_5_T0_4
        _, _, _, _, Cp_5, _, gamma_5, _ = FAIR(item=1, f=0, T=T0_5)
        del_t_ht = T0_5 - T0_4
        del_h_ht = del_t_ht*Cp_4
        high_pressure_turbine_pressure_ratio = (
            1-(1-T0_5_T0_4)/turbine_efficiency)**(gamma_4/(gamma_4-1))

        # ----- LOWER TURBINE -----------------------------------------------------
        T0_15_T0_5 = optimize.fsolve(lambda x: find_turbine_temperature_ratio(
            x, a8d/a4p, turbine_efficiency, -gamma_5/(gamma_5-1)), 0.5)[0]
        T0_15 = T0_5 * T0_15_T0_5
        _, _, _, _, Cp_15, _, gamma_15, _ = FAIR(item=1, f=0, T=T0_15)
        del_t_lt = T0_15 - T0_5
        del_h_lt = del_t_lt*Cp_5
        low_pressure_turbine_pressure_ratio = (
            1-(1-T0_15_T0_5)/turbine_efficiency)**(gamma_5/(gamma_5-1))

        # ----- FAN ---------------------------------------------------------------
        del_h_fan = del_h_lt / (1+bypass_ratio)
        del_t_fan = -del_h_fan / Cp_2
        T0_13 = T0_2 + del_t_fan
        _, _, _, _, Cp_13, _, gamma_13, _ = FAIR(item=1, f=0, T=T0_13)
        T0_13_T0_2 = T0_13 / T0_2
        fan_pressure_ratio = (
            1-(1-T0_13_T0_2)*fan_efficiency)**(gamma_2/(gamma_2-1))
        tau_r_A = T0_0
        pi_cl_A = fan_pressure_ratio
        T_0A = T_0

        # ----- COMPRESSOR --------------------------------------------------------
        del_h_c = del_h_ht
        del_t_c = -del_h_c / Cp_13

        T0_3 = T0_13 + del_t_c
        _, _, _, _, Cp_3, _, gamma_3, _ = FAIR(item=1, f=0, T=T0_3)
        T0_3_T0_13 = T0_3 / T0_13
        compressor_pressure_ratio = (
            1-(1-T0_3_T0_13)*compressor_efficiency)**(gamma_13/(gamma_13-1))
        T0_4_T0_3 = T0_4 / T0_3

        tau_cl_A = T0_13_T0_2
        pi_ch_A = compressor_pressure_ratio

        # ----- total pressures definition ----------------------------------------
        P0_13 = P0_2 * fan_pressure_ratio
        P0_3 = P0_13 * compressor_pressure_ratio
        P0_4 = P0_3 * combustor_pressure_ratio
        P0_5 = P0_4 * high_pressure_turbine_pressure_ratio
        P0_15 = P0_5 * low_pressure_turbine_pressure_ratio

        # ----- overall pressure & temperature ratios -----------------------------
        # print(compressor_pressure_ratio)
        # print(combustor_pressure_ratio)
        epr = inlet_pressure_ratio*compressor_pressure_ratio*combustor_pressure_ratio * \
            high_pressure_turbine_pressure_ratio * \
            fan_pressure_ratio*low_pressure_turbine_pressure_ratio
        etr = T0_2_T0_1 * T0_3_T0_13*T0_4_T0_3*T0_5_T0_4*T0_13_T0_2*T0_15_T0_5

    # ########### G E T   P E R F O R M A N C E  ##############################
    _, _, _, _, Cp_exit, _, gamma_exit, _ = FAIR(
        item=1, f=0, T=T0_5)   # gamma de saída (T0_5 ???)
    Re = (gamma_exit-1)/gamma_exit*Cp_exit       # Constante R de saída
    g = 32.2

    P0_8 = P0_0 * epr
    T0_8 = T0_0 * etr

    fact2 = -0.5*(gamma_exit+1)/(gamma_exit-1)
    fact1 = (1 + 0.5*(gamma_exit-1))**fact2
    mdot = a8*np.sqrt(gamma_exit)*P0_8*fact1 / \
        np.sqrt(T0_8*Re)  # fluxo mássico [kg/s]

    npr = max(P0_8/P_0, 1)

    fact1 = (gamma_exit-1)/gamma_exit
    uexit = np.sqrt(2*R/(gamma_exit-1)*gamma_exit*T0_8 *
                    nozzle_efficiency*(1-(1/npr)**fact1))  # ????
    
    N1ratio = ((T_0A*tau_r_A*pi_cl_A**((gamma-1)/gamma)-1)/(T_0R*tau_r_R*pi_cl_R**((gamma-1)/gamma)-1))**0.5
    N1A = N1ratio*engine['fan_rotation_ref']

    N2ratio = ((T_0A*tau_r_A*tau_cl_A*pi_ch_A**((gamma-1)/gamma)-1)/(T_0R*tau_r_R*tau_cl_R*pi_ch_R**((gamma-1)/gamma)-1))**0.5
    N2A = N2ratio*engine['compressor_rotation_ref']
    
    # Contribuição do núcleo --------------------------------------------------
    npr = max((P0_8/P_0),1)                                    # definição da razão entre a pressão de saída do motor e a pressão ambiente
    fact1 = (gamma_exit-1)/gamma_exit                                      # constante 1
    u0_8 = np.sqrt((2*R/(gamma_exit-1))*(gamma_exit*T0_8*nozzle_efficiency)*(1-(1/npr)**fact1))
     

    if (npr <= 1.893):
        pexit = P_0
    else:
        pexit = 0.52828*P0_8

    fgros = u0_8 + (pexit-P_0)*a8/mdot/g

    # ------ contribuição do fan -------------------------------------------
    snpr = P0_13 / P_0
    fact1 = (gamma-1)/gamma
    ues = np.sqrt(2*R/fact1*T0_13*nozzle_efficiency*(1-1/snpr**fact1))

    uexit = np.sqrt(2*R/(gamma_exit-1)*gamma_exit*T0_8 *
                    nozzle_efficiency*(1-(1/npr)**fact1))  # ????

    if (snpr <= 1.893):
        pfexit = P_0
    else:
        pfexit = 0.52828*P0_13

    fgros = fgros + bypass_ratio*ues + (pfexit-P_0)*bypass_ratio*acore/mdot/g

    dram = u0*(1+bypass_ratio)
    fnet = fgros - dram
    fuel_air = (T0_4_T0_3-1)/(combustion_chamber_efficiency *
                              thermal_energy/(Cp_3*T0_3)-T0_4/T0_3)

    # ####### Estimativa de Peso ##############################################
    ncomp = min(15, round(1+compressor_compression_rate/1.5))
    nturb = 2 + math.floor(ncomp/4)
    dfan = 293.02          # fan density
    dcomp = 293.02         # comp density
    dburn = 515.2          # burner density
    dturb = 515.2          # turbine density
    conv1 = 10.7639104167  # conversão de acore para ft**2

    weight = (4.4552*0.0932*acore*conv1*np.sqrt(acore*conv1/6.965) *
              ((1+bypass_ratio)*dfan*4 + dcomp*(ncomp-3) + 3*dburn + dturb*nturb))  # [N]

    # SUBROUTINE OUTPUTS
    # weightkgf = weight/9.8
    # tracaokgf = fnet*mdot/9.8  #[kgf]
    force = fnet*mdot  # [tracao em Newtons]
    # [kg/h]   # correção de 15# baseado em dados de motores reais
    fuel_flow = 1.15*fuel_air*mdot*3600


    engine['performance_parameters'] = np.array([force, fuel_flow, 0, 0, 0, 0, 0, 0, weight,fgros],dtype=object)
    engine['total_pressures'] = np.array([P0_0, P0_1, P0_2, P0_3, P0_4, P0_5, P0_8, P0_13, P0_15],dtype=object)
    engine['total_temperatures'] = np.array([T0_0, T0_1, T0_2, T0_3, T0_4, T0_5, T0_8, T0_13, T0_15],dtype=object)
    engine['exit_areas'] = np.array([acore, a8, a8*bypass_ratio, 0, 0, 0, 0, 0, 0],dtype=object)
    engine['fuel_flows'] = np.array([mdot, mdot*bypass_ratio, 0, 0, 0, 0, 0, 0, 0],dtype=object)
    engine['gas_exit_speeds'] = np.array([u0_8, ues, 0, 0, 0, 0, 0, 0, 0],dtype=object)
    engine['rotation_speeds'] = np.array([N1A, N2A, 0, 0, 0, 0, 0, 0, 0],dtype=object)

    engine['fan_rotation'] = N1A
    engine['compressor_rotation'] = N2A

    return force, fuel_flow, vehicle


def find_turbine_temperature_ratio(x, a, b, c):
    f = a - np.sqrt(x)*(1-(1-x)/b)**c
    return f


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# h = 457.2014
# mach = 0.388
# throttle_position = 0.95
# force, fuel_flow , vehicle = turbofan(h, mach, throttle_position)

# print(force)
# print(fuel_flow)
