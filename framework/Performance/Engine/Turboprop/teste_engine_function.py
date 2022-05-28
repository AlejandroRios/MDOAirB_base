"""
Function  :
Title     :
Written by:
Email     : aarc.88@gmail.com
Date      :
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
Inputs:
    -
Outputs:
    -
TODO's:
    -substitute dictionary names for complete names

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
import time
from framework.Performance.Engine.Turboprop.turboprop_parametric_analysis import parametric_analysis
from framework.Performance.Engine.Turboprop.turboprop_performance_analysis import performance_analysis
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================


def engine_parameters():
    engine = {}
    engine['M0'] = 0.1188
    engine['beta'] = 0.0281
    engine['epsilon1'] = 0.0511
    engine['epsilon2'] = 0.0948
    engine['pi_d'] = 0.995
    engine['pi_n'] = 0.994
    engine['e_cL'] = 0.8434
    engine['e_cH'] = 0.8342
    engine['e_tH'] = 0.8715
    engine['e_tL'] = 0.8428
    engine['e_tF'] = 0.8287
    engine['eta_mL'] = 0.9888
    engine['eta_mH'] = 0.9751
    engine['eta_prop'] = 0.9564
    engine['pi_c'] = 15.4661
    engine['fpi_cL'] = 0.5812
    engine['tau_t'] = 0.7281
    engine['A_r1'] = 0.6000
    engine['A_r2'] = 0.4365
    engine['A_r3'] = 0.7
    engine['M_c'] = 0.7
    return engine


engine = engine_parameters()

# =========================================================================
# Execution: Parametric Analysis
# =========================================================================
h = 0          # Altitude - [ft]
M0 = engine['M0']      # Mach number
# T0 and P0 are obtained from altitude
# =========================================================================
# Aircraft system parameters:
beta = engine['beta']       # Bleed air fraction
C_TOL = 0         # Power Takeoff shaft power coefficienty - LP Spool
C_TOH = 0         # Power Takeoff shaft power coefficienty - LP Spool
# =========================================================================
# Design limitations:
# Fuel heating value:
h_PR = 42798*1000      # Fuel heating value Kerosene - [J/kg]
# ---------------------------------------------------------------------
# Component figures of merit:
epsilon1 = engine['epsilon1']     # Cooling air #1 mass flow rate
epsilon2 = engine['epsilon2']     # Cooling air #2 mass flow rate

# Total pressure ratio:
pi_b = 0.97        # Burner
pi_dmax = engine['pi_d']     # Inlet/Difusser
pi_n = engine['pi_n']        # Nozzle

# Polytropic efficiency:
e_cL = engine['e_cL']      # Compressor
e_cH = engine['e_cH']
e_tH = engine['e_tH']      # High pressure turbine
e_tL = engine['e_tL']     # Low pressure turbine
e_tF = engine['e_tF']

# Component efficiency
eta_b = eta_b = 0.99           # Burner
# Mechanical
eta_mL = engine['eta_mL']      # Low pressure spool
eta_mH = engine['eta_mH']      # High pressure spool
eta_mPL = 1         # Power takeoff- LP spool
eta_mPH = 1         # Power takeoff -HP spool
eta_prop = engine['eta_prop']        # Proppeler
eta_g = 0.98            # Gearbox
# =========================================================================
# Design choices:
pi_c = engine['pi_c']      # Total pressure ratio compressor
pi_cH = (pi_c/2)*engine['fpi_cL']
pi_cL = pi_c/pi_cH
tau_t = engine['tau_t']    # Total enthalpy ratio of turbine
Tt4 = 1466.15    # Burner exit temperature - [K]
# =========================================================================
# Others
m0_dot = 6.7   # Mass flow rate - [kg/s]


(F_m0_dot, S, P_m0_dot, S_P, f0, C_c, C_prop, eta_P, eta_TH, V9_a0,
    pi_tH, pi_tL,
    tau_cL, tau_cH, tau_tH, tau_tL, tau_tF, tau_lambda, tau_m1, tau_m2,
    f,
    eta_cL, eta_cH, eta_tH, eta_tL, eta_tF,
    M9, Pt9_P9, P9_P0, Prt3_Prt2, T9_T0,
    M0_R, T0_R, P0_R, F_R, P_R, pi_r, MFP4, h0, tau_r) = parametric_analysis(h, M0,
                                                                             beta, C_TOL, C_TOH, h_PR,
                                                                             epsilon1, epsilon2,
                                                                             pi_b, pi_dmax, pi_n,
                                                                             e_cL, e_cH, e_tH, e_tL, e_tF,
                                                                             eta_b, eta_mL, eta_mH, eta_mPL, eta_mPH, eta_prop, eta_g,
                                                                             pi_cL, pi_cH, tau_t, Tt4,
                                                                             m0_dot)

print('\n =========================================================================')
print(
    '\n Uninstalled power specific fuel consumption [Kg/J]  [lb/(hp h)] [Kg/(KW.hr)]', S_P, S_P*5.918e6, S_P*3.6e6)
print('\n Power [kW] [hp] ', (P_R/1000), (P_R*0.001341))
print('\n Thrust   [N]', (F_R))
print('\n Overall Pressure Ratio ', (Prt3_Prt2))
print('\n Compressor mass flow ', (Prt3_Prt2))
print('\n T turbine [K]', (Tt4))


# =========================================================================
# Execution Take-off: Performance Analysis
# =========================================================================
# =========================================================================
# Inputs:
# =========================================================================
# Performance choices:
# Flight parameters:
h = h
M0 = M0
# T0 and P0 are obtained from altitude
# Throttle setting
Tt4 = Tt4         # Burner exit temperature
# =========================================================================
# Design cosntants:
# Total Pressure ratios
pi_dmax = pi_dmax     # Inlet/Diffuser
pi_b = pi_b        # Burner
pi_n = pi_n        # Nozzle

# Adiabatic eff. of components
eta_c = eta_cL*eta_cH        # Compressor
eta_tH = eta_tH       # High pressure turbine
eta_tL = eta_tL       # Low pressure turbine
# Component efficiency
eta_b = eta_b       # Burner
# Mechanical
eta_mL = eta_mL      # Low pressure spool
eta_mH = eta_mH      # High pressure spool
eta_mPL = eta_mPL     # Power takeoff - LP Spool
eta_mPH = eta_mPH     # Power takeoff - HP Spool
eta_propmax = eta_prop     # Propeller

# Area ratio
A4_A4_5 = engine['A_r1']
A4_5_A5 = engine['A_r2']
A5_A8 = engine['A_r3']

# Others
beta = beta       # Bleed air fraction
epsilon1 = epsilon1     # Cooling air #1 mass flow rate
epsilon2 = epsilon2     # Cooling air #2 mass flow rate
h_PR = h_PR  # Fuel heating value
P_TOL = 0      # Power Takeoff shaft power extraction - LP Spool
P_TOH = 0      # Power Takeoff shaft power extraction - HP Spool
# =========================================================================
# Reference condition:
# Flight parameters
M0_R = M0_R
T0_R = T0_R
P0_R = P0_R
# -------------------------------------------------------------------------
# Component behavior:
pi_cL_R = pi_cL
pi_cH_R = pi_cH
pi_tH_R = pi_tH
pi_tL_R = pi_tL
pi_r_R = pi_r
pi_d_R = pi_dmax


tau_cL_R = tau_cL
tau_cH_R = tau_cH
tau_tH_R = tau_tH
tau_tL_R = tau_tL
tau_tF_R = tau_tF
tau_r_R = tau_r
# -------------------------------------------------------------------------
# Others:
Tt4_R = Tt4
tau_m1_R = tau_m1
tau_m2_R = tau_m2
f_R = f
M8_R = M9
C_TOL_R = C_TOL
C_TOH_R = C_TOH
F_R = F_R
m0_dot_R = m0_dot
S_R = S
MFP4_R = MFP4
h0_R = h0
# =========================================================================
# Engine control limits:
pi_c_max = pi_c
Tt3_max = 888.88
Pt3_max = 0
NL_percent = 1.2
NH_percent = 1.2

start_time = time.time()

(F, fuel_flow, P_TO, m0_dot, S_P_TO, S_P, f0, eta_P, eta_TH, eta_O, C_c, C_prop,
            V9_a0, Pt9_P9, P9_P0, T9_T0,
            pi_cL, pi_cH, pi_tH, pi_tL,
            tau_cL, tau_cH, tau_tH, tau_tL, tau_lambda,
            tau_m1, tau_m2, f, M8, M9,
            Tt4,
            T_vec, P_vec) = performance_analysis(h, M0,
                                            Tt4,
                                            pi_dmax, pi_b, pi_n,
                                            eta_cL, eta_cH, eta_tH, eta_tL, eta_tF, eta_b, eta_mL, eta_mH, eta_mPL, eta_mPH, eta_propmax, eta_g,
                                            A4_A4_5, A4_5_A5, A5_A8,
                                            beta, epsilon1, epsilon2, h_PR, P_TOL, P_TOH,
                                            M0_R, T0_R, P0_R,
                                            pi_cL_R, pi_cH_R, pi_tH_R, pi_tL_R, pi_r_R, pi_d_R,
                                            tau_cL_R, tau_cH_R, tau_tH_R, tau_tL_R, tau_tF_R,
                                            Tt4_R, tau_m1_R, tau_m2_R, f_R, M8_R, C_TOL_R, C_TOH_R,
                                            F_R, m0_dot_R, S_R, MFP4_R, h0_R, tau_r_R,
                                            pi_c_max, Tt3_max, Pt3_max, NL_percent, NH_percent)

print("--- %s seconds ---" % (time.time() - start_time))


print('\n =========================================================================')
print(
    '\n Uninstalled power specific fuel consumption [Kg/J]  [lb/(hp h)] [Kg/(KW.hr)]', S_P_TO, S_P_TO*5.918e6, S_P_TO*3.6e6)
print('\n Power  [kW] [hp] ', (P_TO/1000), (P_TO*0.001341))
print('\n Thrust [N]', (F))
print('\n Overall Pressure Ratio ', (Prt3_Prt2))
print('\n Compressor mass flow ', (Prt3_Prt2))
print('\n T turbine [K] ', (Tt4))

P_TO_ref = 1566
ESFC_TO_ref = 0.485

P_TO = P_TO/1000
ESFC_TO = S_P_TO*5.918e6

P_TO_error = abs(P_TO - P_TO_ref)/P_TO_ref
ESFC_TO_error = abs(ESFC_TO - ESFC_TO_ref)/ESFC_TO_ref

print('\n Error P ', (P_TO_error*100))
print('\n Error SFC', (ESFC_TO_error*100))


h = 25000
M0 = 0.7

(F, fuel_flow, P_TO, m0_dot, S_P_TO, S_P, f0, eta_P, eta_TH, eta_O, C_c, C_prop,
            V9_a0, Pt9_P9, P9_P0, T9_T0,
            pi_cL, pi_cH, pi_tH, pi_tL,
            tau_cL, tau_cH, tau_tH, tau_tL, tau_lambda,
            tau_m1, tau_m2, f, M8, M9,
            Tt4,
            T_vec, P_vec) = performance_analysis(h, M0,
                                            Tt4,
                                            pi_dmax, pi_b, pi_n,
                                            eta_cL, eta_cH, eta_tH, eta_tL, eta_tF, eta_b, eta_mL, eta_mH, eta_mPL, eta_mPH, eta_propmax, eta_g,
                                            A4_A4_5, A4_5_A5, A5_A8,
                                            beta, epsilon1, epsilon2, h_PR, P_TOL, P_TOH,
                                            M0_R, T0_R, P0_R,
                                            pi_cL_R, pi_cH_R, pi_tH_R, pi_tL_R, pi_r_R, pi_d_R,
                                            tau_cL_R, tau_cH_R, tau_tH_R, tau_tL_R, tau_tF_R,
                                            Tt4_R, tau_m1_R, tau_m2_R, f_R, M8_R, C_TOL_R, C_TOH_R,
                                            F_R, m0_dot_R, S_R, MFP4_R, h0_R, tau_r_R,
                                            pi_c_max, Tt3_max, Pt3_max, NL_percent, NH_percent)

print("--- %s seconds ---" % (time.time() - start_time))


print('\n =========================================================================')
print(
    '\n Uninstalled power specific fuel consumption [Kg/J]  [lb/(hp h)] [Kg/(KW.hr)]', S_P_TO, S_P_TO*5.918e6, S_P_TO*3.6e6)
print('\n Power  [kW] [hp] ', (P_TO/1000), (P_TO*0.001341))
print('\n Thrust [N]', (F))
print('\n Overall Pressure Ratio ', (Prt3_Prt2))
print('\n Compressor mass flow ', (Prt3_Prt2))
print('\n T turbine [K] ', (Tt4))
