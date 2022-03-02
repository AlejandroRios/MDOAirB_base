"""
Function  :turboprop_parametric_analysis.py
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
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Atmosphere.temperature_dependent_air_properties import FAIR
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Engine.compressible_flow import RGCOMPR
import numpy as np

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def parametric_analysis(h, M0,
                        beta, C_TOL, C_TOH, h_PR,
                        epsilon1, epsilon2,
                        pi_b, pi_dmax, pi_n,
                        e_cL, e_cH, e_tH, e_tL, e_tF,
                        eta_b, eta_mL, eta_mH, eta_mPL, eta_mPH, eta_prop, eta_g,
                        pi_cL, pi_cH, tau_t, Tt4,
                        m0_dot):
    # ========================================================================
    # Inputs:
    # =========================================================================
    # --------------Flight and Atmosphere-------------------------
    delta_ISA = 0
    _, _, _, T0, P0, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)
    # Flight parametres
    P0 = P0       # Pa
    T0 = T0            # K
    # T0 and P0 are obtained from altitude
    # Others
    g_0 = 9.80665
    g_c = 1
    # ========================================================================
    # Execution:
    # =========================================================================

    # Far upstream of freestream ==============================================
    # in: T0 K | out: h0 J/kg Pr0 Pa R0 J/KgK
    _, h0, Pr0, _, _, R0, gamma0, a0 = FAIR(item=1, f=0, T=T0)
    V0 = M0*a0            # m/s
    ht0 = h0 + (V0**2)/(2)  # J/kg

    # in: ht0 J/kg | out: Prt0 Pa
    _, _, Prt0, _, _, _, _, _ = FAIR(item=2, f=0, h=ht0)
    tau_r = ht0/h0
    pi_r = Prt0/Pr0
    pi_d = pi_dmax

    # Compressor LP ===========================================================

    ht2 = ht0                   # J/kg
    Prt2 = Prt0                  # Pa
    Prt2_5 = Prt2*pi_cL**(1/e_cL)   # Pa

    # in: Prt2_5 Pa  | out: ht2_5 J/kg
    _, ht2_5, _, _, _, _, _, _ = FAIR(item=3, f=0, Pr=Prt2_5)
    tau_cL = ht2_5/ht2
    Prt2_5i = Prt2*pi_cL          # Pa

    # in: Prt2_5i Pa  | out: ht2_5i J/kg
    _, ht2_5i, _, _, _, _, _, _ = FAIR(item=3, f=0, Pr=Prt2_5i)
    eta_cL = (ht2_5i - ht2)/(ht2_5 - ht2)

    # Compressor HP ===========================================================

    Prt3 = Prt2_5*pi_cH**(1/e_cH)   # Pa
    # in: Prt3 Pa  | out: ht3 J/kg
    _, ht3, _, _, _, _, _, _ = FAIR(item=3, f=0, Pr=Prt3)
    tau_cH = ht3/ht2_5
    Prt3i = Prt2_5*pi_cH   # Pa

    Prt3_Prt2 = Prt3/Prt2

    # in: Prt3i Pa  | out: ht3i J/kg
    _, ht3i, _, _, _, _, _, _ = FAIR(item=3, f=0, Pr=Prt3i)
    eta_cH = (ht3i - ht2_5)/(ht3 - ht2_5)

    # Turbine HP ==============================================================
    # Set initial value of fuel/air ratio at station 4
    f4i = 0.06
    while True:
        # in: T4 K f4i | out: ht4 J/kg
        _, ht4, _, _, _, _, _, _ = FAIR(item=1, f=f4i, T=Tt4)
        f = (ht4 - ht3)/(eta_b*h_PR - ht4)
        if np.abs(f-f4i) > 0.0001:
            f4i = f
            continue
        else:
            break

    M4 = 1
    _, _, _, MFP4 = RGCOMPR(item=1, Tt=Tt4, M=M4, f=f)

    tau_lambda = ht4/h0
    tau_m1 = ((1-beta - epsilon1 - epsilon2)*(1+f) + ((epsilon1*tau_r*tau_cL *
                                                       tau_cH)/tau_lambda))/((1-beta - epsilon1 - epsilon2)*(1+f) + epsilon1)
    tau_tH = 1 - ((tau_r*tau_cL*(tau_cH-1) + (C_TOL/(eta_mPH)))/(eta_mH*tau_lambda*(
        (1-beta - epsilon1 - epsilon2)*(1+f) + ((epsilon1*tau_r*tau_cL*tau_cH)/tau_lambda))))
    ht4_1 = ht4*tau_m1         # J/kg
    f4_1 = f/((1+f+epsilon1)/(1-beta-epsilon1-epsilon2))

    # in: ht4_1 J/kg f4_1 | out: Prt4_1 Pa
    _, _, Prt4_1, _, _, _, _, _ = FAIR(item=2, f=f4_1, h=ht4_1)
    ht4_4 = ht4_1*tau_tH       # J/kg

    # in: ht4_4 J/kg f4_1 | out: Prt4_4 Pa
    _, _, Prt4_4, _, _, _, _, _ = FAIR(item=2, f=f4_1, h=ht4_4)
    pi_tH = (Prt4_4/Prt4_1)**(1/e_tH)
    Prt4_4i = pi_tH*Prt4_1  # Pa

    # in: Prt4_4i Pa f4_1 | out: ht4_4i J/kg
    _, ht4_4i, _, _, _, _, _, _ = FAIR(item=3, f=f4_1, Pr=Prt4_4i)
    eta_tH = (ht4_1 - ht4_4)/(ht4_1 - ht4_4i)

    # Turbine LP ==============================================================
    tau_m2 = ((1-beta - epsilon1 - epsilon2)*(1+f) + epsilon1 + (epsilon2*((tau_r*tau_cL*tau_cH) /
                                                                           (tau_lambda*tau_m1*tau_tH))))/((1-beta - epsilon1 - epsilon2)*(1+f) + epsilon1 + epsilon2)

    ht4_5 = ht4_4*tau_m2       # J/kg

    f4_5 = f/(1+f+((epsilon1+epsilon2)/(1-beta-epsilon1-epsilon2)))

    # in: ht4_5 J/kg f4_5 | out: Prt4_5 Pa
    _, _, Prt4_5, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht4_5)
    tau_tL = 1 - ((tau_r*(tau_cL-1) + (C_TOL/(eta_mPL)))/(eta_mH*tau_lambda*tau_tH*((1-beta -
                                                                                     epsilon1 - epsilon2)*(1+f) + (((epsilon1+epsilon2)/tau_tH)*tau_r*tau_cL*tau_cH/tau_lambda))))
    ht5 = ht4_5*tau_tL       # J/kg

    # in: ht5 J/kg f4_5 | out: Prt5 Pa Tt5 K
    Tt5, _, Prt5, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht5)
    pi_tL = (Prt5/Prt4_5)**(1/e_tL)
    Prt5i = pi_tL*Prt4_5       # Pa

    # in: Prt5i Pa f4_5 | out: ht5i J/kg
    _, ht5i, _, _, _, _, _, _ = FAIR(item=3, f=f4_5, Pr=Prt5i)
    eta_tL = (ht4_5 - ht5)/(ht4_5 - ht5i)

    # Turbine Free ============================================================
    ht5_5 = ht5
    # in: ht5_5 J/kg f4_5 | out: Prt5_5 Pa Tt5_5 K
    Tt5_5, _, Prt5_5, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht5_5)

    tau_tF = tau_t/tau_tL
    ht6 = ht5_5*tau_tF

    Tt6, _, Prt6, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht6)

    pi_tF = (Prt6/Prt5)**(1/e_tF)
    Prt6i = pi_tF*Prt5_5       # Pa

    # in: Prt5i Pa f4_5 | out: ht5i J/kg
    _, ht6i, _, _, _, _, _, _ = FAIR(item=3, f=f4_5, Pr=Prt6i)

    eta_tF = (ht5_5 - ht6)/(ht5_5 - ht6i)

    # ========================= Engine Exit ====================================
    ht9 = ht6       # J/kg
    Tt9 = Tt6       # K
    Prt9 = Prt6     # Pa
    f9 = f4_5
    M9 = 1

    # in: Tt0 K M f9 | Tt_T9 Pt_P9
    _, Tt_T9, Pt9_P9, _ = RGCOMPR(item=1, Tt=Tt9, M=M9, f=f9)

    Pt9_P0 = pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_tF*pi_n

    if Pt9_P0 >= Pt9_P9:
        T9 = Tt9/(Tt_T9)  # K
        # in: T9 K f9 | out: h9 J/kg Pr9 Pa R9 J/KgK a9 m/s
        _, h9, Pr9, _, _, R9, _, a9 = FAIR(item=1, f=f9, T=T9)
        P0_P9 = Pt9_P9/Pt9_P0
    else:
        Pr9 = Prt9/Pt9_P0
        # in: Pr9 Pa f9 | out: h9 J/kg T0 K
        T9, h9, _, _, _, R9, _, a9 = FAIR(item=3, f=f9, Pr=Pr9)
        P0_P9 = 1

    P9_P0 = 1/P0_P9
    T9_T0 = T9/T0

    # ========================= Engine Exit ====================================

    V9 = np.sqrt(2*g_c*(ht9-h9))  # m/s
    M9 = V9/a9
    f0 = f*(1-beta-epsilon1-epsilon2)

    C_c = (gamma0-1) * M0 * ((1+f0-beta)*(V9/a0) - M0 + (1+f0-beta) *
                             ((R9/R0)*((T9/T0)/(V9/a0))*((1-P0_P9)/gamma0)))  # dimensionless

    V9_a0 = V9/a0  # dimensionless

    # Propeller work interaction coefficient:
    # C_prop = eta_prop*eta_g*(eta_mL*(1+f0-beta)*tau_lambda*tau_m1*tau_tH*tau_m2*(1-tau_tL)*tau_tF - (C_TOL/eta_mPL)) # dimensionless
    C_prop = eta_prop*eta_g*(eta_mL*(1+f0-beta)*tau_lambda*tau_m1 *
                             tau_tH*tau_m2*tau_tL*(1-tau_tF) - (C_TOL/eta_mPL))  # dimensionless

    C_TOTAL = C_prop+C_c  # dimensionless

    # Uninstalled specific power of the engine:
    P_m0_dot = C_TOTAL*h0  # J/Kg
    # Power
    P = m0_dot*C_TOTAL*h0  # W
    # Uninstalled power specific fuel consumption:
    S_P = f0/(C_TOTAL*h0)  # Kg/J
    # S_P = 1e6*(S_P) # mg/W.s

    # Uninstalled equivalent specific thrust:
    F_m0_dot = (C_TOTAL*h0)/V0  # J.s/Kg.m > m/s
    # Thrust
    F = F_m0_dot*m0_dot  # N

    # Uninstalled thrust specific fuel consumption:
    # S_1 = f0*V0/(C_TOTAL*h0) # Kg.m/J.s > s/m
    S = (f0/(F_m0_dot))  # Kg.m/J.s
    # S_mg = S*1e6  # mg/N-s

    # S = f0/(F/m0_dot)

    # Propulsive efficiency:
    eta_P = C_TOTAL/((C_prop/eta_prop) + ((gamma0-1)/2)
                     * ((1+f0-beta)*((V9/a0)**2) - M0**2))

    # Thermanl efficiency:
    eta_TH = (C_TOTAL + C_TOL + C_TOH)/((f0*h_PR)/h0)

    return (F_m0_dot, S, P_m0_dot, S_P, f0, C_c, C_prop, eta_P, eta_TH, V9_a0,
            pi_tH, pi_tL,
            tau_cL, tau_cH, tau_tH, tau_tL, tau_tF, tau_lambda, tau_m1, tau_m2,
            f,
            eta_cL, eta_cH, eta_tH, eta_tL, eta_tF,
            M9, Pt9_P9, P9_P0, Prt3_Prt2, T9_T0,
            M0, T0, P0, F, P, pi_r, MFP4, h0, tau_r)


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
