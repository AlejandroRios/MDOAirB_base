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
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Attributes.Atmosphere.temperature_dependent_air_properties import FAIR
from framework.Performance.Engine.high_pressure_turbine_parameters import TURBC
from framework.Performance.Engine.low_pressure_turbine_parameters import TURB
from framework.Performance.Engine.compressible_flow import RGCOMPR
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def performance_analysis(h, M0, 
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
                         pi_c_max, Tt3_max, Pt3_max, NL_percent, NH_percent):
    # Turboprop Engine Performance Cycle Analysis
    # =========================================================================
    # Inputs:
    # Performance choices:
    # Flight parameters: M0, T0, P0
    # Throttle setting:Tt4
    # Design constants:
    # pi: pi_dmax, pi_b, pi_n
    # eta: eta_c, eta_tH, eta_tL, eta_b, eta_mL, eta_mH, etam_PL, eta_mPH, eta_propmax,
    # A:A4, A4.5, A5.5, A8
    # others:beta, epsilon1, epsilon2, hPR, PTOL.PTOH
    # Reference condition: data from parametric analysis
    # Engine control limits: pi_c_max, Tt3_max, Pt3_max

    # Outputs:
    # Overall performance: F/m_dot_0, S, P/m_dot_0, S_P, f0, C_C, C_prop,
    # eta_P, eta_TH, V9/a0
    # Component behavior: pi_tH, pi_tL, tau_cL, tau_cH, tau_tH, tau_tL, tau_lambda, f,
    # eta_cL, eta_cH, eta_tH, eta_tL, M9, Pt9/P9, P9/P0, , T9/T0
    # =========================================================================

    # function F, P, m0_dot, S, S_P, f0, eta_P, eta_TH, eta_O, C_c, C_prop, ...
    #     V9_a0, Pt9_P9, P9_P0, T9_T0, ...
    #     pi_cL, pi_cH, pi_tH, pi_tL, ...
    #     tau_cL, tau_cH, tau_tH, tau_tL, tau_lambda, ...
    #     tau_m1, tau_m2, f, M8, M9, ...
    #     Tt4, ...
    #     T_vec, P_vec  = TURB_PERFORMANCE(h, M0, ...
    #     Tt4, ...
    #     pi_dmax, pi_b, pi_n, ...
    #     eta_cL, eta_cH, eta_tH, eta_tL, eta_tF, eta_b, eta_mL, eta_mH, eta_mPL, eta_mPH, eta_propmax, eta_g, ...
    #     A4_A4_5, A4_5_A5, A5_A8, ...
    #     beta, epsilon1, epsilon2, h_PR, P_TOL, P_TOH, ...
    #     M0_R, T0_R, P0_R, ...
    #     pi_cL_R, pi_cH_R, pi_tH_R, pi_tL_R, pi_r_R, pi_d_R, ...
    #     tau_cL_R, tau_cH_R, tau_tH_R, tau_tL_R, tau_tF_R, ...
    #     Tt4_R , tau_m1_R, tau_m2_R, f_R, M8_R, C_TOL_R, C_TOH_R, ...
    #     F_R, m0_dot_R, S_R, MFP4_R, h0_R, tau_r_R, ...
    #     pi_c_max, Tt3_max, Pt3_max, NL_percent, NH_percent)

    while True:
        # ========================================================================
        # Inputs:
        # =========================================================================
        # --------------Flight and Atmosphere-------------------------
        delta_ISA = 0
        _, _, _, T0, P0, _, _, _ = atmosphere_ISA_deviation(h, delta_ISA)
        # Flight parametres
        P0 = P0       # KPa
        T0 = T0            # K
        # T0 and P0 are obtained from altitude
        # Others
        g_0 = 9.81
        g_c = 1
        # ========================================================================
        # Execution:
        # =========================================================================

        # Preliminary Computations

        # Far upstream of freestream ==============================================
        # in: T0 K | out: h0 J/kg Pr0 Pa R0 J/KgK
        _, h0, Pr0, _, _, R0, gamma0, a0 = FAIR(item=1, f=0, T=T0)
        V0 = M0*a0             # m/s
        ht0 = h0 + (V0**2)/(2)  # J/kg

        # in: ht0 J/kg | out: Prt0 Pa Tt0 K
        Tt0, _, Prt0, _, _, _, _, _ = FAIR(item=2, f=0, h=ht0)
        tau_r = ht0/h0
        pi_r = Prt0/Pr0
        if M0 <= 1:
            pi_d = pi_dmax

        # eta_prop as function of Mach
        if M0 <= 0.1:
            eta_prop = 10*M0*eta_propmax
        elif M0 > 0.1 and M0 <= 0.7:
            eta_prop = eta_propmax
        elif M0 > 0.7 and M0 <= 0.85:
            eta_prop = (1 - (M0-0.7)/3)*eta_propmax

        ht2 = ht0      # J/kg
        Prt2 = Prt0    # Pa

        # Set initial values
        pi_tH = pi_tH_R
        pi_tL = pi_tL_R
        tau_tH = tau_tH_R
        tau_tL = tau_tL_R
        tau_tF = tau_tF_R

        pi_cL = pi_cL_R
        pi_cH = pi_cH_R
        tau_cL = tau_cL_R
        tau_cH = tau_cH_R

        m0_dot = m0_dot_R
        tau_m1 = tau_m1_R
        tau_m2 = tau_m2_R
        f = f_R
        M4 = 1
        M4_5 = 1
        M5 = 1
        M8 = M8_R
        M9 = M8

        # Hig-pressure and low-pressure turbines:
        # Turbine HP ===========================================================
        # in: Tt4 K f | out: ht4 J/kg
        _, ht4, Prt4, _, _, _, _, _ = FAIR(item=1, f=f, T=Tt4)
        ht4_5 = ht4*tau_m1*tau_tH*tau_m2
        f4_5 = f*(1 - beta - epsilon1 - epsilon2)/(1 - beta)

        # in: ht4_5 J/kg f4_5 | out: Tt4_5i K
        Tt4_5i, _, Prt4_5i, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht4_5)
        ht5 = ht4_5*tau_tL      # J/kg

        # Turbine LP ==============================================================
        # in: ht5 J/kg f4_5 | out: Tt5i K
        Tt5i, _, Prt5i, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht5)

        # Turbine Free ============================================================
        ht6 = ht5*tau_tF
        Tt6i, _, Prt6i, _, _, _, _, _ = FAIR(item=2, f=f4_5, h=ht6)

        ###########################################################################
        # Loop 1
        ###########################################################################
        iterm0dot = 0
        m0_dot_error = 1
        m0_dot_new = 0

        while True:
            iterm0dot = iterm0dot+1
            iterM = 0
            M9_error = 1
            M9_new = M8_R

            while True:
                iterM = iterM+1

                # Compressor HP exit ======================================================
                ht3 = ht0*tau_cL*tau_cH      # J/kg
                # in: ht3 J/kg | out: Tt3 K
                Tt3, _, Prt3, _, _, _, _, _ = FAIR(item=2, f=0, h=ht3)

                # Turbine HP ==============================================================
                # in: Tt4 K f | out: ht4 J/kg
                _, ht4, Prt4, _, _, _, _, _ = FAIR(item=1, f=f, T=Tt4)
                f4_5 = f*(1 - beta - epsilon1 - epsilon2)/(1 - beta)

                # in: Tt4 K, f, A4_A4_5, M4, M4_5, eta_tH, Tt4_5i K, Tt3 K, beta, epsilon1, epsilon2 | out: Tt4_5 J/kg
                pi_tH, tau_tH, Tt4_5 = TURBC(
                    Tt4, f, A4_A4_5, M4, M4_5, eta_tH, Tt4_5i, Tt3, beta, epsilon1, epsilon2)

                # Turbine LP ==============================================================
                # in: Tti, f, (Ai /Ae), Mi, Me, Eta_t, TteR | out: pi_t, tau_t, Tte
                pi_tL, tau_tL, Tt5 = TURB(
                    Tt4_5, f4_5, A4_5_A5, M4_5, M5, eta_tL, Tt5i)
                # in: Tt5 K f4_5 | out: Prt5 Pa
                _, _, Prt5, _, _, _, _, _ = FAIR(item=1, f=f4_5, T=Tt5)

                # Turbine Free ==============================================================
                # in: Tti, f, (Ai /Ae), Mi, Me, Eta_t, TteR | out: pi_t, tau_t, Tte
                pi_tF, tau_tF, Tt6 = TURB(
                    Tt5, f4_5, A5_A8, M5, M8, eta_tF, Tt6i)
                # in: Tt6 K f4_5 | out: Prt6 Pa
                _, _, Prt6, _, _, _, _, _ = FAIR(item=1, f=f4_5, T=Tt6)

                tau_lambda = ht4/h0
                #     tau_cL  = 1 + (((1-tau_tL)*eta_mL * ((1 - beta - epsilon1 - epsilon2)*(1+f)*((tau_lambda*tau_tH)/(tau_r)) + (epsilon1*tau_tH + epsilon2)*tau_cL*tau_cH)) - (C_TOL_R/eta_mPL))
                #     tau_cL  = 1 + (((1-tau_tL)*eta_mL * ((1 - beta - epsilon1 - epsilon2)*(1+f)*((tau_lambda*tau_tH)/(tau_r)) +  epsilon2*tau_cL)) - (C_TOL_R/eta_mPL))
                tau_cL = tau_cL_R
                tau_cH = 1 + (1-tau_tH)*eta_mH * ((1 - beta - epsilon1 - epsilon2)*(1+f)
                                                  * (tau_lambda/(tau_r*tau_cL)) + epsilon1*tau_cH) - (C_TOH_R/eta_mPH)

                ht2_5 = ht2*tau_cL                         # J/kg
                ht2_5i = ht2 * (1 + eta_cL*(tau_cL - 1))   # J/kg
                ht3 = ht2_5*tau_cH                         # J/kg
                ht3i = ht2_5*(1 + eta_cH*(tau_cH - 1))      # J/kg

                # Compressor LP ===========================================================
                # in: ht2_5 J/kg | out: Tt2_5 K
                Tt2_5, _, Prt2_5, _, _, _, _, _ = FAIR(item=2, f=0, h=ht2_5)

                # in: ht2_5i J/kg | out: Prt2_5i Pa
                Tt2_5i, _, Prt2_5i, _, _, _, _, _ = FAIR(item=2, f=0, h=ht2_5i)

                # Compressor HP ===========================================================
                # in: ht3 J/kg | out: Tt3 K
                Tt3, _, Prt3, _, _, _, _, _ = FAIR(item=2, f=0, h=ht3)

                # in: ht3i J/kg | out: Prt3i Pa
                Tt3i, _, Prt3i, _, _, _, _, _ = FAIR(item=2, f=0, h=ht3i)

                # Compressor Total ========================================================
                pi_cL = Prt2_5i/Prt2
                pi_cH = Prt3i/Prt2_5
                pi_c = pi_cL*pi_cH
                tau_c = tau_cL*tau_cH

                # Iterative process to find f==============================================
                ftemp = f
                # in: Tt4 K f | out: ht4 J/kg
                _, ht4, _, _, _, _, _, _ = FAIR(item=1, f=f, T=Tt4)
                f = (ht4 - ht3)/(h_PR*eta_b - ht4)
                iterf = 0
                while True:
                    iterf = iterf+1
                    # in: Tt4 K f | out: ht4 J/kg
                    _, ht4, Prt4, _, _, _, _, _ = FAIR(item=1, f=f, T=Tt4)
                    f = (ht4 - ht3)/(h_PR*eta_b - ht4)
                    if abs(f-ftemp) > 0.0001:
                        ftemp = f
                        continue
                    else:
                        break

                # ========================= Engine Exit ====================================

                Pt9_P0 = pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n
                # in: Tt5 K f4_5 Pt9_P0 | M9_new Tt_T9
                M9_new, Tt_T9, _, _ = RGCOMPR(
                    item=3, Tt=Tt6, f=f4_5, PtdP=Pt9_P0)

                # print(M9_new)
                # print(M9)

                M9_error = np.abs(M9-M9_new)
                if M9_error > 0.01:
                    M9 = M9_new
                    continue
                else:
                    break

            if M9 > 1:
                M8 = 1
            else:
                M8 = M9

            # in: Tt4 K M4 f | MFP4
            _, _, _, MFP4 = RGCOMPR(item=1, Tt=Tt4, M=M4, f=f)

            m0_dot_new = (m0_dot_R * ((1+f_R)/(1+f))*((P0*pi_r*pi_d*pi_cL*pi_cH)/(P0_R*pi_r_R*pi_d_R*pi_cL_R*pi_cH_R)) *
                          (MFP4/MFP4_R)*np.sqrt(Tt4_R/Tt4))                  # kg/s

            m0_dot_error = np.abs((m0_dot_new - m0_dot)/m0_dot_R)

            if m0_dot_error > 0.001:
                m0_dot = m0_dot_new  # kg/s
                continue
            else:
                break

        # ========================= Overall Parameters ====================================
        Tt9 = Tt6  # K

        # in: Tt9 K M9 f4_5 | MFP4
        _, _, Pt9_P9, MFP9 = RGCOMPR(item=1, Tt=Tt9, M=M9, f=f4_5)
        P0_P9 = (Pt9_P9)/(Pt9_P0)
        P9 = P0/P0_P9

        T9 = Tt9/Tt_T9  # K

        # in: T9 K f4_5 | out: R0 J/KgK a9 m/s
        _, _, Prt9, _, _, R9, _, a9 = FAIR(item=1, f=f4_5, T=T9)

        V9 = M9*a9  # m/s

        f0 = f*(1-beta-epsilon1-epsilon2) 

        C_TOL = C_TOL_R*(m0_dot_R/m0_dot)*(h0_R/h0)
        C_TOH = C_TOL_R*(m0_dot_R/m0_dot)*(h0_R/h0)

        C_c = (gamma0-1) * M0 * ((1+f0-beta)*(V9/a0) - M0 + (1+f0-beta)
                                 * ((R9/R0)*((T9/T0)/(V9/a0))*((1-P0_P9)/gamma0)))  # dimensionless

        V9_a0 = V9/a0  # dimensionless

        # Propeller work interaction coefficient:
        C_prop = eta_prop*eta_g*(eta_mL*(1+f0-beta)*tau_lambda*tau_m1 *
                                 tau_tH*tau_m2*(1-tau_tL)*tau_tF - (C_TOL/eta_mPL))  # dimensionless

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

        # Thrust force:
        F = F_m0_dot*m0_dot  # N

        # Uninstalled thrust specific fuel consumption:
        # S11 = f0*V0/(C_TOTAL*h0) # Kg.m/J.s 
        S = (f0/(F_m0_dot))  # Kg.m/J.s > Kg/N*s
        # S_mg = S*1e6  # mg/N-s

        # S2 = f0/(F/m0_dot)

        # print(S*3600) # Kg/N*hr
        # print((S*3600)*F) # kg/hr

        fuel_flow = (S*3600)*F

        # Propulsive efficiency:
        eta_P = C_TOTAL/((C_prop/eta_prop) + ((gamma0-1)/2)
                         * ((1+f0-beta)*((V9/a0)**2) - M0**2))

        # Thermanl efficiency:
        eta_TH = (C_TOTAL + C_TOL + C_TOH)/((f0*h_PR)/h0)

        eta_O = eta_TH*eta_P

        # in: Tt0 K M0 f |  out: Tt_T, PtP, MFP0
        _, Tt_T, PtP, MFP0 = RGCOMPR(item=1, Tt=Tt0, M=M0, f=0)

        A0 = ((m0_dot*g_0)*np.sqrt(Tt0))/((P0) * MFP0)  # m2

        # in: Tt4_5 K f4_5 | out: ht4_5 J/kg
        _, ht4_5, Prt4_5, _, _, _, _, _ = FAIR(item=1, f=f4_5, T=Tt4_5)

        # Total pressures at stages (numbering reference just for plot order)
        P2 = (Prt2/Prt0)*P0
        P3 = (Prt2_5/Prt2)*P2
        P4 = (Prt3/Prt2_5)*P3
        P5 = P4*pi_b
        P6 = (Prt4_5/Prt4)*P5
        P7 = (Prt5/Prt4_5)*P6
        P8 = (Prt6/Prt5)*P7
        P9 = (Prt9/Prt6)*P8

        f4_5_R = f_R*(1-beta-epsilon1-epsilon2)/(1-beta)

        # in: Tt4_R K f_R | out: ht4_R J/kg
        _, ht4_R, _, _, _, _, _, _ = FAIR(item=1, f=f_R, T=Tt4_R)

        ht4_5_R = ht4_R*tau_m1_R*tau_tH_R*tau_m2_R

        percent_N_L = 100*np.sqrt((ht4_5*(1-tau_tL))/(ht4_5_R*(1-tau_tL_R)))
        percent_N_H = 100 * \
            np.sqrt((h0*tau_r*(tau_cL-1))/(h0_R*tau_r_R*(tau_cL_R-1)))

        V9_a0 = V9/a0
        P9_P0 = P9/P0
        T9_T0 = T9/T0

        # Vectors of total temperature and pressure along stages:
        T_vec = [T0,  Tt2_5,  Tt3,  Tt4,  Tt4_5,  Tt5,  Tt6,  Tt9,  Tt9]
        P_vec = [P0,     P3,   P4,   P5,   P6, P7, P8, P9, P9]

        if pi_c > pi_c_max or Tt3 > Tt3_max:
            Tt4_new = Tt4-10
            Tt4 = Tt4_new
            continue
        else:
            break

    return (F, fuel_flow, P, m0_dot, S, S_P, f0, eta_P, eta_TH, eta_O, C_c, C_prop,
            V9_a0, Pt9_P9, P9_P0, T9_T0,
            pi_cL, pi_cH, pi_tH, pi_tL,
            tau_cL, tau_cH, tau_tH, tau_tL, tau_lambda,
            tau_m1, tau_m2, f, M8, M9,
            Tt4,
            T_vec, P_vec)

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
