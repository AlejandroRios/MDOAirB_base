"""
MDOAirB

Description:
    - This module calculates the direct operational cost using the
    Roskam formulation

Reference: 
    - Reference: ROSKAM

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
# import matplotlib.pyplot as plt
import pandas as pd
# import matplotlib.pyplot as plt

from framework.Economics.crew_salary import crew_salary

# =============================================================================
# CLASSES
# =============================================================================

class structtype():
    pass

salary = structtype()
var = structtype()

# =============================================================================
# FUNCTIONS
# =============================================================================


def direct_operational_cost(
    time_between_overhaul,
    total_mission_flight_time,
    fuel_mass,
    operational_empty_weight,
    total_mission_distance,
    max_engine_thrust,
    engines_number,
    engines_weight,
    max_takeoff_mass,
    vehicle
    ):
    """
    Description:
        - DOC calculation
    Inputs:
        - time_between_overhaul - [hr]
        - total_mission_flight_time - [min]
        - fuel_mass - [kg]
        - operational_empty_weight - [kg]
        - total_mission_distance - [nm]
        - max_engine_thrust - maximum engine thrust [kg]
        - engines_number
        - engines_weight - [kg]
        - max_takeoff_mass - maximum takeoff weight [kg]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - DOC [US$]
    """

    # Constants
    operations = vehicle['operations']
    kg2lb = 2.20462262
    kg_l_to_lb_gal = 8.3454
    N_to_lbf = 0.224809
    var.Range = total_mission_distance
    salary.Captain, salary.FO, _ = crew_salary(max_takeoff_mass)
    Fuel_price = operations['fuel_price_per_kg']

    # =============================================================================
    # Mision data
    # =============================================================================

    tbl = total_mission_flight_time/60  # [HRS] BLOCK TIME EQ 5.6
    Block_Range = var.Range  # [NM] BLOCK RANGE
    vbl = Block_Range / tbl  # [KTS]BLOCK SPEED EQ 5.5
    Uannbl = 1e3 * (3.4546 * tbl + 2.994 - (12.289 * (tbl**2) -
                                            5.6626 * (tbl) + 8.964)**(1/2))  # [HRS] ANNUAL UTILIZATION

    # ==============================
    # DOC PAG 146
    # ==============================

    # ==============================#
    # DOC FLT
    # ==============================#

    # 1) CREW COST -> Ccrew
    nc1 = 1  # NUMBER OF CAPTAIN
    nc2 = 1  # NUMBER OF FIRST OFFICER
    kj = 0.26  # FACTOR WHICH ACCOUNTS FOR VACATION PAY, COST TRAINING ...
    # THESE DATA ARE ASSUMED TO BE APPLICABLE FOR 1990
    SAL1 = salary.Captain  # SALARY CAPTAIN [USD/YR]
    SAL2 = salary.FO  # SALARY FIRST OFFICER [USD/YR]
    AH1 = 800  # [HRS]
    AH2 = 800  # [HRS]
    TEF1 = 7  # [USD/blhr]
    TEF2 = TEF1  # TRAVEL EXPENSE FACTOR
    Ccrew = (nc1*((1 + kj)/vbl)*(SAL1/AH1) + (TEF1/vbl)) + (nc2 *
                                                            ((1 + kj)/vbl)*(SAL2/AH2) + (TEF2/vbl))  # [USD/NM] EQ 5.21 PAG 109

    # 2) FUEL AND OIL COST -> Cpol (PAG 148)
    pfuel = Fuel_price*1.08  # PRICE [USD/GALLON]
    dfuel = operations['fuel_density']*kg_l_to_lb_gal  # DENSITY [LBS/GALLON]
    Wfbl = fuel_mass*kg2lb  # [LBS] OPERATIONAL MISSION FUEL
    Cpol = 1.05*(Wfbl/Block_Range)*(pfuel/dfuel)  # EQ 5.30 PAG 116 5# DO DOC

    # 3) INSURANCE COST -> Cins (PAG 148)
    # Cins = 0.02 * (DOC) # EQ 5.32 * PAG 117
    Cins = 0

    DOCflt = Ccrew + Cpol + Cins

    # ==============================#
    # DOC MAINT
    # ==============================#

    fnrev = 1.03  # NON-REVENUE FACTOR. IT ACCOUNTS FOR EXTRA MAINTENANCE COSTS INCURRED DUE FLIGHT DELAYS

    # 1) MAINTENANCE LABOR COST FOR AIRFRAME AND SYSTEMS -> Clab_ap (PAG 148)
    weight_empty_lb = operational_empty_weight*kg2lb
    weight_engine_lb = engines_weight*kg2lb
    Wa = weight_empty_lb - engines_number * \
        weight_engine_lb  # [LBS] AIRFRAME WEIGHT
    MHRmap_bl = 3*0.067*(Wa/1000)  # [mhrs/blhr] FIGURE 5.5 PAG 122
    Rlap = 15.5  # [USD/mhr] RATE PER MANHOUR
    Clab_ap = fnrev * MHRmap_bl * Rlap / vbl  # [USD/NM] EQ 5.34 PAG 119

    # 2) MAINTENANCE LABOR COST FOR ENGINES -> Clab_eng  (PAG 149)
    # BPR = razao de passagem
    Tto = engines_number*max_engine_thrust*N_to_lbf  # lbf
    Tto_Ne = Tto / engines_number  # [LBS] TAKE-OFF THRUST PER ENGINE
    Hem = time_between_overhaul  # [HRS] OVERHAUL PERIOD
    Rleng = Rlap
    MHRmeng_bl = ((0.718 + 0.0317*(Tto_Ne/1000))*(1100/Hem) +
                  0.1)  # [mhrs/blhr] FIGURE 5.6 PAG 123
    Clab_eng = fnrev*1.3*engines_number*MHRmeng_bl * \
        (Rleng/vbl)  # [USD/NM] EQ 5.36 PAG 120

    # 3) MAINTANENCE MATERIALS COST FOR AIRPLANE -> Cmat_ap (PAG 150)
    # ENGINE PRICE
    CEF = (3.10/3.02)  # FATOR DE CORRECAO DO PRECO
    # [USD] PAG 65 APPENDIX B4 EQ B10 PAG 351
    EP1989 = (10**(2.3044 + 0.8858 * (np.log10(Tto_Ne))))
    EP = CEF*EP1989  # [1992]
    # AIRPLANE PRICE
    # [USD] PAG 89 APPENDIX A9 EQ A12 PAG 331
    AEP1989 = (10**(3.3191 + 0.8043 * (np.log10(max_takeoff_mass * kg2lb))))
    AEP = CEF*AEP1989  # [1992]
    # AIRFRAME PRICE
    AFP = AEP - engines_number*EP  # [USD]

    if max_takeoff_mass * kg2lb >= 10000:  # FIGURE 5.8 PAG 126
        ATF = 1.0
    elif max_takeoff_mass * kg2lb < 10000 and max_takeoff_mass * kg2lb < 5000:
        ATF = 0.5
    else:
        ATF = 0.25

    CEFy = 1.0  # PAG 150
    Cmat_apbhr = 30 * CEFy * ATF + 0.79*1e-5 * AFP  # PAG 150 FIGURE 5.8 PAG 126
    Cmat_ap = fnrev * (Cmat_apbhr/vbl)  # [USD/NM] EQ 5.37 PAG 120

    # 4) MAINTANENCE MATERIALS COST FOR ENGINE -> Cmat_eng (PAG 150)
    KHem = 0.021 * (Hem/100) + 0.769  # FIGURE 5.11 PAG 129
    ESPPF = 1.5  # ENGINE SPARE PARTS PRICE PAG 133
    Cmat_engblhr = (5.43*1e-5 * EP * ESPPF - 0.47)/KHem  # FIGURE 5.9 PAG 127
    Cmat_eng = fnrev * 1.3 * engines_number * \
        (Cmat_engblhr/vbl)  # [USD/NM] EQ 5.38 PAG 125

    # 5) APPLIED MAINTENANCE BURDEN COST -> Camb (PAG 151)
    famb_lab = 1.2  # OVERHEAD DISTRIBUTION FACTOR LABOUR COST PAG 129 -> MIDDLE VALUE
    famb_mat = 0.55  # OVERHEAD DISTRIBUTION FACTOR MATERIAL COST PAG 129 -> MIDDLE VALUE
    Camb = fnrev * (famb_lab * (MHRmap_bl * Rlap + engines_number * MHRmeng_bl * Rleng) +
                    famb_mat * (Cmat_apbhr + engines_number * Cmat_engblhr))/vbl  # [USD/NM] EQ 5.39 PAG 125

    # TOTAL MAINTENANCE COST
    DOCmaint = Clab_ap + Clab_eng + Cmat_ap + \
        Cmat_eng + Camb  # [USD/NM] PAG 151

    # ==============================#
    # DOC DEPRECIATION
    # ==============================#
    # (PAG 130)
    # 1) AIRPLANE DEPRECIATION COST -> Cdap (PAG 151)
    fdap = 0.85  # AIRFRAME DEPRECIATION FACTOR TABELA 5.7 PAG 134
    DPap = 10  # AIRPLANE DEPRECIATION PERIOD TABELA 5.7 PAG 134
    # [USD] PAG 151 AVIONICS SYSTEM PRICE APPENDIX C THE SAME VALUE WHICH WAS USED IN THE EXAMPLE
    ASP = 2670000
    Cdap = fdap * (AEP - engines_number * EP - ASP) / \
        (DPap * Uannbl * vbl)  # [USD/NM] EQ 5.41 PAG 130

    # 2) ENGINE DEPRECIATION FACTOR -> Cdeng (PAG 152)
    fdeng = 0.85  # ENGINE DEPRECIATION FACTOR TABELA 5.7 PAG 134
    DPeng = 7  # ENGINE DEPRECIATION PERIOD TABELA 5.7 PAG 134
    # [USD/NM] EQ 5.42 PAG 131
    Cdeng = fdeng * engines_number * EP / (DPeng * Uannbl * vbl)

    # 3) AVIONICS DEPRECIATION FACTOR -> Cdav (PAG 152)
    fdav = 1.0  # AVIONICS DEPRECIATION FACTOR TABELA 5.7 PAG 134
    DPav = 5.0  # AVIONICS DEPRECIATION PERIOD TABELA 5.7 PAG 134
    Cdav = fdav * ASP / (DPav * Uannbl * vbl)  # [USD/NM] EQ 5.44 PAG 131

    # 4) AIRPLANE SPARE PARTS DEPRECIATION FACTOR -> Cdapsp (PAG 152)
    fdapsp = 0.85  # AIRPLANE SPARE PARTS DEPRECIATION FACTOR TABELA 5.7 PAG 134
    fapsp = 0.1  # AIRPLANE SPARE PARTS FACTOR PAG 132
    DPapsp = 10  # AIRPLANE SPARE PARTS DEPRECIATION PERIOD TABELA 5.7 PAG 134
    # [USD/NM] EQ 5.45 PAG 132
    Cdapsp = fdapsp * fapsp * \
        (AEP - engines_number * EP) / (DPapsp * Uannbl * vbl)

    # 5) ENGINE SPARE PARTS DEPRECIATION FACTOR -> Cdengsp (PAG 153)
    fdengsp = 0.85  # ENGINE SPARE PARTS DEPRECIATION FACTOR TABELA 5.7 PAG 134
    fengsp = 0.5  # ENGINE SPARE PARTS FACTOR PAG 133
    ESPDF = 1.5  # ENGINE SPARE PARTS PRICE FACTOR PAG 133
    DPengsp = 7.0  # ENGINE SPARE PARTS DEPRECIATION PERIOD TABELA 5.7 PAG 134
    Cdengsp = fdengsp * fengsp * engines_number * EP * ESPDF / \
        (DPengsp * Uannbl * vbl)  # [USD/NM] EQ 5.46 PAG 133

    # TOTAL DEPRECIATION COST
    DOCdepr = Cdap + Cdeng + Cdav + Cdapsp + Cdengsp  # [USD/NM] PAG 153

    # ==============================#
    # DOC TAXES
    # ==============================#
    # (PAG 130)
    # 1) COST OF LANDING FEES -> Clf (PAG 154)
    # EQ 5.49 PAG 135 AIRPLANE LANDING FEE PER LANDING
    Caplf = 0.002 * max_takeoff_mass * kg2lb
    Clf = Caplf/(vbl*tbl)  # [USD/NM] EQ 5.48 PAG 135 LANDING FEE

    # 2) COST OF NAVIGATION FEES -> Cnf (PAG 154)
    Capnf = 10  # [USD/FLIGHT] PAG 136 OPERATIONS OUTSIDE THE USA
    Cnf = Capnf/(vbl*tbl)  # [USD/NM] EQ 5.52 PAG 135

    # 3) COST OF REGISTRY FEES -> Crf (PAG 154)
    frt = 0.001 + 1e-8 * max_takeoff_mass * kg2lb
    # Crt        = frt * DOC # EQ 5.53 PAG 136
    # DOClnr    = Clf + Cnf + Crt
    # ==============================#
    # TOTAL DOC
    # ==============================#

    DOCcalc = (DOCflt + DOCmaint + DOCdepr + Clf + Cnf) / \
        (1 - (0.02 + frt + 0.07))  # [USD/NM] PAG 155
    # print('DOC = ', DOCcalc, 'USD$/nm ')
    # ==============================#
    # DOC FINANCING
    # ==============================#

    # DOCfin     = 0.07 * DOC # EQ 5.5 PAG 136

    # ==============================#
    # DOC TOTAL
    # ==============================#
    # DOC = DOC + DOCfin
    # ==============================#
    # IOC PAG 155
    # ==============================#

    fioc = - 0.5617 * np.log(Block_Range) + 4.5765  # FIGURE 5.12 PAG 139
    IOC = fioc * DOCcalc  # [USD/NM] EQ 5.56 PAG 137

    return(DOCcalc)


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# from framework.Database.Aircrafts.baseline_aircraft_parameters import *
# print(direct_operational_cost(
#     2500,
#     62,
#     1403,
#     29105,
#     358,
#     69350,
#     2,
#     6789,
#     34022,
#     vehicle))

# print(direct_operational_cost(
#     2500,
#     62,
#     1403,
#     29105,
#     358,
#     169350,
#     2,
#     10186,
#     51552,
#     vehicle))

#     def direct_operational_cost(
#     time_between_overhaul,
#     total_mission_flight_time,
#     fuel_mass,
#     operational_empty_weight,
#     total_mission_distance,
#     max_engine_thrust,
#     engines_number,
#     engines_weight,
#     max_takeoff_mass,
#     vehicle
# ):