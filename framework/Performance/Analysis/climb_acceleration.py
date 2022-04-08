"""
MDOAirB

Description:
    - This module estimate the climb acceleration

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
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Attributes.Airspeed.airspeed import V_cas_to_mach
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Analysis.descent_to_altitude import acceleration_factor_calculation
from framework.Performance.Engine.engine_performance import turbofan

import numpy as np
from joblib import dump, load
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY, fpm_to_mps, kt_to_ms, m_to_nm
GRAVITY = 9.8067
fpm_to_mps = 0.005
kt_to_ms = 0.514
m_to_nm = 0.000539957
m_to_ft = 3.281

def acceleration_to_250(rate_of_climb, climb_V_cas, delta_ISA, vehicle):
    '''
    Description:
        - This function performs the evaluationof the acceleration to 250 kt

    Inputs:
        - rate_of_climb - aircraft rate of climb [ft/min]
        - climb_V_cas - calibrated airspeed during climb [kt]
        - delta_ISA - ISA temperature deviation [deg C]
        - vehicle - dictionary containing aircraft parameters

    Outputs:
        - delta_distance - increase in distances [m]
        - delta_time - increase in time [s]
        - delta_altitude - increase in altitude [ft]
        - delta_fuel - decrease in fuel [kg]
    '''
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']

    if engine['type'] == 1:
        scaler_F = load('Performance/Engine/Turboprop/ANN_skl_force/scaler_force_PW120_in.bin') 
        nn_unit_F = load('Performance/Engine/Turboprop/ANN_skl_force/nn_force_PW120.joblib')

        scaler_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin') 
        nn_unit_FC = load('Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib')


    delta_altitude_initial = 1000
    delta_altitude_final = 0
    delta_error = np.abs(delta_altitude_final-delta_altitude_initial)
    rate_of_climb = rate_of_climb*fpm_to_mps

    while delta_error > 100:
        _, _, _, _, _, _, _, a_1 = atmosphere_ISA_deviation(
            10000, delta_ISA)
        _, _, _, _, _, _, _, a_2 = atmosphere_ISA_deviation(
            10000, delta_ISA)

        mach_1 = V_cas_to_mach(250, 10000, delta_ISA)
        mach_2 = V_cas_to_mach(climb_V_cas, 10000, delta_ISA)

        V_1 = (a_1*mach_1)*kt_to_ms  # [m/s]
        V_2 = (a_2*mach_2)*kt_to_ms  # [m/s]

        acceleration_factor_V_CAS_1, _ = acceleration_factor_calculation(
            10000+delta_altitude_initial, delta_ISA, mach_1)
        acceleration_factor_V_CAS_2, _ = acceleration_factor_calculation(
            10000+delta_altitude_initial, delta_ISA, mach_2)
        
        if engine['type'] == 0:
            # force [N], fuel flow [kg/hr]
            _, fuel_flow_1 , vehicle = turbofan(10000, mach_1, 0.95, vehicle)
            # force [N], fuel flow [kg/hr]
            _, fuel_flow_2 , vehicle = turbofan(
                10000+delta_altitude_initial, mach_2, 0.95, vehicle)
        else:
            fuel_flow_1 = nn_unit_FC.predict(scaler_FC.transform([(10000, mach_1, 0.95)]))
            fuel_flow_2 = nn_unit_FC.predict(scaler_FC.transform([(10000+delta_altitude_initial, mach_2, 0.95,)]))


        a_1 = GRAVITY*(rate_of_climb*(1+acceleration_factor_V_CAS_1))/V_1  # [m/s2]
        a_2 = GRAVITY*(rate_of_climb*(1+acceleration_factor_V_CAS_2))/V_2  # [m/s2]

        average_a = (a_1+a_2)/2  # [m/s2]
        average_fuel_flow = (fuel_flow_1+fuel_flow_2)/2  # [kg/hr]

        delta_distance = ((V_2**2 - V_1**2)/(2*average_a))*m_to_ft  # [ft]
        delta_time = ((V_2-V_1)/average_a)/60  # [min]
        delta_altitude = (rate_of_climb/fpm_to_mps)*delta_time  # [ft]
        delta_fuel = aircraft['number_of_engines'] * \
            (average_fuel_flow/60)*delta_time  # [kg]
        delta_error = np.abs(delta_altitude-delta_altitude_initial)
        delta_altitude_initial = delta_altitude

    return delta_distance, delta_time, delta_altitude, delta_fuel

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
