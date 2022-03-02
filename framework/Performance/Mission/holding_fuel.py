"""
MDOAirB

Description:
    - This module calculates the holding fuel spent. 

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
import numpy as np
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def holding_fuel(altitude, delta_ISA, holding_time, vehicle):
    """
    Description:
        - This function calculates the fuel spent during holding
    Inputs:
        - altitude - [ft]
        - delta_ISA - ISA temperature deviation [deg C]
        - holding_time [min]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - fuel_mass_holding [kg]
    """

    altitude = altitude + 1500
    _, _, fuel_flow_holding = best_holding_speed(altitude, delta_ISA, vehicle)

    fuel_mass_holding = fuel_flow_holding*holding_time

    return fuel_mass_holding


def best_holding_speed(altitude, delta_ISA, vehicle):
    """
    Description:
        - This function calculates the best holding speed.
    Inputs:
        - altitude - [ft]
        - delta_ISA - ISA temperature deviation [deg C]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - V_hold_kt - hold velocity [kt]
        - CL_to_CD - lift to drag ration
        - fuel_flow_holding - fuel flow spent during holding [kg]
    """

    kt_to_ms = 0.514444
    ft_to_m = 0.3048
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    race_track_factor = 1.05
    bank_angle = 40

    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(altitude, delta_ISA)

    mach_initial = 0.10
    step = 0.001
    mach = mach_initial
    altitude = altitude+1500
    CD_to_CL_min = 10
    mach_min = 0
    nm = np.round((0.6-mach_initial)/step)

    table = []
    for i in range(1, nm):
        m = m+step
        V_tas = mach*a
        q = (1/2)*rho_ISA*(V_tas**2)
        CL = (W*g)/(q*wS)

        # Input for neural network: 0 for CL | 1 for alpha
        switch_neural_network = 0
        alpha_deg = 1
        CD_wing, _ = aerodynamic_coefficients_ANN(
            vehicle, h*ft_to_m, mach, CL, alpha_deg, switch_neural_network)

        friction_coefficient = wing['friction_coefficient']
        CD_ubrige = friction_coefficient * \
            (aircraft['wetted_area'] - wing['wetted_area']) / \
            wing['area']

        CD = CD_wing + CD_ubrige

        CL_to_CD = CL/CD

        if CL_to_CD < CD_to_CL_min:
            mach_min = m
            CD_to_CL_min = CL_to_CD

        table[i, 0] = mach[i]
        table[i, 1] = CL_to_CD[i]

    # check bank angle protection
    theta = bank_angle*(np.pi/180)
    load_dactor_z = 1/np.cos(theta)
    buffet_margin = np.sqrt(load_dactor_z)
    Vmin = buffet_margin * \
        np.sqrt(2*GRAVITY*weight/(rho_ISA*wing['area']*CL_max))*ms_to_kt
    mach_hold = buffet_margin*mach_min
    V_hold = mach_hold*a
    V_hold_kt = V_hold*ms_kt
    if V_hold_kt < V_min:
        V+hold_kt = V_min
        mach_hold = V_min/a

    # Calculate required Fuel Flow
    FnR = mass*GRAVITY/CL_to_CD

    step_throttle = 0.01
    throttle_position = 0.6
    total_thrust_force = 0

    while (total_thrust_force < FnR and throttle_position <= 1):
        thrust_force, fuel_flow , vehicle = turbofan(
            altitude, mach, throttle_position, vehicle)  # force [N], fuel flow [kg/hr]
        TSFC = (fuel_flow*GRAVITY)/thrust_force
        total_thrust_force = aircraft['number_of_engines'] * thrust_force
        throttle_position = throttle_position+step_throttle

    fuel_flow_holding = racetrack_factor*fuel_flow/60

    return V_hold_kt, CL_to_CD, fuel_flow_holding
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
