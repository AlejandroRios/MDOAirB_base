"""
MDOAirB

Description:
    - This module performs the mission sizing analysis
    of the aircraft. This mission consideres the maximum range as input.

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
import math

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from datetime import datetime

from framework.Attributes.Airspeed.airspeed import (V_cas_to_mach,
                                                    crossover_altitude,
                                                    mach_to_V_cas)
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import \
    atmosphere_ISA_deviation
# from framework.baseline_aircraft_parameters import *
from framework.Economics.crew_salary import crew_salary
from framework.Economics.direct_operational_cost import direct_operational_cost
from framework.Performance.Analysis.climb_integration import climb_integration
from framework.Performance.Analysis.cruise_performance import *
from framework.Performance.Analysis.descent_integration import \
    descent_integration
from framework.Performance.Analysis.maximum_range_cruise import \
    maximum_range_mach
from framework.Performance.Analysis.mission_altitude import (maximum_altitude,
                                                             optimum_altitude)
from framework.Sizing.performance_constraints import (regulated_landing_weight,
                                                      regulated_takeoff_weight)

from framework.Performance.Engine.engine_performance import turbofan

from framework.Weights.weights import aircraft_empty_weight
from framework.Performance.Mission.reserve_fuel import reserve_fuel
from framework.Sizing.Geometry.sizing_tail import sizing_tail
from framework.utilities.logger import get_logger
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])


global GRAVITY
GRAVITY = 9.80665
gallon_to_liter = 3.7852
feet_to_nautical_miles = 0.000164579


def mission_sizing(vehicle, airport_departure, airport_destination):
    """
    Description:
        - This function performs the mission sizing analysis
        of the aircraft. This mission consideres the maximum range as input.

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - airport_destination
    Outputs:
        - vehicle - dictionary containing aircraft parameters
        - MTOW_calculated - [kg]
        - total_mission_burned_fuel_complete - [´kg]
        - landing_weight - [kg]
    """

    start_time = datetime.now()
    # log.info('---- Start mission sizing function ----')

    tolerance = 100

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    performance = vehicle['performance']

    operations = vehicle['operations']
    performance = vehicle['performance']

    passenger_capacity_initial = aircraft['passenger_capacity']
    engines_number = aircraft['number_of_engines']
    max_engine_thrust = engine['maximum_thrust']

    heading = 0

    # Operations and certification parameters:
    buffet_margin = operations['buffet_margin']  # [g]
    residual_rate_of_climb = performance['residual_rate_of_climb']  # [ft/min]
    ceiling = operations['max_ceiling']  # [ft] 
    descent_altitude = operations['descent_altitude']
    # Network and mission parameters
    holding_time = operations['holding_time']  # [min]
    fuel_density = operations['fuel_density']  # [kg/l]
    fuel_price_per_kg = operations['fuel_price_per_kg']  # [per kg]
    fuel_price = (fuel_price_per_kg/fuel_density)*gallon_to_liter
    time_between_overhaul = operations['time_between_overhaul']  # [hr]
    taxi_fuel_flow_reference = operations['taxi_fuel_flow_reference']  # [kg/min]
    contingency_fuel_percent = operations['contingency_fuel_percent']
    min_cruise_time = operations['min_cruise_time']  # [min]
    go_around_allowance = operations['go_around_allowance']

    # Initial flight speed schedule
    climb_V_cas = operations['climb_V_cas']
    mach_climb = operations['mach_climb']
    cruise_V_cas = operations['cruise_V_cas']
    mach_cruise_altertaive = operations['mach_cruise_alternative']
    descent_V_cas = operations['descent_V_cas']
    mach_descent = operations['mach_descent']

    delta_ISA = operations['flight_planning_delta_ISA']

    takeoff_fuel = aircraft['maximum_takeoff_weight']*0.005


    max_takeoff_mass = aircraft['maximum_takeoff_weight'] - takeoff_fuel

    takeoff_allowance_mass = operations['takeoff_allowance']
    approach_allowance_mass = operations['approach_allowance_mass']
    average_taxi_in_time = operations['average_taxi_in_time']
    average_taxi_out_time = operations['average_taxi_out_time']

    payload = round(
        aircraft['passenger_capacity']
        * operations['passenger_mass']
    )

    initial_altitude = airport_departure['elevation']

    # start_time = datetime.now()
    # Maximum altitude calculation
    max_altitude, rate_of_climb = maximum_altitude(
        vehicle,
        initial_altitude,
        ceiling,
        max_takeoff_mass,
        climb_V_cas,
        mach_climb,
        delta_ISA
    )
    # end_time = datetime.now()
    # print('max alt calculation time: {}'.format(end_time - start_time))
    
    # start_time = datetime.now()
    # Optimal altitude calculation
    optim_altitude, rate_of_climb, _ = optimum_altitude(
        vehicle,
        initial_altitude,
        ceiling,
        max_takeoff_mass,
        climb_V_cas,
        mach_climb,
        delta_ISA
    )
    # end_time = datetime.now()
    # print('opt alt calculation time: {}'.format(end_time - start_time))

    g_climb = 4/1000
    g_descent = 3/1000
    K1 = g_climb + g_descent
    # Minimum distance at cruise stage
    Dmin = 10*operations['mach_cruise']*min_cruise_time

    K2 = (
        performance['range']
        - Dmin
        + g_climb*(airport_departure['elevation'] + 1500)
        + g_descent*(airport_destination['elevation'] + 1500)
    )
    max_altitude_check = K2/K1

    if max_altitude_check > ceiling:
        max_altitude_check = ceiling

    if max_altitude > max_altitude_check:
        max_altitude = max_altitude_check

    if optim_altitude < max_altitude:
        final_altitude = optim_altitude
    else:
        final_altitude = max_altitude

    'TODO: this should be replaced for information from ADS-B'
    # Check for next lower feasible RVSN FK check according to
    # present heading
    final_altitude = 1000*(math.floor(final_altitude/1000))

    flight_level = final_altitude/100
    odd_flight_level = [
        90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330,
        350, 370, 390, 410, 430, 450, 470, 490, 510
    ]

    even_flight_level = [
        80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
        340, 360, 380, 400, 420, 440, 460, 480, 500, 520
    ]

    if (heading > 0 and heading <= 180):
        flight_level = min(
            odd_flight_level, key=lambda x: abs(x-flight_level)
        )
        final_altitude = flight_level*100
    elif (heading > 180 and heading <= 360):
        flight_level = min(
            even_flight_level, key=lambda x: abs(x-flight_level)
        )
        final_altitude = flight_level*100
    
    # start_time = datetime.now()
    # Initial climb fuel estimation
    initial_altitude = initial_altitude + 1500
    _, _, total_burned_fuel0, _, _, _, _, _, _, _, _, _, _, _ = climb_integration(
        max_takeoff_mass,
        mach_climb,
        climb_V_cas,
        delta_ISA,
        final_altitude,
        initial_altitude,
        vehicle
    )
    # end_time = datetime.now()
    # print('climb integration calculation time: {}'.format(end_time - start_time))
    

    # start_time = datetime.now()
    # Calculate best cruise mach
    mass_at_top_of_climb = max_takeoff_mass - total_burned_fuel0
    operations['mach_cruise'] = maximum_range_mach(
        mass_at_top_of_climb,
        final_altitude,
        delta_ISA,
        vehicle
    )
    # end_time = datetime.now()
    # print('best cruise mach calc time: {}'.format(end_time - start_time))

    mach_climb = operations['mach_cruise']
    mach_descent = operations['mach_cruise']
    
    # start_time = datetime.now()
    # Recalculate climb with new mach
    final_distance, total_climb_time, total_burned_fuel, final_altitude, _, _, _, _, _, _, _, _, _, _ = climb_integration(
        max_takeoff_mass,
        mach_climb,
        climb_V_cas,
        delta_ISA,
        final_altitude,
        initial_altitude,
        vehicle
    )
    # end_time = datetime.now()
    # print('climb integration re calculation time: {}'.format(end_time - start_time))


    mass_at_top_of_climb = max_takeoff_mass - total_burned_fuel

    initial_cruise_altitude = final_altitude

    distance_climb = final_distance*feet_to_nautical_miles

    distance_cruise = performance['range'] - distance_climb

    altitude = initial_cruise_altitude

    flag = 1
    iteration = 0
    while flag == 1 and iteration <100:

        # start_time = datetime.now()
        transition_altitude = crossover_altitude(
            operations['mach_cruise'],
            cruise_V_cas,
            delta_ISA
        )
        # end_time = datetime.now()
        # print('cross over altitude calculation time: {}'.format(end_time - start_time))
        _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(
            initial_cruise_altitude,
            delta_ISA
        )

        if altitude <= 10000:
            mach = V_cas_to_mach(250, altitude, delta_ISA)

        if (altitude > 10000 and altitude <= transition_altitude):
            mach = V_cas_to_mach(cruise_V_cas, altitude, delta_ISA)

        if altitude > transition_altitude:
            mach = operations['mach_cruise']

        # if distance_cruise < 0:
        #     print('duh')
        
        # start_time = datetime.now()
        # Breguet calculation type for cruise performance
        total_cruise_time, final_cruise_mass = cruise_performance_simple(
            altitude,
            delta_ISA,
            mach,
            mass_at_top_of_climb,
            distance_cruise,
            vehicle
        )
        # print(total_cruise_time)
        # print(final_cruise_mass)
        # end_time = datetime.now()
        # print('cruise performance calculation time: {}'.format(end_time - start_time))

        final_cruise_altitude = altitude
        
        # start_time = datetime.now()
        final_distance, total_descent_time, total_burned_fuel, final_altitude, _, _, _, _, _, _, _, _, _, _ = descent_integration(
            final_cruise_mass,
            mach_descent,
            descent_V_cas,
            delta_ISA,
            descent_altitude,
            final_cruise_altitude,
            vehicle
        )
        # end_time = datetime.now()
        # print('descent integration calculation time: {}'.format(end_time - start_time))
        
        distance_descent = final_distance*feet_to_nautical_miles

        # print('dclimb',distance_climb)
        # print('dcruiz',distance_cruise)
        # print('ddesc',distance_descent)
        distance_mission = distance_climb + distance_cruise + distance_descent
        # print('distance_mission',distance_mission)
        distance_error = np.abs(performance['range'] - distance_mission)

        iteration = iteration + 1

        if distance_error <= 5:
            flag = 0
            # print('Mission distance:', distance_mission)
        else:
            if distance_mission > performance['range']:
                distance_cruise = distance_cruise - distance_error*0.95
            else:
                distance_cruise = distance_cruise + distance_error*0.95

    if iteration >= 200:
        raise ValueError

    final_mission_mass = final_cruise_mass - total_burned_fuel
    total_mission_burned_fuel = max_takeoff_mass - final_mission_mass
    total_mission_flight_time = total_climb_time + \
        total_cruise_time + total_descent_time
    total_mission_distance = distance_mission


    # Alternative airport climb
    initial_altitude = airport_destination['elevation'] + 1500
    max_alternative_mass = final_mission_mass

    max_altitude, rate_of_climb = maximum_altitude(
        vehicle,
        initial_altitude,
        ceiling,
        max_alternative_mass,
        climb_V_cas,
        mach_climb,
        delta_ISA
    )

    # Optimal altitude calculation
    optim_altitude, rate_of_climb, _ = optimum_altitude(
        vehicle,
        initial_altitude,
        ceiling,
        max_alternative_mass,
        climb_V_cas,
        mach_climb,
        delta_ISA
    )

    g_climb = 4/1000
    g_descent = 3/1000
    K1 = g_climb + g_descent
    # Minimum distance at cruise stage
    Dmin = 10*operations['mach_cruise']*min_cruise_time

    K2 = (
        performance['range']
        - Dmin
        + g_climb*(airport_departure['elevation'] + 1500)
        + g_descent*(airport_destination['elevation'] + 1500)
    )
    max_altitude_check = K2/K1

    if max_altitude_check > ceiling:
        max_altitude_check = ceiling

    if max_altitude > max_altitude_check:
        max_altitude = max_altitude_check

    if optim_altitude < max_altitude:
        final_altitude = optim_altitude
    else:
        final_altitude = max_altitude

    # Check for next lower feasible RVSN FK check according to
    # present heading
    final_altitude = 1000*(math.floor(final_altitude/1000))

    flight_level = final_altitude/100

    if (heading > 0 and heading <= 180):
        flight_level = min(
            odd_flight_level, key=lambda x: abs(x-flight_level)
        )
        final_altitude = flight_level*100
    elif (heading > 180 and heading <= 360):
        flight_level = min(
            even_flight_level, key=lambda x: abs(x-flight_level)
        )
        final_altitude = flight_level*100

    # Initial climb fuel estimation
    # initial_altitude = initial_altitude
    _, _, total_burned_fuel0, _, _, _, _, _, _, _, _, _, _, _ = climb_integration(
        max_alternative_mass,
        mach_climb,
        climb_V_cas,
        delta_ISA,
        final_altitude,
        initial_altitude,
        vehicle
    )

    # Calculate best cruise mach
    mass_at_top_of_climb = max_alternative_mass - total_burned_fuel0
    operations['mach_cruise'] = maximum_range_mach(
        mass_at_top_of_climb,
        final_altitude,
        delta_ISA,
        vehicle
    )
    mach_climb = operations['mach_cruise']
    mach_descent = operations['mach_cruise']

    # Recalculate climb with new mach
    final_distance, total_climb_time, total_burned_fuel, final_altitude, _, _, _, _, _, _, _, _, _, _ = climb_integration(
        max_alternative_mass,
        mach_climb,
        climb_V_cas,
        delta_ISA,
        final_altitude,
        initial_altitude,
        vehicle
    )

    mass_at_top_of_climb = max_alternative_mass - total_burned_fuel

    initial_cruise_altitude = final_altitude

    distance_climb = final_distance*feet_to_nautical_miles

    distance_cruise = operations['alternative_airport_distance'] - distance_climb

    # if distance_cruise < 0:
    #     print('duh')

    altitude = initial_cruise_altitude

    flag = 1
    iteration = 0

    while flag == 1 and iteration <100:

        transition_altitude = crossover_altitude(
            operations['mach_cruise'],
            cruise_V_cas,
            delta_ISA
        )
        _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(
            initial_cruise_altitude,
            delta_ISA
        )

        if altitude <= 10000:
            mach = V_cas_to_mach(250, altitude, delta_ISA)

        if (altitude > 10000 and altitude <= transition_altitude):
            mach = V_cas_to_mach(cruise_V_cas, altitude, delta_ISA)

        if altitude > transition_altitude:
            mach = operations['mach_cruise']

        # Breguet calculation type for cruise performance
        total_cruise_time, final_cruise_mass = cruise_performance_simple(
            altitude,
            delta_ISA,
            mach,
            mass_at_top_of_climb,
            distance_cruise,
            vehicle
        )

        final_cruise_altitude = altitude

        # final_distance, total_descent_time, total_burned_fuel, final_altitude = descent_integration(
        #     final_cruise_mass,
        #     mach_descent,
        #     descent_V_cas,
        #     delta_ISA,
        #     descent_altitude,
        #     final_cruise_altitude,
        #     vehicle
        # )

        # distance_descent = final_distance*feet_to_nautical_miles
        # distance_mission = distance_climb + distance_cruise + distance_descent

        distance_mission = distance_climb + distance_cruise 
        distance_error = np.abs(operations['alternative_airport_distance'] - distance_mission)

        iteration = iteration + 1

        if distance_error <= 5:
            flag = 0
            # print('Mission distance:', distance_mission)
        else:
            if distance_mission > operations['alternative_airport_distance'] :
                distance_cruise = distance_cruise - distance_error*0.95
            else:
                distance_cruise = distance_cruise + distance_error*0.95
    
    if iteration >= 200:
        raise ValueError

    final_mission_mass_alternative = final_cruise_mass - total_burned_fuel
    total_mission_burned_fuel_alternative = max_alternative_mass - final_mission_mass_alternative
    total_mission_flight_time_alternative = total_climb_time + \
        total_cruise_time + total_descent_time
    total_mission_distance_alternative = distance_mission

    # Reserve fuel
    # 0 if simplified computation | 1 if full computation
    reserve_fuel_calculation_type = 0
    contingency_fuel = contingency_fuel_percent*(total_mission_burned_fuel + total_mission_burned_fuel_alternative)

    landing_weight = max_takeoff_mass - total_mission_burned_fuel

    taxi_time = (0.51 * 1E-6 * max_takeoff_mass + 0.125)*60
    taxi_fuel = 0.005 * max_takeoff_mass

    total_mission_burned_fuel_complete = total_mission_burned_fuel + total_mission_burned_fuel_alternative + taxi_fuel
    total_mission_time_complete = total_mission_flight_time + total_mission_flight_time_alternative + taxi_time
    total_mission_distance_complete = total_mission_distance


    engine_static_thrust, fuel_flow , vehicle = turbofan(
        0, 0, 1, vehicle)

    MTOW_error = 1E06
    MTOW_calculated = max_takeoff_mass
    # np.save('vehicle_test.npy', vehicle)
    while MTOW_error > 50:

        vehicle = aircraft_empty_weight(vehicle, max_takeoff_mass, total_mission_burned_fuel,
                                        engine_static_thrust, operations['mach_maximum_operating']-0.02, max_altitude)
                                        
        MTOW_new = aircraft['payload_weight'] + aircraft['operational_empty_weight'] + \
            aircraft['crew_number']*100 + total_mission_burned_fuel_complete*1.0025
        MTOW_error = abs(MTOW_calculated - MTOW_new)

        MTOW_calculated = 0.2*MTOW_calculated + 0.8*MTOW_new

        horizontal_tail_wetted_area_old = horizontal_tail['wetted_area']
        vertical_tail_wetted_area_old = vertical_tail['wetted_area']
        aircraft_wetted_area_old = aircraft['wetted_area']

        vehicle = sizing_tail(vehicle, mach, altitude)

        aircraft['wetted_area'] = aircraft_wetted_area_old + horizontal_tail['wetted_area'] + \
            vertical_tail['wetted_area'] - \
            (horizontal_tail_wetted_area_old + vertical_tail_wetted_area_old)

    # log.info('---- End mission sizing function ----')
    # end_time = datetime.now()
    # log.info('Mission sizing execution time: {}'.format(end_time - start_time))

    return vehicle, MTOW_calculated, total_mission_burned_fuel_complete, landing_weight


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# vehicle = np.load('vehicle_test.npy',allow_pickle='TRUE').item()
# # print(vehicle['wing']) # displays "world"
# aircraft = vehicle['aircraft']
# horizontal_tail = vehicle['horizontal_tail']
# vertical_tail = vehicle['vertical_tail']
# engine = vehicle['engine']
# fuselage = vehicle['fuselage']
# # print(aircraft)
# # print('---------------------------------------------------------')
# mach = 0.8
# altitude = 41000
# vehicle = mission(vehicle)
# # aircraft = vehicle['aircraft']
# # horizontal_tail = vehicle['horizontal_tail']
# # vertical_tail = vehicle['vertical_tail']

# print(vertical_tail)
