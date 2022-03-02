"""
MDOAirB

Description:
    - This module performs the mission analysis of the aircraft for alternate airport.

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
import math
import numpy as np

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665
gallon_to_liter = 3.7852
feet_to_nautical_miles = 0.000164579

def mission_alternative(vehicle,landing_weight):
    """
    Description:
        - This function performs the mission analysis of the aircraft for alternate airport.
 
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - landing_weight - [kg]
    Outputs:
        - total_mission_burned_fuel - [kg]
    """

    performance = vehicle['performance']

    tolerance = 100

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    wing = vehicle['wing']

    airport_departure = vehicle['airport_departure']
    airport_destination = vehicle['airport_destination']

    operations = vehicle['operations']
    performance = vehicle['performance']


    # [kg]
    max_zero_fuel_weight = aircraft['maximum_zero_fuel_weight']
    # [kg]
    operational_empty_weight = aircraft['operational_empty_weight']
    passenger_capacity_initial = aircraft['passenger_capacity']
    engines_number = aircraft['number_of_engines']
    max_engine_thrust = engine['maximum_thrust']
    
    reference_load_factor = operations['reference_load_factor']

    heading = 0

    # Operations and certification parameters:
    ceiling = operations['max_ceiling']  # [ft] UPDATE INPUT!!!!!!!!!
    descent_altitude = operations['descent_altitude']
    # Network and mission parameters
    holding_time = operations['holding_time']  # [min]
    fuel_density = operations['fuel_density']  # [kg/l]
    time_between_overhaul = operations['time_between_overhaul'] # [hr]
    taxi_fuel_flow_reference = operations['taxi_fuel_flow_reference']  # [kg/min]
    contingency_fuel_percent = operations['contingency_fuel_percent']
    min_cruise_time = operations['min_cruise_time']  # [min]
    go_around_allowance = operations['go_around_allowance']

    # Initial flight speed schedule
    climb_V_cas = operations['climb_V_cas']
    mach_climb = operations['mach_climb']
    cruise_V_cas = operations['cruise_V_cas']
    descent_V_cas = operations['descent_V_cas']
    mach_descent = operations['mach_descent']

    delta_ISA = operations['flight_planning_delta_ISA']

    # regulated_takeoff_mass = regulated_takeoff_weight(vehicle)
    # regulated_landing_mass = regulated_landing_weight(vehicle)

    max_takeoff_mass = landing_weight - go_around_allowance

    takeoff_allowance_mass = operations['takeoff_allowance']
    approach_allowance_mass = operations['approach_allowance_mass']
    average_taxi_in_time = operations['average_taxi_in_time']
    average_taxi_out_time = operations['average_taxi_out_time']

    payload = round(
        aircraft['passenger_capacity']
        * operations['passenger_mass']
        * reference_load_factor
    )


    initial_altitude = airport_departure['elevation']

    out = 0
    while out == 0:

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
        # Maximum altitude with minimum cruise time check
        g_climb = 4/1000
        g_descent = 3/1000
        K1 = g_climb + g_descent
        # Minimum distance at cruise stage
        Dmin = 10*operations['mach_cruise']*min_cruise_time

        K2 = (
            operations['alternative_airport_distance']
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

        # Initial climb fuel estimation
        initial_altitude = initial_altitude + 1500
        _, _, total_burned_fuel0, _ = climb_integration(
            max_takeoff_mass,
            mach_climb,
            climb_V_cas,
            delta_ISA,
            final_altitude,
            initial_altitude,
            vehicle
        )

        # Calculate best cruise mach
        mass_at_top_of_climb = max_takeoff_mass - total_burned_fuel0
        operations['mach_cruise'] = maximum_range_mach(
            mass_at_top_of_climb,
            final_altitude,
            delta_ISA,
            vehicle
        )
        mach_climb = operations['mach_cruise']
        mach_descent = operations['mach_cruise']

        # Recalculate climb with new mach
        final_distance, total_climb_time, total_burned_fuel, final_altitude = climb_integration(
            max_takeoff_mass,
            mach_climb,
            climb_V_cas,
            delta_ISA,
            final_altitude,
            initial_altitude,
            vehicle
        )

        delta = total_burned_fuel0 - total_burned_fuel

        if delta < tolerance:
            out = 1

    mass_at_top_of_climb = max_takeoff_mass - total_burned_fuel

    initial_cruise_altitude = final_altitude

    distance_climb = final_distance*feet_to_nautical_miles

    distance_cruise = operations['alternative_airport_distance']  - distance_climb

    altitude = initial_cruise_altitude

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

        # Type of descent: 1 = full calculation | 2 = no descent computed
        type_of_descent = 1

        if type_of_descent == 1:

            # Recalculate climb with new mach
            final_distance, total_descent_time, total_burned_fuel, final_altitude = descent_integration(
                final_cruise_mass,
                mach_descent,
                descent_V_cas,
                delta_ISA,
                descent_altitude,
                final_cruise_altitude,
                vehicle
            )
            distance_descent = final_distance*feet_to_nautical_miles
            distance_mission = distance_climb + distance_cruise + distance_descent
            distance_error = np.abs(operations['alternative_airport_distance'] -distance_mission)
            
            iteration = iteration + 1
            if distance_error <= 10:
                flag = 0
            else:
                if distance_mission > operations['alternative_airport_distance']:
                    distance_cruise = distance_cruise - distance_error*0.95
                else:
                    distance_cruise = distance_cruise + distance_error*0.95

    if iteration >= 200:
        raise ValueError

        if type_of_descent == 2:
            flag = 0
            total_burned_fuel = 0
            final_distance = 0
            total_decent_time = 0
            final_altitude = 0

    final_mission_mass = final_cruise_mass - total_burned_fuel
    total_mission_burned_fuel = max_takeoff_mass - final_mission_mass

    return total_mission_burned_fuel

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================


# DOC = mission(400)
# print(DOC)
