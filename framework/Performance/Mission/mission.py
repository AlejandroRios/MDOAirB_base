"""
MDOAirB

Description:
    - This module performs the mission analysis of the aircraft and computes the DOC.

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
from datetime import datetime
import math

from framework.Attributes.Airspeed.airspeed import (V_cas_to_mach,
                                                    crossover_altitude,
                                                    mach_to_V_cas)
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import \
    atmosphere_ISA_deviation
from framework.Attributes.Geo.bearing import calculate_bearing
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

def mission(vehicle, airport_departure, takeoff_runway, airport_destination, landing_runway, mission_range):
    """
    Description:
        - This function performs the mission analysis of the aircraft and computes the DOC.
 
    Inputs:
        - mission_range - [nm]
        - heading - [deg]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - fuel_mass - [kg]
        - complete_mission_flight_time - [min]
        - DOC - direct operational cost [US$]
        - mach - mach number
        - passenger_capacity - passenger capacity
        - SAR - cruise average specific air range [nm/kg]
    """
    
    start_time = datetime.now()
    # log.info('---- Start DOC mission function ----')

    performance = vehicle['performance']

    tolerance = 100

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    wing = vehicle['wing']

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

    # Heading                                                          
    bearing = calculate_bearing((airport_departure['latitude'],airport_departure['longitude']),(airport_destination['latitude'],airport_destination['longitude']))
    heading = round(bearing - (airport_departure['dmg'] + airport_destination['dmg'])/2)
    if heading < 0:
        heading = heading + 360

    # Operations and certification parameters:
    buffet_margin = operations['buffet_margin']  # [g]
    residual_rate_of_climb = performance['residual_rate_of_climb'] # [ft/min]
    ceiling = operations['max_ceiling']  # [ft] UPDATE INPUT!!!!!!!!!
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
    descent_V_cas = operations['descent_V_cas']
    mach_descent = operations['mach_descent']

    delta_ISA = operations['flight_planning_delta_ISA']
    captain_salary, first_officer_salary, flight_attendant_salary = crew_salary(aircraft['maximum_takeoff_weight'])
    
    regulated_takeoff_mass, takeoff_field_length_computed = regulated_takeoff_weight(vehicle, airport_departure, takeoff_runway)
    regulated_landing_mass, landing_field_length_computed = regulated_landing_weight(vehicle, airport_destination, landing_runway)

    max_takeoff_mass = regulated_takeoff_mass
    max_landing_mass = regulated_landing_mass

    # takeoff_allowance_mass = 200*max_takeoff_mass/22000
    # approach_allowance_mass = 100*max_takeoff_mass/22000
    takeoff_allowance_mass = operations['takeoff_allowance'] 
    approach_allowance_mass =operations['approach_allowance_mass']
    average_taxi_in_time = operations['average_taxi_in_time']
    average_taxi_out_time = operations['average_taxi_out_time']

    payload = round(
        aircraft['passenger_capacity']
        * operations['passenger_mass']
    )
    # payload = round(
    #     aircraft['passenger_capacity']
    #     * operations['passenger_mass']
    #     * reference_load_factor
    # )

    

    f = 0
    while f == 0:
        step = 500
        out = 0
        initial_altitude = airport_departure['elevation']

        distance_vec = []
        altitude_vec = []
        mass_vec = []
        time_vec = []
        sfc_vec = []
        thrust_vec = []
        mach_vec = []
        CL_vec = []
        CD_vec = []
        LoD_vec = []
        throttle_vec = []
        vcas_vec = []

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
                mission_range
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
            _, _, total_burned_fuel0, _, _, _, _, _, _, _, _, _, _, _, _, _ = climb_integration(
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
            final_distance, total_climb_time, total_burned_fuel, final_altitude, distancev, altitudev, massv, timev, sfcv, thrustv, machv, CLv, CDv, LoDv, throttlev, vcasv = climb_integration(
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
        
        distance_vec = np.append(distance_vec, distancev)
        altitude_vec = np.append(altitude_vec, altitudev)
        mass_vec = np.append(mass_vec, massv)
        time_vec = np.append(time_vec, timev)
        sfc_vec = np.append(sfc_vec, sfcv)
        thrust_vec = np.append(thrust_vec, thrustv)
        mach_vec = np.append(mach_vec, machv)
        CL_vec = np.append(CL_vec, CLv)
        CD_vec = np.append(CD_vec, CDv)
        LoD_vec = np.append(LoD_vec, LoDv)
        throttle_vec = np.append(throttle_vec, throttlev)
        vcas_vec = np.append(vcas_vec, vcasv)
        
        mass_at_top_of_climb = max_takeoff_mass - total_burned_fuel

        initial_cruise_altitude = final_altitude

        distance_climb = final_distance*feet_to_nautical_miles

        distance_cruise = mission_range  - distance_climb

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

            # Type of descent: 1 = full calculation | 2 = no descent computed
            type_of_descent = 1

            if type_of_descent == 1:

                # Recalculate climb with new mach
                final_distance, total_descent_time, total_burned_fuel, final_altitude, distancev, altitudev, massv, timev, sfcv, thrustv, machv, CLv, CDv, LoDv, throttlev, vcasv = descent_integration(
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
                distance_error = np.abs(mission_range -distance_mission)

                iteration = iteration + 1

                if distance_error <= 5:
                    flag = 0
                else:
                    if distance_mission > mission_range:
                        distance_cruise = distance_cruise - distance_error*0.95
                    else:
                        distance_cruise = distance_cruise + distance_error*0.95
                        
        distance_vec = np.append(distance_vec, final_distance + distancev)
        altitude_vec = np.append(altitude_vec, altitudev)
        mass_vec = np.append(mass_vec, massv)
        time_vec = np.append(time_vec, total_climb_time +
                             total_cruise_time + timev)
        sfc_vec = np.append(sfc_vec, sfcv)
        thrust_vec = np.append(thrust_vec, thrustv)
        mach_vec = np.append(mach_vec, machv)
        CL_vec = np.append(CL_vec, CLv)
        CD_vec = np.append(CD_vec, CDv)
        LoD_vec = np.append(LoD_vec, LoDv)
        throttle_vec = np.append(throttle_vec, throttlev)
        vcas_vec = np.append(vcas_vec, vcasv)
        
        if iteration >= 200:
            raise ValueError 


        if type_of_descent == 2:
            flag = 0
            total_burned_fuel = 0
            final_distance = 0
            total_decent_time = 0
            total_burned_fuel = 0
            final_altitude = 0

        final_mission_mass = final_cruise_mass - total_burned_fuel
        total_mission_burned_fuel = max_takeoff_mass - final_mission_mass
        total_mission_flight_time = total_climb_time + \
            total_cruise_time + total_descent_time + operations['landing_time_allowance'] + operations['takeoff_time_allowance']
        total_mission_distance = distance_mission

        # Reserve fuel
        reserve_fuel_calculation_type = 0  # 0 if simplified computation | 1 if full computation
        contingency_fuel = contingency_fuel_percent*total_mission_burned_fuel

        landing_weight = max_takeoff_mass - total_mission_burned_fuel

        if reserve_fuel_calculation_type == 0:
            reserve_fuel_calculated = reserve_fuel(landing_weight, operations['alternative_airport_distance'], holding_time, delta_ISA,)
            final_reserve_fuel = reserve_fuel_calculated + contingency_fuel
        else:
            fuel_mass_alterative_airport = mission_alternative(vehicle, airport_departure, airport_destination,landing_weight)
            fuel_mass_holding = holding_fuel(altitude, delta_ISA, holding_time, vehicle)
            final_reserve_fuel =fuel_mass_alterative_airport + fuel_mass_holding + contingency_fuel
            

        # Rule of three to estimate fuel flow during taxi
        taxi_fuel_flow = taxi_fuel_flow_reference*max_takeoff_mass/22000
        taxi_in_fuel = average_taxi_in_time*taxi_fuel_flow
        takeoff_fuel = total_mission_burned_fuel + final_reserve_fuel + takeoff_allowance_mass + taxi_in_fuel
        taxi_out_fuel = average_taxi_out_time*taxi_fuel_flow
        total_fuel_on_board = takeoff_fuel + taxi_out_fuel
        remaining_fuel = takeoff_fuel - total_mission_burned_fuel - \
            approach_allowance_mass - taxi_in_fuel

        # Payload range envelope check

        MTOW_ZFW = max_zero_fuel_weight + total_fuel_on_board
        MTOW_LW = max_landing_mass + total_mission_burned_fuel

        delta_1 = max_takeoff_mass - MTOW_ZFW
        delta_2 = total_fuel_on_board - wing['fuel_capacity']
        delta_3 = max_takeoff_mass - MTOW_LW

        extra = (max_takeoff_mass - operational_empty_weight -
                 payload) - takeoff_fuel
        delta = max([delta_1, delta_2, delta_3, extra])

        if delta > tolerance:
            max_takeoff_mass = max_takeoff_mass-delta
        else:
            # Payload reduction if restricted
            max_takeoff_mass = min(
                [max_takeoff_mass, MTOW_ZFW, MTOW_LW])
            payload_calculation = max_takeoff_mass - \
                takeoff_fuel - operational_empty_weight
            if payload_calculation > payload:
                payload = payload
            else:
                payload = payload_calculation

            f = 1

        passenger_capacity = np.round(payload/operations['passenger_mass'])
        load_factor = passenger_capacity/passenger_capacity_initial*100

    
    # DOC calculation
    fuel_mass = total_mission_burned_fuel + \
        (average_taxi_out_time + average_taxi_in_time)*taxi_fuel_flow
        
    DOC = direct_operational_cost(
        time_between_overhaul,
        total_mission_flight_time,
        fuel_mass,
        operational_empty_weight,
        total_mission_distance,
        max_engine_thrust,
        engines_number,
        0.35*operational_empty_weight,
        regulated_takeoff_mass,
        vehicle)
    
    # Cruise average specific air range
    SAR = fuel_mass/mission_range

    complete_mission_flight_time = total_mission_flight_time + operations['average_departure_delay'] + operations['average_arrival_delay'] + operations['turn_around_time'] 

    _, _, _, _, _, rho_ISA, _, _ = atmosphere_ISA_deviation(0, delta_ISA)
    # Approach speed
    app_speed = 1.23*math.sqrt(2*aircraft['maximum_landing_weight']*GRAVITY/(wing['area']*rho_ISA*aircraft['CL_maximum_landing']))

    # log.info('---- End DOC mission function ----')
    # end_time = datetime.now()
    # log.info('DOC mission execution time: {}'.format(end_time - start_time))

    return float(fuel_mass), float(complete_mission_flight_time), float(DOC), float(mach), float(passenger_capacity), float(SAR), float(landing_field_length_computed), float(takeoff_field_length_computed), float(app_speed), distance_vec, altitude_vec, mass_vec, time_vec, sfc_vec, thrust_vec, mach_vec, CL_vec, CD_vec, LoD_vec, throttle_vec, vcas_vec


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
# vehicle = initialize_aircraft_parameters()

# performance = vehicle['performance']

# tolerance = 100
# x =  [130, 90, 30, 22, -2, 35, 50, 12, 22, 1450, 25, 80, 4,1500, 41000, 78, 1, 1, 1, 1]
# aircraft = vehicle['aircraft']
# engine = vehicle['engine']
# wing = vehicle['wing']
# fuselage = vehicle['fuselage']
# horizontal_tail = vehicle['horizontal_tail']

# airport_departure = vehicle['airport_departure']
# airport_destination = vehicle['airport_destination']

# operations = vehicle['operations']
# performance = vehicle['performance']

# mission_range = 300
# aircraft['maximum_zero_fuel_weight'] = 31000
# aircraft['operational_empty_weight'] = 30000
# # aircraft['passenger_capacity'] = 77
# aircraft['number_of_engines'] = 2
# # engine['maximum_thrust'] = 63173
# operations['reference_load_factor'] = 0.85
# aircraft['maximum_takeoff_weight'] = 38790
# aircraft['maximum_landing_weight'] = 34100

# # Upload dictionary variables with optimization variables input vector x
# wing['area'] = x[0]
# wing['aspect_ratio'] = x[1]/10
# wing['taper_ratio'] = x[2]/100
# wing['sweep_c_4'] = x[3]
# wing['twist'] = x[4]
# wing['semi_span_kink'] = x[5]/100
# aircraft['passenger_capacity'] = x[11]
# fuselage['seat_abreast_number'] = x[12]
# performance['range'] = x[13]
# aircraft['winglet_presence'] = x[17]
# # aircraft['winglet_presence'] = 1
# aircraft['slat_presence'] = x[18]
# # aircraft['slat_presence'] = 1
# horizontal_tail['position'] = x[19]
# # horizontal_tail['position'] = 1

# engine['bypass'] = x[6]/10
# engine['fan_diameter'] = x[7]/10
# engine['compressor_pressure_ratio'] = x[8]
# engine['turbine_inlet_temperature'] = x[9]
# engine['fan_pressure_ratio'] = x[10]/10
# engine['design_point_pressure'] = x[14]
# engine['design_point_mach'] = x[15]/100
# engine['position'] = x[16]

# vehicle = np.load('Database/Aircrafts/baseline_EMB.npy',allow_pickle = True)
# vehicle = vehicle.item()
# operations = vehicle['operations']

# operations['flight_planning_delta_ISA'] = 0
# heading = 180
# mission_range = 697.40
# print(mission(mission_range,heading,vehicle))


