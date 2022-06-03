"""
MDOAirB

Description:
    - This module calculates the network profit following the following steps:
        - Vehicle sizing and checks (airplane_sizing)
        - Revenue calculation (reveneu)
        - Direct operational cost calculation (mission)
        - Profit calculation (network_optimization)

TODO's:

| Authors: Alejandro Rios
           Lionel Guerin
           
  
| Email: aarc.88@gmail.com
| Creation: January 2021
| Last modification: July 2021
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================
import copy
import csv
import getopt
import json
import os
import pickle
import sys
from datetime import datetime
from random import randrange

import haversine
import jsonschema
import numpy as np
import pandas as pd
from framework.Database.Aircrafts.baseline_aircraft_parameters import \
    initialize_aircraft_parameters
from framework.Database.Airports.airports_database import AIRPORTS_DATABASE
from framework.Economics.revenue import revenue
from framework.Network.family_network_optimization import family_network_optimization
from framework.Performance.Mission.mission import mission
from framework.Sizing.airplane_sizing_check import airplane_sizing
from framework.utilities.logger import get_logger
from framework.utilities.output import (write_bad_results, write_kml_results,
                                        write_newtork_results,
                                        write_optimal_results,
                                        write_unfeasible_results)
from haversine import Unit, haversine
from jsonschema import validate

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])


def objective_function0(x, original_vehicle, computation_mode, route_computation_mode, airports, distances, demands):

    log.info('==== Start network profit module ====')
    start_time = datetime.now()

    # Do a copy of original vehicle
    vehicle = copy.deepcopy(original_vehicle)
    # with open('Database/DOC/Vehicle.pkl', 'rb') as f:
    #     vehicle = pickle.load(f)

    # Try running profit calculation. If error appears during run profit = 0
    try:
        # =============================================================================
        # Airplane sizing and checks
        # with open('Database/Family/40_to_100/airports/'+str(x[0])+'.pkl', 'rb') as f:
        #     airports = pickle.load(f)

        # with open('Database/Family/40_to_100/distances/'+str(x[0])+'.pkl', 'rb') as f:
        #     distances = pickle.load(f)

        # with open('Database/Family/40_to_100/demands/'+str(x[0])+'.pkl', 'rb') as f:
        #     demands = pickle.load(f)

        # with open('Database/Family/40_to_100/DOC/'+str(x[0])+'.pkl', 'rb') as f:
        #     DOC_ik = pickle.load(f)

        # with open('Database/Family/40_to_100/vehicle/'+str(x[0])+'.pkl', 'rb') as f:
        #     vehicle = pickle.load(f)

        with open('Database/Family/40_to_100/all_dictionaries/'+str(x[0])+'.pkl', 'rb') as f:
            all_info_acft1 = pickle.load(f)

        with open('Database/Family/101_to_160/all_dictionaries/'+str(x[1])+'.pkl', 'rb') as f:
            all_info_acft2 = pickle.load(f)

        with open('Database/Family/161_to_220/all_dictionaries/'+str(x[2])+'.pkl', 'rb') as f:
            all_info_acft3 = pickle.load(f)

        print(all_info_acft3['vehicle']['aircraft']['passenger_capacity'])
        print(all_info_acft3['vehicle']['fuselage']['seat_abreast_number'])
        print(all_info_acft3['vehicle']['wing']['area'])
        print(all_info_acft3['vehicle']['wing']['aspect_ratio'])
        print(all_info_acft3['vehicle']['wing']['semi_span_kink'])
        print(all_info_acft3['vehicle']['wing']['sweep_c_4'])

        print(all_info_acft3['vehicle']['wing']['taper_ratio'])
        print(all_info_acft3['vehicle']['wing']['twist'])

        print(all_info_acft3['vehicle']['engine']['type'])
        print(all_info_acft3['vehicle']['engine']['bypass'])
        print(all_info_acft3['vehicle']['engine']['fan_diameter'])
        print(all_info_acft3['vehicle']['engine']['maximum_thrust'])
        print(all_info_acft3['vehicle']['engine']['turbine_inlet_temperature'])
        print(all_info_acft3['vehicle']['performance']['range'])

        airports = all_info_acft1['airports']
        distances = all_info_acft1['distances']
        demands = all_info_acft1['demands']
        DOC_ik = all_info_acft1['DOC_ik']
        DOC_nd = all_info_acft1['DOC_nd']
        fuel_mass = all_info_acft1['fuel_mass']
        total_mission_flight_time = all_info_acft1['total_mission_flight_time']
        mach = all_info_acft1['mach']
        passenger_capacity = all_info_acft1['passenger_capacity']
        SAR = all_info_acft1['SAR']
        vehicle = all_info_acft1['vehicle']

        def load_info_from_dicts(dictionary):
            airports = dictionary['airports']
            distances = dictionary['distances']
            demands = dictionary['demands']
            DOC_ik = dictionary['DOC_ik']
            DOC_nd = dictionary['DOC_nd']
            fuel_mass = dictionary['fuel_mass']
            total_mission_flight_time = dictionary['total_mission_flight_time']
            mach = dictionary['mach']
            passenger_capacity = dictionary['passenger_capacity']
            SAR = dictionary['SAR']
            vehicle = dictionary['vehicle']
            return airports, distances, demands, DOC_ik, DOC_nd, fuel_mass, total_mission_flight_time, mach, passenger_capacity, SAR, vehicle

        airports_1, distances_1, demands_1, DOC_ik_1, DOC_nd_1, fuel_mass_1, total_mission_flight_time_1, mach_1, passenger_capacity_1, SAR_1, vehicle_1 = load_info_from_dicts(
            all_info_acft1)
        airports_2, distances_2, demands_2, DOC_ik_2, DOC_nd_2, fuel_mass_2, total_mission_flight_time_2, mach_2, passenger_capacity_2, SAR_2, vehicle_2 = load_info_from_dicts(
            all_info_acft2)
        airports_3, distances_3, demands_3, DOC_ik_3, DOC_nd_3, fuel_mass_3, total_mission_flight_time_3, mach_3, passenger_capacity_3, SAR_3, vehicle_3 = load_info_from_dicts(
            all_info_acft3)


        print(passenger_capacity_1)

        # SAR_all = float(all_info_acft1['SAR']) + float(all_info_acft2['SAR']) + float(all_info_acft3['SAR'])

        status = 0
        results_1 = vehicle_1['results']
        performance_1 = vehicle_1['performance']
        operations_1 = vehicle_1['operations']
        aircraft_1 = vehicle_1['aircraft']

        results_2 = vehicle_2['results']
        performance_2 = vehicle_2['performance']
        operations_2 = vehicle_2['operations']
        aircraft_2 = vehicle_2['aircraft']

        results_3 = vehicle_3['results']
        performance_3 = vehicle_3['performance']
        operations_3 = vehicle_3['operations']
        aircraft_3 = vehicle_3['aircraft']

        # =============================================================================
        # If airplane pass checks, status = 0, else status = 1 and profit = 0
        if status == 0:
            log.info('Aircraft passed sizing and checks status: {}'.format(status))
            market_share = operations_1['market_share']

            results_1['nodes_number'] = len(airports)
            results_2['nodes_number'] = len(airports)
            results_3['nodes_number'] = len(airports)

            pax_capacity_1 = aircraft_1['passenger_capacity']  # Passenger capacity
            pax_capacity_1 = aircraft_2['passenger_capacity']  # Passenger capacity
            pax_capacity_1 = aircraft_2['passenger_capacity']  # Passenger capacity
            #=============================================================================
            log.info('---- Start DOC matrix calculation ----')
            # The DOC is estimated for each city pair and stored in the DOC dictionary
            city_matrix_size = len(airports)*len(airports)
            airports_keys = list(airports.keys())

            # log.info('Aircraft DOC matrix: {}'.format(DOC_ik))
            # =============================================================================
            log.info('---- Start Network Optimization ----')
            # Network optimization that maximizes the network profit
            try:
                profit, vehicle01, vehicle02, vehicle03, kpi_df1_1, kpi_df2_1, kpi_df1_2, kpi_df2_2, kpi_df1_3, kpi_df2_3, airplanes_ik = family_network_optimization(
                    computation_mode, list(airports.keys()), all_info_acft1, all_info_acft2, all_info_acft3)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< network_optimization", exc_info=True)
                profit = 0

            log.info('Network profit [$USD]: {}'.format(profit))
            # =============================================================================

            def flatten_dict(dd, separator='_', prefix=''):
                return {prefix + separator + k if prefix else k: v
                        for kk, vv in dd.items()
                        for k, v in flatten_dict(vv, separator, kk).items()
                        } if isinstance(dd, dict) else {prefix: dd}

            try:

                def flatten_update(vehicle, mach, passenger_capacity, fuel_mass, total_mission_flight_time, DOC_nd, SAR, distances,kpi_df2):

                    results = vehicle['results']
                    aircraft = vehicle['aircraft']

                    mach_flatt = flatten_dict(mach)
                    mach_df = pd.DataFrame.from_dict(
                        mach_flatt, orient="index", columns=['mach'])
                    passenger_capacity_flatt = flatten_dict(passenger_capacity)
                    passenger_capacity_df = pd.DataFrame.from_dict(
                        passenger_capacity_flatt, orient="index", columns=['pax_num'])
                    fuel_used_flatt = flatten_dict(fuel_mass)
                    fuel_used_df = pd.DataFrame.from_dict(
                        fuel_used_flatt, orient="index", columns=['fuel'])
                    mission_time_flatt = flatten_dict(
                        total_mission_flight_time)
                    mission_time_df = pd.DataFrame.from_dict(
                        mission_time_flatt, orient="index", columns=['time'])
                    DOC_nd_flatt = flatten_dict(DOC_nd)
                    DOC_nd_df = pd.DataFrame.from_dict(
                        DOC_nd_flatt, orient="index", columns=['DOC_nd'])
                    SAR_flatt = flatten_dict(SAR)
                    SAR_df = pd.DataFrame.from_dict(
                        SAR_flatt, orient="index", columns=['SAR'])
                    distances_flatt = flatten_dict(distances)
                    kpi_df3 = pd.DataFrame.from_dict(
                        distances_flatt, orient="index", columns=['distances01'])

                    kpi_df3['mach'] = mach_df['mach'].values
                    kpi_df3['pax_num'] = passenger_capacity_df['pax_num'].values
                    kpi_df3['fuel'] = fuel_used_df['fuel'].values
                    kpi_df3['time'] = mission_time_df['time'].values
                    kpi_df3['DOC_nd'] = DOC_nd_df['DOC_nd'].values
                    kpi_df3['SAR'] = SAR_df['SAR'].values
                    kpi_df3 = kpi_df3.drop(
                        kpi_df3[kpi_df3.distances01 == 0].index)
                    kpi_df3 = kpi_df3.reset_index(drop=True)
                    kpi_df2 = pd.concat([kpi_df2, kpi_df3], axis=1)
                    # Number of active nodes
                    kpi_df2['active_arcs'] = np.where(
                        kpi_df2["aircraft_number"] > 0, 1, 0)
                    results['arcs_number'] = kpi_df2['active_arcs'].sum()
                    # Number of aircraft
                    kpi_df2['aircraft_number'] = kpi_df2['aircraft_number'].fillna(
                        0)
                    # Average cruise mach
                    kpi_df2['mach_tot_aircraft'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['mach']
                    # Total fuel
                    kpi_df2['total_fuel'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['fuel']
                    # total CEMV
                    kpi_df2['total_CEMV'] = kpi_df2['aircraft_number'] * \
                        ((1/kpi_df2['SAR']) *
                         (1/(aircraft['wetted_area']**0.24)))
                    # Total distance
                    kpi_df2['total_distance'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['distances']
                    print('=======================================================')
                    print('total dist',kpi_df2['total_distance'].sum() )
                    # Total pax
                    kpi_df2['total_pax'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['pax_num']
                    print('totalpax',kpi_df2['total_pax'].sum() )
                    # Total cost
                    kpi_df2['total_cost'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['doc']
                    print('total cost',kpi_df2['total_cost'].sum() )
                    print('aircraft num',kpi_df2['aircraft_number'].sum() )
                    results['network_density'] = results['arcs_number'] / \
                        (results['nodes_number'] *
                         results['nodes_number']-results['nodes_number'])
                    kpi_df2['total_time'] = kpi_df2['aircraft_number'] * \
                        kpi_df2['time']

                    return kpi_df2, kpi_df3

                kpi_df2_1, kpi_df3_1 = flatten_update(vehicle_1,
                    mach_1, passenger_capacity_1, fuel_mass_1, total_mission_flight_time_1, DOC_nd_1, SAR_1, distances_1,kpi_df2_1)
                kpi_df2_2, kpi_df3_2 = flatten_update(vehicle_2,
                    mach_2, passenger_capacity_2, fuel_mass_2, total_mission_flight_time_2, DOC_nd_2, SAR_2, distances_2, kpi_df2_2)
                kpi_df2_3, kpi_df3_3 = flatten_update(vehicle_3,
                    mach_3, passenger_capacity_3, fuel_mass_3, total_mission_flight_time_3, DOC_nd_3, SAR_3, distances_3, kpi_df2_3)

                total_fuel = kpi_df2_1['total_fuel'].sum() + kpi_df2_2['total_fuel'].sum() + kpi_df2_3['total_fuel'].sum()
                total_CO2 = total_fuel*3.15
                total_distance = kpi_df2_1['total_distance'].sum() + kpi_df2_2['total_distance'].sum() +kpi_df2_3['total_distance'].sum()
                total_pax = results_1['covered_demand']
                CO2_efficiency =  total_CO2 / \
                    (total_pax*total_distance*1.852)

                print(total_pax)
                print(CO2_efficiency)

            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< writting dataframes", exc_info=True)

            try:
                write_optimal_results(x, list(airports.keys(
                )), distances, demands, profit, DOC_ik, vehicle, kpi_df2_1, airplanes_ik)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< write_optimal_results", exc_info=True)

            try:
                write_kml_results(airports, profit, airplanes_ik)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< write_kml_results", exc_info=True)

            try:
                write_newtork_results(profit, kpi_df1_1, kpi_df2_1)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< write_newtork_results", exc_info=True)

        else:
            profit = 0
            write_unfeasible_results(flags, x)
            log.info(
                'Aircraft did not pass sizing and checks, profit: {}'.format(profit))

    except:

        log.error(
            ">>>>>>>>>> Error at <<<<<<<<<<<< objective_function", exc_info=True)
        error = sys.exc_info()[0]
        profit = 0
        log.info('Exception ocurred during calculations')
        log.info('Aircraft not passed sizing and checks, profit: {}'.format(profit))

        try:
            write_bad_results(error, x)
        except:
            log.error(
                ">>>>>>>>>> Error at <<<<<<<<<<<< write_bad_results", exc_info=True)

    else:
        print("Final individual results is:", profit)
    finally:
        print("Executing finally clause")
    end_time = datetime.now()
    log.info('Network profit excecution time: {}'.format(end_time - start_time))
    log.info('==== End network profit module ====')

    return profit, CO2_efficiency

# def objective_function(x, original_vehicle, computation_mode, route_computation_mode, airports, distances, demands):
# 	print("--------------------------------------------------------------------")
# 	print(x)
# 	print("--------------------------------------------------------------------")
# 	print(original_vehicle)
# 	print("--------------------------------------------------------------------")
# 	print(computation_mode)
# 	print("--------------------------------------------------------------------")
# 	print(route_computation_mode)
# 	print("--------------------------------------------------------------------")
# 	print(airports)
# 	print("--------------------------------------------------------------------")
# 	print(distances)
# 	print("--------------------------------------------------------------------")
# 	print(demands)
# 	print("--------------------------------------------------------------------")
# 	return randrange(0,10000),

# =============================================================================
# TEST
# =============================================================================
# global NN_induced, NN_wave, NN_cd0, NN_CL, num_Alejandro
# num_Alejandro = 100000000000000000000000
# global NN_induced, NN_wave, NN_cd0, NN_CL


# from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters

# # # # # x = [130, 8.204561481970153, 0.3229876327660606, 31, -4, 0.3896951781733875, 4.826332970409506, 1.0650795018081771, 27, 1485, 1.6, 101, 4, 2185, 41000, 0.78, 1, 1, 1, 1]
# # # # # # x = [73, 8.210260198894748, 0.34131954092766925, 28, -5, 0.32042307969643524, 5.000456116634125, 1.337333818504011, 27, 1442, 1.6, 106, 6, 1979, 41000, 0.78, 1, 1, 1, 1]
# # # # # # x = [106, 9.208279852593964, 0.4714790814543369, 16, -3, 0.34987438995033143, 6.420120321538892, 1.7349297171205607, 29, 1461, 1.6, 74, 6, 1079, 41000, 0.78, 1, 1, 1, 1]

# # # # print(result)
# # # # x =[0, 77, 35, 19, -3, 33, 63, 17, 29, 1396, 25, 120, 6, 2280, 41000, 78]

# # # # result = objective_function(x, vehicle)

# # # # x = [103, 81, 40, 16, -4, 34, 59, 14, 29, 1370, 18, 114, 6, 1118]


# # # # x = [9.700e+01,9.900e+01,4.400e+01,1.800e+01,-2.000e+00,3.200e+01, 4.800e+01,1.400e+01,3.000e+01,1.462e+03,1.700e+01,6.000e+01, 6.000e+00,1.525e+03]
# # # # # x = [7.300e+01,8.600e+01,2.900e+01,1.600e+01,-5.000e+00,3.400e+01, 4.600e+01,2.000e+01,2.700e+01,1.372e+03,1.800e+01,1.160e+02, 4.000e+00,2.425e+03]
# # # # # x = [1.210e+02,9.600e+01,4.100e+01,2.600e+01,-3.000e+00,3.600e+01, 6.200e+01,1.800e+01,2.900e+01,1.478e+03,1.800e+01,6.800e+01, 5.000e+00,1.975e+03]
# # # # # x = [7.900e+01,9.400e+01,3.100e+01,2.000e+01,-4.000e+00,3.700e+01, 5.600e+01,1.000e+01,2.900e+01,1.448e+03,1.600e+01,8.200e+01, 5.000e+00,1.825e+03]
# # # # # x = [1.270e+02,7.600e+01,3.600e+01,2.800e+01,-4.000e+00,3.800e+01, 6.000e+01,1.800e+01,3.000e+01,1.432e+03,1.700e+01,8.800e+01, 5.000e+00,1.225e+03]
# # # # # x = [1.150e+02,8.400e+01,4.900e+01,3.200e+01,-2.000e+00,3.600e+01, 5.000e+01,1.400e+01,2.800e+01,1.492e+03,1.900e+01,1.100e+02, 4.000e+00,1.375e+03]
# # # # # x = [1.090e+02,8.100e+01,2.600e+01,2.400e+01,-5.000e+00,4.000e+01, 5.200e+01,1.600e+01,2.700e+01,1.402e+03,1.400e+01,7.400e+01, 4.000e+00,2.125e+03]
# # # # # x = [9.100e+01,8.900e+01,3.400e+01,3.000e+01,-3.000e+00,3.900e+01, 6.400e+01,1.200e+01,2.800e+01,1.358e+03,2.000e+01,9.600e+01, 5.000e+00,1.675e+03]
# # # # # x = [8.500e+01,9.100e+01,3.900e+01,3.400e+01,-3.000e+00,3.300e+01, 5.800e+01,1.200e+01,2.800e+01,1.418e+03,1.600e+01,1.020e+02, 6.000e+00,2.275e+03]
# # # # # x = [1.030e+02,7.900e+01,4.600e+01,2.200e+01,-4.000e+00,3.500e+01, 5.400e+01,1.600e+01,2.900e+01,1.388e+03,1.500e+01,5.400e+01, 6.000e+00,1.075e+03]

# # # # x = [1.150e+02,8.400e+01,4.900e+01,3.200e+01,-2.000e+00,3.600e+01, 5.000e+01,1.400e+01,2.800e+01,1.492e+03,1.900e+01,1.100e+02, 4.000e+00,1.375e+03,41000, 78, 1, 1, 1, 1] # Prifit ok
# # # # x =  [127, 82, 46, 22, -2, 44, 48, 21, 27, 1358, 22,  92, 5, 2875, 41200, 82, 1, 1, 1, 1]
# # # # x =  [115, 84, 49, 32, -2, 36, 50, 14, 28, 1492, 19, 110, 4, 1375, 41000, 78, 1, 1, 1, 1] #good one
# # # x =  [72, 86, 28, 26, -5, 34, 50, 13, 28, 1450, 14, 70, 4, 1600, 41000, 78, 1, 1, 1, 1] # Baseline

# x =  [130, 91, 38, 29, -4.5, 33, 62, 17, 30, 1480, 18, 144, 6, 1900, 41000, 78, 1, 1, 1, 1] # 144 seat
# # # # x =  [121, 80, 40, 18, -2, 40, 52, 13, 28, 1358, 15, 108, 4, 1875, 41000, 82, 1, 1, 1, 1] # Baseline2
# # # # # # # x = [int(x) for x in x]
# # # # # # # print(x)

# # x =  [121, 114, 27, 25, -4.0, 35, 50, 14, 29, 1430, 23, 142, 6, 1171, 41000, 78, 1, 1, 1, 1] # Optim_Jose

# # # # # # x = [76, 118, 46, 23, -3, 33, 55, 19, 30, 1357, 18, 86, 6, 2412, 42260, 79, 1, 1, 1, 1]
# # # # # # x = [91, 108, 50, 29, -3, 34, 52, 12, 27, 1366, 19, 204, 4, 1812, 39260, 80, 1, 1, 1, 1]
# # # x = [110, 82, 34, 25, -5, 38, 52, 11, 30, 1462, 19, 92, 4, 1375, 39600, 80, 1, 1, 1, 1]
# # # # # vehicle = initialize_aircraft_parameters()

# from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters

# x =[98,78,31,16,-4,40,61,20,28,1418,17,98,4,2005,41000,78,1,1,1,1]
# vehicle = initialize_aircraft_parameters()

# result = objective_function(x, vehicle, [])

# print(result)

CUSTOM_INPUTS_SCHEMA = 'Database/JsonSchema/Custom_Inputs.schema.json'


class CustomInputsError(Exception):
    def __init__(self, message):
        self.message = f"Custom inputs issue: {message}"
        super().__init__(self.message)


def check_runways(demands, airports):
    for departure in demands:
        for arrival in demands[departure]:
            takeoff_runway = demands[departure][arrival]["takeoff_runway"]
            if takeoff_runway not in airports[departure]["runways"]:
                raise CustomInputsError(
                    f'Take-of runway {takeoff_runway} do not exit for airport {departure} in MDO database')
            landing_runway = demands[departure][arrival]["landing_runway"]
            if landing_runway not in airports[arrival]["runways"]:
                raise CustomInputsError(
                    f'Landing runway {landing_runway} do not exit for airport {arrival} in MDO database')


def check_airports(airport_keys):
    try:
        airports = {k: AIRPORTS_DATABASE[k] for k in airport_keys}
    except KeyError as key_error:
        raise CustomInputsError(
            f'Airports {key_error.args} do not exit in MDO database')

    return airports


def haversine_distance(coordinates_departure, coordinates_arrival):
    # Perform haversine distance calculation in nautical miles
    distance = float(haversine(coordinates_departure,
                     coordinates_arrival, unit='nmi'))
    return distance


def check_demands(data, fixed_parameters):
    airport_keys = list(data.keys())
    for key in data:
        if (key in data[key]):
            raise CustomInputsError(
                f'Airport {key} exist on both departure and arrival for the same demand')
        airport_keys = airport_keys + \
            list(set(data[key].keys()) - set(airport_keys))

    airports = check_airports(airport_keys)

    check_runways(data, airports)

    market_share = fixed_parameters['operations']['market_share']

    distances = {}
    for departure in airport_keys:
        distances[departure] = {}
        if (departure not in data):
            data[departure] = {}
        for arrival in airport_keys:
            if (arrival not in data[departure]):
                data[departure][arrival] = {}
                data[departure][arrival]['demand'] = 0
                distances[departure][arrival] = 0
            else:
                data[departure][arrival]['demand'] = np.round(
                    market_share * data[departure][arrival]['demand'])

                coordinates_departure = (
                    airports[departure]["latitude"], airports[departure]["longitude"])
                coordinates_arrival = (
                    airports[arrival]["latitude"], airports[arrival]["longitude"])
                distances[departure][arrival] = round(
                    haversine_distance(coordinates_departure, coordinates_arrival))

    return airports, distances, data


def check_design_variables(data):
    for key in data:
        if (data[key]["lower_band"] > data[key]["upper_band"]):
            raise CustomInputsError(
                f'Lower band {data[key]["lower_band"]} is greater than upper band {data[key]["upper_band"]} for {key}')


def read_custom_inputs(schema_path, file_path):
    computation_mode = 0
    route_computation_mode = 0
    design_variables = {}
    custom_fixed_parameters = {}
    fixed_aircraft = {}

    try:
        with open(schema_path) as f:
            schema = json.load(f)

        with open(file_path) as f:
            data = json.load(f)

        validate(instance=data, schema=schema)

        if ("computation_mode" in data):
            computation_mode = data["computation_mode"]

        if ("route_computation_mode" in data):
            route_computation_mode = data["route_computation_mode"]

        if ("demands" not in data):
            raise CustomInputsError(f"Demands is mandatory in custom inputs")

        if ("design_variables" in data):
            design_variables = data["design_variables"]
            check_design_variables(design_variables)

        if ("fixed_parameters" in data):
            custom_fixed_parameters = data["fixed_parameters"]

        # Update vehicle with fixed parameters
        fixed_parameters = initialize_aircraft_parameters()
        fixed_parameters = update_vehicle(
            fixed_parameters, custom_fixed_parameters)

        if ("fixed_aircraft" in data):
            fixed_aircraft = data["fixed_aircraft"]

        airports, distances, demands = check_demands(
            data["demands"], fixed_parameters)

    except OSError as os_error:
        raise CustomInputsError(f"{os_error.strerror} [{os_error.filename}]")
    except json.JSONDecodeError as dec_error:
        raise CustomInputsError(
            f"{dec_error.msg} [line {dec_error.lineno} column {dec_error.colno} (char {dec_error.pos})]")
    except jsonschema.exceptions.SchemaError:
        raise CustomInputsError(f"There is an error with the schema")
    except jsonschema.exceptions.ValidationError as val_error:
        path_error = ""
        for path in val_error.path:
            if (path_error):
                path_error += "."
            path_error += path
        raise CustomInputsError(f"{val_error.message} in path [{path_error}]")

    return computation_mode, route_computation_mode, airports, distances, demands, design_variables, fixed_parameters, fixed_aircraft


def update_vehicle(vehicle, fixed_parameters):
    for key in fixed_parameters:
        if (key in vehicle):
            vehicle[key].update(fixed_parameters[key])
    return vehicle


def usage():
    print("This is the usage function")
    print(f"Usage: {sys.argv[0]} -f <custom inputs file>")


def readArgv(argv):
    customInputsfile = ""
    try:
        opts, _ = getopt.getopt(argv, "hf:", ["help", "file="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-f", "--file"):
            customInputsfile = arg
    return customInputsfile


def objective_function(vehicle, x=None):

    argv = ['--file', 'Database/JsonSchema/00_Demands_Only.json']

    fixed_parameters = {}
    fixed_aircraft = {}
    customInputsfile = readArgv(argv)
    if not customInputsfile or not os.path.isfile(customInputsfile):
        print(f"Custom file {customInputsfile} does not exist")
        sys.exit(1)

    try:
        computation_mode, route_computation_mode, airports, distances, demands, _, fixed_parameters, fixed_aircraft = read_custom_inputs(
            CUSTOM_INPUTS_SCHEMA, customInputsfile)
    except Exception as err:
        print(
            f"Exception ocurred while playing custom inputs file {customInputsfile}")
        print(f"Error: {err}")
        sys.exit(1)

    # x = [37, 7, 5]
    # x = [9, 7, 5]
    # x = [33, 7, 4]
    # x = [32, 11, 5] # opt
    # x = [52,32,56]
    # x = [38, 29, 60]
    # x = [15,21,60] # mono
    x = [46,19,69] # mono

    if not fixed_aircraft:
        res, res2 = objective_function0(x, fixed_parameters, computation_mode,
                                        route_computation_mode, airports, distances, demands)

    return res, res2


if __name__ == "__main__":
    objective_function(sys.argv[1:])


# # # x = [9.700e+01,9.900e+01,4.400e+01,1.800e+01,-2.000e+00,3.200e+01, 4.800e+01,1.400e+01,3.000e+01,1.462e+03,1.700e+01,6.000e+01, 6.000e+00,1.525e+03]
# # # # x = [7.300e+01,8.600e+01,2.900e+01,1.600e+01,-5.000e+00,3.400e+01, 4.600e+01,2.000e+01,2.700e+01,1.372e+03,1.800e+01,1.160e+02, 4.000e+00,2.425e+03]
# # # # x = [1.210e+02,9.600e+01,4.100e+01,2.600e+01,-3.000e+00,3.600e+01, 6.200e+01,1.800e+01,2.900e+01,1.478e+03,1.800e+01,6.800e+01, 5.000e+00,1.975e+03]
# # # # x = [7.900e+01,9.400e+01,3.100e+01,2.000e+01,-4.000e+00,3.700e+01, 5.600e+01,1.000e+01,2.900e+01,1.448e+03,1.600e+01,8.200e+01, 5.000e+00,1.825e+03]
# # # # x = [1.270e+02,7.600e+01,3.600e+01,2.800e+01,-4.000e+00,3.800e+01, 6.000e+01,1.800e+01,3.000e+01,1.432e+03,1.700e+01,8.800e+01, 5.000e+00,1.225e+03]
# # # # x = [1.150e+02,8.400e+01,4.900e+01,3.200e+01,-2.000e+00,3.600e+01, 5.000e+01,1.400e+01,2.800e+01,1.492e+03,1.900e+01,1.100e+02, 4.000e+00,1.375e+03]
# # # # x = [1.090e+02,8.100e+01,2.600e+01,2.400e+01,-5.000e+00,4.000e+01, 5.200e+01,1.600e+01,2.700e+01,1.402e+03,1.400e+01,7.400e+01, 4.000e+00,2.125e+03]
# # # # x = [9.100e+01,8.900e+01,3.400e+01,3.000e+01,-3.000e+00,3.900e+01, 6.400e+01,1.200e+01,2.800e+01,1.358e+03,2.000e+01,9.600e+01, 5.000e+00,1.675e+03]
# # # # x = [8.500e+01,9.100e+01,3.900e+01,3.400e+01,-3.000e+00,3.300e+01, 5.800e+01,1.200e+01,2.800e+01,1.418e+03,1.600e+01,1.020e+02, 6.000e+00,2.275e+03]
# # # # x = [1.030e+02,7.900e+01,4.600e+01,2.200e+01,-4.000e+00,3.500e+01, 5.400e+01,1.600e+01,2.900e+01,1.388e+03,1.500e+01,5.400e+01, 6.000e+00,1.075e+03]

# # # x = [1.150e+02,8.400e+01,4.900e+01,3.200e+01,-2.000e+00,3.600e+01, 5.000e+01,1.400e+01,2.800e+01,1.492e+03,1.900e+01,1.100e+02, 4.000e+00,1.375e+03,41000, 78, 1, 1, 1, 1] # Prifit ok
# # # x =  [127, 82, 46, 22, -2, 44, 48, 21, 27, 1358, 22,  92, 5, 2875, 41200, 82, 1, 1, 1, 1]
# # # x =  [115, 84, 49, 32, -2, 36, 50, 14, 28, 1492, 19, 110, 4, 1375, 41000, 78, 1, 1, 1, 1] #good one
# # x =  [72, 86, 28, 26, -5, 34, 50, 13, 28, 1450, 14, 70, 4, 1600, 41000, 78, 1, 1, 1, 1] # Baseline

# x =  [130, 91, 38, 29, -4.5, 33, 62, 17, 30, 1480, 18, 144, 6, 1900, 41000, 78, 1, 1, 1, 1] # 144 seat
# # # x =  [121, 80, 40, 18, -2, 40, 52, 13, 28, 1358, 15, 108, 4, 1875, 41000, 82, 1, 1, 1, 1] # Baseline2
# # # # # # x = [int(x) for x in x]
# # # # # # print(x)

# x =  [121, 114, 27, 25, -4.0, 35, 50, 14, 29, 1430, 23, 142, 6, 1171, 41000, 78, 1, 1, 1, 1] # Optim_Jose

# # # # # x = [76, 118, 46, 23, -3, 33, 55, 19, 30, 1357, 18, 86, 6, 2412, 42260, 79, 1, 1, 1, 1]
# # # # # x = [91, 108, 50, 29, -3, 34, 52, 12, 27, 1366, 19, 204, 4, 1812, 39260, 80, 1, 1, 1, 1]
# # x = [110, 82, 34, 25, -5, 38, 52, 11, 30, 1462, 19, 92, 4, 1375, 39600, 80, 1, 1, 1, 1]
