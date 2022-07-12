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
from framework.Performance.Mission.mission_real import mission
from framework.Network.network_optimization import network_optimization
from framework.Economics.revenue import revenue
from framework.Sizing.airplane_sizing_check import airplane_sizing
import pandas as pd
import sys
import pickle
import numpy as np
import csv
from datetime import datetime
from random import randrange
from framework.utilities.logger import get_logger
from framework.utilities.output import write_optimal_results, write_kml_results, write_bad_results, write_newtork_results, write_unfeasible_results

import getopt
import haversine
import json
import jsonschema
import os

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Database.Airports.airports_database import AIRPORTS_DATABASE

from haversine import haversine, Unit
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
        try:
            status, flags, vehicle = airplane_sizing(vehicle, x)
        except:
            log.error(
                ">>>>>>>>>> Error at <<<<<<<<<<<< airplane_sizing", exc_info=True)

        # with open('Database/Family/161_to_220/all_dictionaries/'+str(5)+'.pkl', 'rb') as f:
        #     all_info_acft3 = pickle.load(f)

        # vehicle= all_info_acft3['vehicle']
        # status = 0
        # flags = [0,0,0,0,0]
        
        # status = 0
        results = vehicle['results']
        performance = vehicle['performance']
        operations = vehicle['operations']
        aircraft = vehicle['aircraft']

        # =============================================================================
        # If airplane pass checks, status = 0, else status = 1 and profit = 0
        if status == 0:
            log.info('Aircraft passed sizing and checks status: {}'.format(status))

            market_share = operations['market_share']

            results['nodes_number'] = len(airports)

            pax_capacity = aircraft['passenger_capacity']  # Passenger capacity
            # =============================================================================
            log.info('---- Start DOC matrix calculation ----')
            # The DOC is estimated for each city pair and stored in the DOC dictionary
            city_matrix_size = len(airports)*len(airports)
            DOC_ik = {}
            DOC_nd = {}
            fuel_mass = {}
            total_mission_flight_time = {}
            mach = {}
            passenger_capacity = {}
            SAR = {}

            airports_keys = list(airports.keys())
            try:
                for i in range(len(airports)):

                    DOC_ik[airports_keys[i]] = {}
                    DOC_nd[airports_keys[i]] = {}
                    fuel_mass[airports_keys[i]] = {}
                    total_mission_flight_time[airports_keys[i]] = {}
                    mach[airports_keys[i]] = {}
                    passenger_capacity[airports_keys[i]] = {}
                    SAR[airports_keys[i]] = {}

                    for k in range(len(airports)):

                        print(
                            'INFO >>>> Current pair: ', airports_keys[i],airports_keys[k])
                        DOC_nd[airports_keys[i]][airports_keys[k]] = 0
                        DOC_ik[airports_keys[i]][airports_keys[k]] = 0
                        fuel_mass[airports_keys[i]][airports_keys[k]] = 0
                        total_mission_flight_time[airports_keys[i]
                                                  ][airports_keys[k]] = 0
                        mach[airports_keys[i]][airports_keys[k]] = 0
                        passenger_capacity[airports_keys[i]
                                           ][airports_keys[k]] = 0
                        SAR[airports_keys[i]][airports_keys[k]] = 0

                        if (i != k) and (distances[airports_keys[i]][airports_keys[k]] <= performance['range']) and (demands[airports_keys[i]][airports_keys[k]]['demand'] > 0):
                            # Update information about orign-destination pair airports:
                            airport_departure = {}
                            airport_departure['latitude'] = airports[airports_keys[i]]['latitude']
                            airport_departure['longitude'] = airports[airports_keys[i]]['longitude']
                            airport_departure['elevation'] = airports[airports_keys[i]]['elevation']
                            airport_departure['dmg'] = airports[airports_keys[i]]['dmg']
                            airport_departure['tref'] = airports[airports_keys[i]]['tref']

                            takeoff_runway = airports[airports_keys[i]
                                                      ]['runways'][demands[airports_keys[i]][airports_keys[k]]['takeoff_runway']]

                            airport_destination = {}
                            airport_destination['latitude'] = airports[airports_keys[k]]['latitude']
                            airport_destination['longitude'] = airports[airports_keys[k]]['longitude']
                            airport_destination['elevation'] = airports[airports_keys[k]]['elevation']
                            airport_destination['dmg'] = airports[airports_keys[k]]['dmg']
                            airport_destination['tref'] = airports[airports_keys[k]]['tref']

                            landing_runway = airports[airports_keys[k]
                                                      ]['runways'][demands[airports_keys[i]][airports_keys[k]]['landing_runway']]

                            # Calculate DOC and mission parameters for origin-destination airports pair:
                            mission_range = distances[airports_keys[i]
                                                      ][airports_keys[k]]
                            fuel_mass[airports_keys[i]][airports_keys[k]], total_mission_flight_time[airports_keys[i]][airports_keys[k]], DOC, mach[airports_keys[i]][airports_keys[k]], passenger_capacity[airports_keys[i]
                                                                                                                                                                                                            ][airports_keys[k]], SAR[airports_keys[i]][airports_keys[k]], _ = mission(vehicle, airport_departure, takeoff_runway, airport_destination, landing_runway, mission_range,airports_keys[i],airports_keys[k])
                            DOC_nd[airports_keys[i]][airports_keys[k]] = DOC
                            DOC_ik[airports_keys[i]][airports_keys[k]] = int(
                                DOC*mission_range)

                        elif (i != k) and (demands[airports_keys[i]][airports_keys[k]]['demand'] > 0) and computation_mode == 0:
                            DOC_ik[airports_keys[i]][airports_keys[k]] = 1e10

                        city_matrix_size = city_matrix_size - 1
                        print(
                            'INFO >>>> city pairs remaining to finish DOC matrix fill: ', city_matrix_size)

                log.info('---- End DOC matrix calculation ----')
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< DOC matrix generation", exc_info=True)

            # with open('Database/DOC/DOC.csv', 'w') as f:
            #     for key in DOC_ik.keys():
            #         f.write("%s,%s\n"%(key,DOC_ik[key]))

            # with open('Database/DOC/DOC.pkl', 'wb') as f:
            #     pickle.dump(DOC_ik, f, pickle.HIGHEST_PROTOCOL)

            # with open('Database/DOC/Vehicle.pkl', 'wb') as f:
            #     pickle.dump(vehicle, f, pickle.HIGHEST_PROTOCOL)

            log.info('Aircraft DOC matrix: {}'.format(DOC_ik))
            # =============================================================================
            log.info('---- Start Network Optimization ----')
            # Network optimization that maximizes the network profit
            try:
                profit, vehicle, kpi_df1, kpi_df2, airplanes_ik = network_optimization(
                    computation_mode, list(airports.keys()), distances, demands, DOC_ik, vehicle)
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
                mach_flatt = flatten_dict(mach)
                mach_df = pd.DataFrame.from_dict(
                    mach_flatt, orient="index", columns=['mach'])
                passenger_capacity_flatt = flatten_dict(passenger_capacity)
                passenger_capacity_df = pd.DataFrame.from_dict(
                    passenger_capacity_flatt, orient="index", columns=['pax_num'])
                fuel_used_flatt = flatten_dict(fuel_mass)
                fuel_used_df = pd.DataFrame.from_dict(
                    fuel_used_flatt, orient="index", columns=['fuel'])
                mission_time_flatt = flatten_dict(total_mission_flight_time)
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

                # mach_flatt = flatten_dict(mach)
                # mach_df =  pd.DataFrame.from_dict(mach_flatt,orient="index",columns=['mach'])
                # passenger_capacity_flatt = flatten_dict(passenger_capacity)
                # passenger_capacity_df =  pd.DataFrame.from_dict(passenger_capacity_flatt,orient="index",columns=['pax_num'])
                # fuel_used_flatt = flatten_dict(fuel_mass)
                # fuel_used_df =  pd.DataFrame.from_dict(fuel_used_flatt,orient="index",columns=['fuel'])
                # mission_time_flatt = flatten_dict(total_mission_flight_time)
                # mission_time_df =  pd.DataFrame.from_dict(mission_time_flatt,orient="index",columns=['time'])
                # DOC_nd_flatt = flatten_dict(DOC_nd)
                # DOC_nd_df =  pd.DataFrame.from_dict(DOC_nd_flatt,orient="index",columns=['DOC_nd'])
                # SAR_flatt = flatten_dict(SAR)
                # SAR_df =  pd.DataFrame.from_dict(SAR_flatt,orient="index",columns=['SAR'])

                kpi_df3['mach'] = mach_df['mach'].values
                kpi_df3['pax_num'] = passenger_capacity_df['pax_num'].values
                kpi_df3['fuel'] = fuel_used_df['fuel'].values
                kpi_df3['time'] = mission_time_df['time'].values
                kpi_df3['DOC_nd'] = DOC_nd_df['DOC_nd'].values
                kpi_df3['SAR'] = SAR_df['SAR'].values

                kpi_df3 = kpi_df3.drop(kpi_df3[kpi_df3.distances01 == 0].index)
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
                    ((1/kpi_df2['SAR'])*(1/(aircraft['wetted_area']**0.24)))
                # Total distance
                kpi_df2['total_distance'] = kpi_df2['aircraft_number'] * \
                    kpi_df2['distances']
                # Total pax
                kpi_df2['total_pax'] = kpi_df2['aircraft_number'] * \
                    kpi_df2['pax_num']
                # Total cost
                kpi_df2['total_cost'] = kpi_df2['aircraft_number'] * \
                    kpi_df2['doc']
                results['network_density'] = results['arcs_number'] / \
                    (results['nodes_number'] *
                     results['nodes_number']-results['nodes_number'])
                kpi_df2['total_time'] = kpi_df2['aircraft_number'] * \
                    kpi_df2['time']
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< writting dataframes", exc_info=True)

            try:
                write_optimal_results(x,list(airports.keys(
                )), distances, demands, profit, DOC_ik, vehicle, kpi_df2, airplanes_ik)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< write_optimal_results", exc_info=True)

            try:
                write_kml_results(airports, profit, airplanes_ik)
            except:
                log.error(
                    ">>>>>>>>>> Error at <<<<<<<<<<<< write_kml_results", exc_info=True)

            try:
                write_newtork_results(profit, kpi_df1, kpi_df2)
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

    return profit,

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

def objective_function(vehicle,x=None):

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

    # x = [121, 114, 27, 25, -4.0, 35, 50, 14, 29, 1430, 23, 142, 6, 1171, 41000, 78, 1, 1, 1, 1]
    # x = [1.04013290e+02,  8.71272735e+01,  3.42639119e+01,  2.12550036e+01,
    #    -3.42824373e+00,  4.12149389e+01,  4.98606638e+01,  1.47169661e+01,
    #     2.87241618e+01,  1.36584947e+03,  2.09763441e+01,  1.61607474e+02,
    #     5.55661531e+00,  1.27054142e+03,  4.10000000e+04,  7.80000000e+01,
    #                1,            1,             1,            1]
    #    0      1   2   3     4     5    6   7  8     9   10   11  12  13    14    15  16 17 18 19
    # x = [130,  91, 38, 29,  -4.5,   33, 62, 17, 30,
    #      1480, 18, 144, 6, 3000, 0]
    # # x = [130, 100, 30, 25, -2.25, 38.5, 60, 20, 27, 1350, 15, 250, 6, 3000, 37000, 78, 1, 1, 1, 1] # Sylvain

    # distances = {'FRA': {'FRA': 0, 'LHR': 355, 'CDG': 243, 'AMS': 198, 'MAD': 768, 'BCN': 591, 'FCO': 517, 'DUB': 589, 'VIE': 336, 'ZRH': 154}, 'LHR': {'FRA': 355, 'LHR': 0, 'CDG': 188, 'AMS': 200, 'MAD': 672, 'BCN': 620, 'FCO': 781, 'DUB': 243, 'VIE': 690, 'ZRH': 427}, 'CDG': {'FRA': 243, 'LHR': 188, 'CDG': 0, 'AMS': 215, 'MAD': 574, 'BCN': 463, 'FCO': 595, 'DUB': 425, 'VIE': 561, 'ZRH': 258}, 'AMS': {'FRA': 198, 'LHR': 200, 'CDG': 215, 'AMS': 0, 'MAD': 788, 'BCN': 670, 'FCO': 700, 'DUB': 406, 'VIE': 519, 'ZRH': 326}, 'MAD': {'FRA': 768, 'LHR': 672, 'CDG': 574, 'AMS': 788, 'MAD': 0, 'BCN': 261, 'FCO': 720, 'DUB': 784, 'VIE': 977, 'ZRH': 670}, 'BCN': {
    #     'FRA': 591, 'LHR': 620, 'CDG': 463, 'AMS': 670, 'MAD': 261, 'BCN': 0, 'FCO': 459, 'DUB': 802, 'VIE': 741, 'ZRH': 463}, 'FCO': {'FRA': 517, 'LHR': 781, 'CDG': 595, 'AMS': 700, 'MAD': 720, 'BCN': 459, 'FCO': 0, 'DUB': 1020, 'VIE': 421, 'ZRH': 375}, 'DUB': {'FRA': 589, 'LHR': 243, 'CDG': 425, 'AMS': 406, 'MAD': 784, 'BCN': 802, 'FCO': 1020, 'DUB': 0, 'VIE': 922, 'ZRH': 670}, 'VIE': {'FRA': 336, 'LHR': 690, 'CDG': 561, 'AMS': 519, 'MAD': 977, 'BCN': 741, 'FCO': 421, 'DUB': 922, 'VIE': 0, 'ZRH': 327}, 'ZRH': {'FRA': 154, 'LHR': 427, 'CDG': 258, 'AMS': 326, 'MAD': 670, 'BCN': 463, 'FCO': 375, 'DUB': 670, 'VIE': 327, 'ZRH': 0}}

    if not fixed_aircraft:
        objective_function0(x, fixed_parameters, computation_mode,
                           route_computation_mode, airports, distances, demands)

    return 0


def main(argv):

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

    # x = [121, 114, 27, 25, -4.0, 35, 50, 14, 29, 1430, 23, 142, 6, 1171, 41000, 78, 1, 1, 1, 1]
    # x = [1.04013290e+02,  8.71272735e+01,  3.42639119e+01,  2.12550036e+01,
    #    -3.42824373e+00,  4.12149389e+01,  4.98606638e+01,  1.47169661e+01,
    #     2.87241618e+01,  1.36584947e+03,  2.09763441e+01,  1.61607474e+02,
    #     5.55661531e+00,  1.27054142e+03,  4.10000000e+04,  7.80000000e+01,
    #                1,            1,             1,            1]
    #    0      1   2   3     4     5    6   7  8     9   10   11  12  13    14    15  16 17 18 19

        #     0   | 1   | 2   |  3     |   4    |   5      | 6    | 7        |  8     |   9    |
        #    Areaw| ARw | TRw | Sweepw | Twistw | b/2kinkw | pax  | seat abr | range  | engine |
    # x = [90,    70,   20,     20,      -5,        30,     161,       4,       1500,     45]

    x = [130,    91,   35,     29,      -4.5,        33,     120,       6,       1000,     35]
    # # x = [130, 100, 30, 25, -2.25, 38.5, 250, 6, 3000,44] # Sylvain

    # distances = {'FRA': {'FRA': 0, 'LHR': 355, 'CDG': 243, 'AMS': 198, 'MAD': 768, 'BCN': 591, 'FCO': 517, 'DUB': 589, 'VIE': 336, 'ZRH': 154}, 'LHR': {'FRA': 355, 'LHR': 0, 'CDG': 188, 'AMS': 200, 'MAD': 672, 'BCN': 620, 'FCO': 781, 'DUB': 243, 'VIE': 690, 'ZRH': 427}, 'CDG': {'FRA': 243, 'LHR': 188, 'CDG': 0, 'AMS': 215, 'MAD': 574, 'BCN': 463, 'FCO': 595, 'DUB': 425, 'VIE': 561, 'ZRH': 258}, 'AMS': {'FRA': 198, 'LHR': 200, 'CDG': 215, 'AMS': 0, 'MAD': 788, 'BCN': 670, 'FCO': 700, 'DUB': 406, 'VIE': 519, 'ZRH': 326}, 'MAD': {'FRA': 768, 'LHR': 672, 'CDG': 574, 'AMS': 788, 'MAD': 0, 'BCN': 261, 'FCO': 720, 'DUB': 784, 'VIE': 977, 'ZRH': 670}, 'BCN': {
    #     'FRA': 591, 'LHR': 620, 'CDG': 463, 'AMS': 670, 'MAD': 261, 'BCN': 0, 'FCO': 459, 'DUB': 802, 'VIE': 741, 'ZRH': 463}, 'FCO': {'FRA': 517, 'LHR': 781, 'CDG': 595, 'AMS': 700, 'MAD': 720, 'BCN': 459, 'FCO': 0, 'DUB': 1020, 'VIE': 421, 'ZRH': 375}, 'DUB': {'FRA': 589, 'LHR': 243, 'CDG': 425, 'AMS': 406, 'MAD': 784, 'BCN': 802, 'FCO': 1020, 'DUB': 0, 'VIE': 922, 'ZRH': 670}, 'VIE': {'FRA': 336, 'LHR': 690, 'CDG': 561, 'AMS': 519, 'MAD': 977, 'BCN': 741, 'FCO': 421, 'DUB': 922, 'VIE': 0, 'ZRH': 327}, 'ZRH': {'FRA': 154, 'LHR': 427, 'CDG': 258, 'AMS': 326, 'MAD': 670, 'BCN': 463, 'FCO': 375, 'DUB': 670, 'VIE': 327, 'ZRH': 0}}

    if not fixed_aircraft:
        objective_function0(x, fixed_parameters, computation_mode,
                           route_computation_mode, airports, distances, demands)


import cProfile
if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main(sys.argv[1:])

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats() 

    stats = pstats.Stats(profiler)
    stats.dump_stats('p02.prof')



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
