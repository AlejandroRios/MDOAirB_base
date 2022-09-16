"""
File name : Network profit function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : July 2020
Last edit : February 2021
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    - This module calculates the network profit following the following steps:
        - Vehicle sizing and checks (airplane_sizing)
        - Revenue calculation (reveneu)
        - Direct operational cost calculation (mission)
        - Profit calculation (network_optimization)

Inputs:
    - Optimization variables (array x)
    - Mutable dictionary with aircraft, perfomance, operations and airports
    departure and destiny information
Outputs:
    - Profit wich is the objective function
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
import copy
from framework.Performance.Mission.mission import mission
from framework.Network.network_optimization import network_optimization
from framework.Economics.revenue import revenue
from framework.Sizing.airplane_sizing_check_fpo import airplane_sizing, objective_function_FPO
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
from datetime import datetime

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Database.Airports.airports_database_fpo import AIRPORTS_DATABASE

from haversine import haversine, Unit
from jsonschema import validate

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])

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

# CUSTOM_INPUTS_SCHEMA = 'Database/JsonSchema/Custom_Inputs.schema.json'

CUSTOM_INPUTS_SCHEMA = 'Database/JsonSchema/Aircraft_Input_Data.schema.json'


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
                    f'Take-of runway {takeoff_runway} do not exit for airport {departure} in MDO database'
                )
            landing_runway = demands[departure][arrival]["landing_runway"]
            if landing_runway not in airports[arrival]["runways"]:
                raise CustomInputsError(
                    f'Landing runway {landing_runway} do not exit for airport {arrival} in MDO database'
                )


def check_airports(airport_keys):
    try:
        airports = {k: AIRPORTS_DATABASE[k] for k in airport_keys}
    except KeyError as key_error:
        raise CustomInputsError(
            f'Airports {key_error.args} do not exit in MDO database')

    return airports


def haversine_distance(coordinates_departure, coordinates_arrival):
    # Perform haversine distance calculation in nautical miles
    distance = float(
        haversine(coordinates_departure, coordinates_arrival, unit='nmi'))
    return distance


def check_demands(data, fixed_parameters):
    airport_keys = list(data.keys())
    for key in data:
        if (key in data[key]):
            raise CustomInputsError(
                f'Airport {key} exist on both departure and arrival for the same demand'
            )
        airport_keys = airport_keys + list(
            set(data[key].keys()) - set(airport_keys))

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

                coordinates_departure = (airports[departure]["latitude"],
                                         airports[departure]["longitude"])
                coordinates_arrival = (airports[arrival]["latitude"],
                                       airports[arrival]["longitude"])
                distances[departure][arrival] = round(
                    haversine_distance(coordinates_departure,
                                       coordinates_arrival))

    return airports, distances, data


def check_design_variables(data):
    for key in data:
        if (data[key]["lower_band"] > data[key]["upper_band"]):
            raise CustomInputsError(
                f'Lower band {data[key]["lower_band"]} is greater than upper band {data[key]["upper_band"]} for {key}'
            )


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
        fixed_parameters = update_vehicle(fixed_parameters,
                                          custom_fixed_parameters)

        if ("fixed_aircraft" in data):
            fixed_aircraft = data["fixed_aircraft"]

        airports, distances, demands = check_demands(data["demands"],
                                                     fixed_parameters)

    except OSError as os_error:
        raise CustomInputsError(f"{os_error.strerror} [{os_error.filename}]")
    except json.JSONDecodeError as dec_error:
        raise CustomInputsError(
            f"{dec_error.msg} [line {dec_error.lineno} column {dec_error.colno} (char {dec_error.pos})]"
        )
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


def main(argv):
    fixed_parameters = {}
    fixed_aircraft = {}
    # readArgv(argv)
    customInputsfile = "Database/JsonSchema/05_FPO_Use_case.json"
    if not customInputsfile or not os.path.isfile(customInputsfile):
        print(f"Custom file {customInputsfile} does not exist")
        sys.exit(1)

    try:
        computation_mode, route_computation_mode, airports, distances, demands, _, fixed_parameters, fixed_aircraft = read_custom_inputs(
            CUSTOM_INPUTS_SCHEMA, customInputsfile)
    except Exception as err:
        print(
            f"Exception ocurred while playing custom inputs file {customInputsfile}"
        )
        print(f"Error: {err}")
        sys.exit(1)

    distances = {'FAD': {'FAD': 0, 'FAA': 800}, 'FAA': {'FAA': 0, 'FAD': 800}}

    WingArea = np.array([150, 155, 160])# np.linspace(135, 150, 4)
    DiamRef = 2.2*10 # x % of thrust is equivalent to x^2% of fan diameter (prop to surface)
    FanDiameter = np.array([DiamRef*(1-np.sqrt(0.067)), DiamRef, DiamRef*(1+np.sqrt(0.067))])
    status = []

    X = []
    Y = []
    for wa in WingArea:
        for fd in FanDiameter:
            x = [
                wa,  # WingArea - x0
                10 * (42**2) / wa,  # AspectRatio x 10 - x1
                30,  # TaperRatio - x2
                25,  # sweep_c4 - x3
                -2.25,  # twist - x4
                38.5,  # semi_span_kink - x5
                250,  # PAX number - x6
                6,  # seat abreast - x7
                3000,  # range - x8
                80,  # BPR - x9
                fd,  # FanDiameter - x10
                27,  # Compressor pressure ratio - x11
                1350,  # turbine inlet temperature - x12
                15,  # FPR - x13
                38000,  # design point pressure - x14
                78  # design point mach x 10 - x15
            ]
            X.append(x)
            # if not fixed_aircraft:
            (y1) = objective_function_FPO(
                x, fixed_parameters, computation_mode, route_computation_mode,
                airports, distances, demands)
            Y.append(list(y1))
            #print("==============================================")
            #print("Results = ", y1[0:23])
            #print("==============================================")

    # Create a pandas dataframe with all the information
    X = np.array(X)
    Y = np.array(Y)
    df_input = pd.DataFrame({'WingArea': X[:,0],
                            'x1': X[:,1],
                             'x2': X[:,2],
                             'x3': X[:,3],
                             'x4': X[:,4],
                             'x5': X[:,5],
                             'x6': X[:,6],
                             'x7': X[:,7],
                             'x8': X[:,8],
                             'x9': X[:,9],
                             'FanDiameter': X[:,10],
                             'x11': X[:,11],
                             'x12': X[:,12],
                             'x13': X[:,13],
                             'x14': X[:,14],
                             'x15': X[:,15]})
    df_output = pd.DataFrame({'MTOW': Y[:,0],
                              'DOC': Y[:,1],
                              'fuel_mass': Y[:,2],
                              'total_mission_flight_time': Y[:,3],
                              'mach': Y[:,4],
                              'passenger_capacity': Y[:,5],
                              'SAR': Y[:,6],
                              'landing_field_length_computed': Y[:,7],
                              'takeoff_field_length_computed': Y[:,8],
                              'app_speed': Y[:,9],
                              'status': Y[:,10],
                              'design_status': Y[:,11],
                              'distance': Y[:,12], 
                              'altitude': Y[:,13], 
                              'mass': Y[:, 14],
                              'time': Y[:,15],
                              'sfc':  Y[:,16],
                              'thrust': Y[:,17], 
                              'mach': Y[:,18], 
                              'CL': Y[:,19], 
                              'CD': Y[:,20], 
                              'LoD': Y[:,21], 
                              'throttle': Y[:,22], 
                              'vcas': Y[:,23],
                              'OWE': Y[:,24],
                              })

    df_input.to_csv(
        "Results/analysis_input_refstudy.csv", index=False)
    df_output.to_csv(
        "Results/analysis_output_refstudy.csv", index=False)

    df_input.to_csv("Results/analysis_input"+datetime.now().strftime("%d_%m_%Y-%H_%M_%S")+".csv", index=False)
    df_output.to_csv(
        "Results/analysis_output"+datetime.now().strftime("%d_%m_%Y-%H_%M_S")+".csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
