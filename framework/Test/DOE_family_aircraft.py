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
import copy
import csv
import getopt
import json
import os
import pickle
import sys
from datetime import datetime
from multiprocessing import Pool
from random import randrange

import haversine
import jsonschema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from framework.Database.Aircrafts.baseline_aircraft_parameters import \
    initialize_aircraft_parameters
from framework.Database.Airports.airports_database import AIRPORTS_DATABASE
from framework.Economics.revenue import revenue
from framework.Network.network_optimization import network_optimization
from framework.Performance.Mission.mission import mission
from framework.Sizing.airplane_sizing_check import airplane_sizing
from framework.utilities.logger import get_logger
from framework.utilities.output import (write_bad_results, write_kml_results,
                                        write_newtork_results,
                                        write_optimal_results,
                                        write_unfeasible_results)
from haversine import Unit, haversine
from jsonschema import validate
from pymoo.factory import get_sampling
from pymoo.interface import sample

# =============================================================================
# IMPORTS
# =============================================================================
from aux_tools import corrdot

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])


def objective_function(args):

    x = args[0]
    original_vehicle= args[1]
    computation_mode= args[2]
    route_computation_mode= args[3]
    airports= args[4]
    distances= args[5]
    demands = args[6]
    index= args[7]

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
                                                                                                                                                                                                            ][airports_keys[k]], SAR[airports_keys[i]][airports_keys[k]] = mission(vehicle, airport_departure, takeoff_runway, airport_destination, landing_runway, mission_range)
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
                write_optimal_results(args[0],list(airports.keys(
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

            print("Save dictionary results:")
            all_in_one = {'airports': airports, 'distances': distances, 'demands': demands, 'DOC_ik': DOC_ik, 'DOC_nd': DOC_nd,
                'fuel_mass': fuel_mass, 'total_mission_flight_time': total_mission_flight_time , 'mach': mach, 'passenger_capacity':passenger_capacity, 'SAR':SAR,  'vehicle': vehicle}

            with open('Database/Family/161_to_220/all_dictionaries/'+str(index)+'.pkl', 'wb') as f:
                pickle.dump(all_in_one, f)

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

    return profit


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


# IMPORTS


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
        n_inputs = 10

        # Lower and upeer bounds of each input variable
        #     0   | 1   | 2   |  3     |   4    |   5      | 6    | 7        |  8     |   9    |
        #    Areaw| ARw | TRw | Sweepw | Twistw | b/2kinkw | pax  | seat abr | range  | engine |
        # lb = [72,    75,   25,     0,      -5,       32,     40,       4,       1000,    0]
        # ub = [130,  120,  50,     30,       0 ,       45,     100,       6,       3500,    50]

        # lb = [40,    70,   20,     0,      -5,        30,     40,        3,       950,     0]
        # ub = [100,  120,   50,     25,       0 ,       45,     100,       6,       1955,    60]

        # lb = [70,    70,   20,     15,      -5,        30,     101,       4,       1300,     0]
        # ub = [130,  120,   50,     25,       0 ,       45,     160,       6,       3200,    44]

        lb = [90,    70,   20,     20,      -5,        30,     161,       4,       1500,     0]
        ub = [290,  120,   50,     35,       0 ,       45,     220,       6,       3200,    44]




        # Desired number of samples
        n_samples = 200

        # Sampling type
        # sampling_type = 'real_random'
        # sampling_type = 'int_lhs'
        sampling_type = 'real_lhs'

        # Plot type (0-simple, 1-complete)
        plot_type = 1
        # =========================================

        # EXECUTION

        # Set random seed to make results repeatable
        np.random.seed(321)

        # Initialize sampler
        sampling = get_sampling(sampling_type)

        # Draw samples
        X = sample(sampling, n_samples, n_inputs)

        vehicle = initialize_aircraft_parameters()

        # Samples are originally between 0 and 1,
        # so we need to scale them to the desired interval
        for ii in range(n_inputs):
            X[:, ii] = lb[ii] + (ub[ii] - lb[ii])*X[:, ii]

        # Execute all cases and store outputs
        y1_samples = []
        # y2_samples = []
        index = 0

        input_array = []
        for ii in range(n_samples):

            # Evaluate sample
            # (y1)= objective_function(vehicle,X[ii,:])
            # y1 = objective_function(
            #     X[ii, :], fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands,index)
            # y1_samples.append(float(y1))

            input_array.append([X[ii, :], fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands,index])

            index = index+1


        # input_array = zip(X[ii, :], fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands,range(n_samples ))

        with Pool(14) as p:
            y1 = p.map(objective_function,input_array)
            y1_samples = float(y1)


        # y2_samples.append(y2)
        # Create a pandas dataframe with all the information
        df = pd.DataFrame({'Sw': X[:, 0],
                           'ARw': X[:, 1],
                           'TRw': X[:, 2],
                           'Sweepw': X[:, 3],
                           'Twistw': X[:, 4],
                           'kinkw': X[:, 5],
                           'pax': X[:, 6],
                           'seatabr': X[:, 7],
                           'range': X[:, 8],
                           'engine': X[:, 9],
                           'profit': y1_samples})
        # Plot the correlation matrix

        df.to_pickle('doe.pkl')
        sns.set(style='white', font_scale=1.4)

        if plot_type == 0:
        
            # Simple plot
            ax = sns.pairplot(df,corner=True)
        
        elif plot_type == 1:
        
        
            # Complete plot
            # based on: https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
            ax = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
            ax.map_lower(sns.regplot, lowess=True, line_kws={'color': 'black'})
            ax.map_diag(sns.histplot)
            ax.map_upper(corrdot)

        for ax in ax.axes[:,0]:
            ax.get_yaxis().set_label_coords(-0.22,0.5)

        # Plot window
        plt.tight_layout()
        plt.show()
        
        plt.savefig('doe.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    except Exception as err:
        print(
            f"Exception ocurred while playing custom inputs file {customInputsfile}")
        print(f"Error: {err}")
        sys.exit(1)

    # distances = {'FRA': {'FRA': 0, 'LHR': 355, 'CDG': 243, 'AMS': 198, 'MAD': 768, 'BCN': 591, 'FCO': 517, 'DUB': 589, 'VIE': 336, 'ZRH': 154}, 'LHR': {'FRA': 355, 'LHR': 0, 'CDG': 188, 'AMS': 200, 'MAD': 672, 'BCN': 620, 'FCO': 781, 'DUB': 243, 'VIE': 690, 'ZRH': 427}, 'CDG': {'FRA': 243, 'LHR': 188, 'CDG': 0, 'AMS': 215, 'MAD': 574, 'BCN': 463, 'FCO': 595, 'DUB': 425, 'VIE': 561, 'ZRH': 258}, 'AMS': {'FRA': 198, 'LHR': 200, 'CDG': 215, 'AMS': 0, 'MAD': 788, 'BCN': 670, 'FCO': 700, 'DUB': 406, 'VIE': 519, 'ZRH': 326}, 'MAD': {'FRA': 768, 'LHR': 672, 'CDG': 574, 'AMS': 788, 'MAD': 0, 'BCN': 261, 'FCO': 720, 'DUB': 784, 'VIE': 977, 'ZRH': 670}, 'BCN': {
    #     'FRA': 591, 'LHR': 620, 'CDG': 463, 'AMS': 670, 'MAD': 261, 'BCN': 0, 'FCO': 459, 'DUB': 802, 'VIE': 741, 'ZRH': 463}, 'FCO': {'FRA': 517, 'LHR': 781, 'CDG': 595, 'AMS': 700, 'MAD': 720, 'BCN': 459, 'FCO': 0, 'DUB': 1020, 'VIE': 421, 'ZRH': 375}, 'DUB': {'FRA': 589, 'LHR': 243, 'CDG': 425, 'AMS': 406, 'MAD': 784, 'BCN': 802, 'FCO': 1020, 'DUB': 0, 'VIE': 922, 'ZRH': 670}, 'VIE': {'FRA': 336, 'LHR': 690, 'CDG': 561, 'AMS': 519, 'MAD': 977, 'BCN': 741, 'FCO': 421, 'DUB': 922, 'VIE': 0, 'ZRH': 327}, 'ZRH': {'FRA': 154, 'LHR': 427, 'CDG': 258, 'AMS': 326, 'MAD': 670, 'BCN': 463, 'FCO': 375, 'DUB': 670, 'VIE': 327, 'ZRH': 0}}

    # if not fixed_aircraft:
    #     objective_function(x, fixed_parameters, computation_mode,
    #                        route_computation_mode, airports, distances, demands)


if __name__ == "__main__":
    main(sys.argv[1:])


