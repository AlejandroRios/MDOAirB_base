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
from framework.Performance.Mission.mission import mission
from framework.Network.network_optimization import network_optimization
from framework.Economics.revenue import revenue
from framework.Test.airplane_check_modules_test import airplane_sizing
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

def objective_function(x, original_vehicle, computation_mode, route_computation_mode, airports, distances, demands):


    # Do a copy of original vehicle
    vehicle = copy.deepcopy(original_vehicle)
    # with open('Database/DOC/Vehicle.pkl', 'rb') as f:
    #     vehicle = pickle.load(f)

    # Try running profit calculation. If error appears during run profit = 0
    # =============================================================================
    # Airplane sizing and checks
    try:
        status, flags, vehicle = airplane_sizing(vehicle,x)
    except:
        status = 2
 
    if status == 0:
        profit = 1
    elif status == 1:
        profit = 0
    elif status == 2:
        profit = 0

    return profit,

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
				raise CustomInputsError(f'Take-of runway {takeoff_runway} do not exit for airport {departure} in MDO database')
			landing_runway = demands[departure][arrival]["landing_runway"]
			if landing_runway not in airports[arrival]["runways"]:
				raise CustomInputsError(f'Landing runway {landing_runway} do not exit for airport {arrival} in MDO database')

def check_airports(airport_keys):
	try:
		airports = { k: AIRPORTS_DATABASE[k] for k in airport_keys }
	except KeyError as key_error:
		raise CustomInputsError(f'Airports {key_error.args} do not exit in MDO database')

	return airports

def haversine_distance(coordinates_departure,coordinates_arrival):
    # Perform haversine distance calculation in nautical miles
    distance = float(haversine(coordinates_departure,coordinates_arrival,unit='nmi'))
    return distance

def check_demands(data, fixed_parameters):
	airport_keys = list(data.keys())
	for key in data:
		if (key in data[key]):
			raise CustomInputsError(f'Airport {key} exist on both departure and arrival for the same demand')
		airport_keys = airport_keys + list(set(data[key].keys()) - set(airport_keys))

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
				data[departure][arrival]['demand'] = np.round(market_share * data[departure][arrival]['demand'])

				coordinates_departure = (airports[departure]["latitude"],airports[departure]["longitude"])
				coordinates_arrival = (airports[arrival]["latitude"],airports[arrival]["longitude"])
				distances[departure][arrival] = round(haversine_distance(coordinates_departure,coordinates_arrival))

	return airports, distances, data

def check_design_variables(data):
	for key in data:
		if (data[key]["lower_band"] > data[key]["upper_band"]):
			raise CustomInputsError(f'Lower band {data[key]["lower_band"]} is greater than upper band {data[key]["upper_band"]} for {key}')

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
		fixed_parameters = update_vehicle(fixed_parameters, custom_fixed_parameters)

		if ("fixed_aircraft" in data):
			fixed_aircraft = data["fixed_aircraft"]

		airports, distances, demands = check_demands(data["demands"], fixed_parameters)

	except OSError as os_error:
		raise CustomInputsError(f"{os_error.strerror} [{os_error.filename}]")
	except json.JSONDecodeError as dec_error:
		raise CustomInputsError(f"{dec_error.msg} [line {dec_error.lineno} column {dec_error.colno} (char {dec_error.pos})]")
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

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymoo.factory import get_sampling
from pymoo.interface import sample
from aux_tools import corrdot

def main(argv):
	fixed_parameters = {}
	fixed_aircraft = {}
	customInputsfile = readArgv(argv)
	if not customInputsfile or not os.path.isfile(customInputsfile):
		print(f"Custom file {customInputsfile} does not exist")
		sys.exit(1)

	try:
		computation_mode, route_computation_mode, airports, distances, demands, _, fixed_parameters, fixed_aircraft = read_custom_inputs(CUSTOM_INPUTS_SCHEMA, customInputsfile)
		n_inputs = 16
		
		# Lower and upeer bounds of each input variable
		#     0   | 1   | 2   |  3     |   4    |   5      | 6     | 7     |  8     |   9   | 10    | 11    |   12     | 13    | 14          |  15
		#    Areaw| ARw | TRw | Sweepw | Twistw | b/2kinkw | bypass| Ediam | PRcomp |  Tin  | PRfan | PAX   | seat abr | range | design pres | mach
		lb = [72,  75 ,  25,     15,      -5,       32,       45,    10,      27,      1350,   14,     70,     4,        1000,     39000,       78]
		ub = [130, 120,  50,     30,      -2,       45,       65,    15,      30,      1500,   25,     220,    6,        3500,     43000,       82]
		# Desired number of samples
		n_samples = 200
		
		# Sampling type
		sampling_type = 'real_random'
		# sampling_type = 'int_lhs'
		
		# Plot type (0-simple, 1-complete)
		plot_type = 1
		#=========================================
		
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
		    X[:,ii] = lb[ii] + (ub[ii] - lb[ii])*X[:,ii]
		
		# Execute all cases and store outputs
		y1_samples = []
		# y2_samples = []
		for ii in range(n_samples):
		
		    # Evaluate sample
		    # (y1)= objective_function(vehicle,X[ii,:])
		    (y1) = objective_function(X[ii,:], fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands)
		    # Store the relevant information
		    y1_samples.append(y1)
		# y2_samples.append(y2)
		
		# Create a pandas dataframe with all the information
		df = pd.DataFrame({'x1' : X[:,0],
		                'x2' : X[:,1],
		                'x3' : X[:,2],
		                'x4' : X[:,3],
		                'x5' : X[:,4],
		                'x6' : X[:,5],
		                'x7' : X[:,6],
		                'x8' : X[:,7],
		                'x9' : X[:,8],
		                'x10' : X[:,9],
		                'x11' : X[:,10],
		                'x12' : X[:,11],
		                'x13' :X[:,12],
		                'x14' :X[:,13],
		                'x15' :X[:,14],
		                'x16' :X[:,15],
		                'y1' : y1_samples})
		# Plot the correlation matrix
		sns.set(style='white', font_scale=1.4)
		
		if plot_type == 0:
		
		    # Simple plot
		    fig = sns.pairplot(df,corner=True)
		
		elif plot_type == 1:
		
		    # Complete plot
		    # based on: https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
		    fig = sns.PairGrid(df, diag_sharey=False)
		    fig.map_lower(sns.regplot, lowess=True, line_kws={'color': 'black'})
		    fig.map_diag(sns.histplot)
		    fig.map_upper(corrdot)
		
		# Plot window
		plt.tight_layout()
		plt.show()
	except Exception as err:
		print(f"Exception ocurred while playing custom inputs file {customInputsfile}")
		print(f"Error: {err}")
		sys.exit(1)




	distances = {'FRA': {'FRA': 0, 'LHR': 355, 'CDG': 243, 'AMS': 198, 'MAD': 768, 'BCN': 591, 'FCO': 517, 'DUB': 589, 'VIE': 336, 'ZRH': 154}, 'LHR': {'FRA': 355, 'LHR': 0, 'CDG': 188, 'AMS': 200, 'MAD': 672, 'BCN': 620, 'FCO': 781, 'DUB': 243, 'VIE': 690, 'ZRH': 427}, 'CDG': {'FRA': 243, 'LHR': 188, 'CDG': 0, 'AMS': 215, 'MAD': 574, 'BCN': 463, 'FCO': 595, 'DUB': 425, 'VIE': 561, 'ZRH': 258}, 'AMS': {'FRA': 198, 'LHR': 200, 'CDG': 215, 'AMS': 0, 'MAD': 788, 'BCN': 670, 'FCO': 700, 'DUB': 406, 'VIE': 519, 'ZRH': 326}, 'MAD': {'FRA': 768, 'LHR': 672, 'CDG': 574, 'AMS': 788, 'MAD': 0, 'BCN': 261, 'FCO': 720, 'DUB': 784, 'VIE': 977, 'ZRH': 670}, 'BCN': {'FRA': 591, 'LHR': 620, 'CDG': 463, 'AMS': 670, 'MAD': 261, 'BCN': 0, 'FCO': 459, 'DUB': 802, 'VIE': 741, 'ZRH': 463}, 'FCO': {'FRA': 517, 'LHR': 781, 'CDG': 595, 'AMS': 700, 'MAD': 720, 'BCN': 459, 'FCO': 0, 'DUB': 1020, 'VIE': 421, 'ZRH': 375}, 'DUB': {'FRA': 589, 'LHR': 243, 'CDG': 425, 'AMS': 406, 'MAD': 784, 'BCN': 802, 'FCO': 1020, 'DUB': 0, 'VIE': 922, 'ZRH': 670}, 'VIE': {'FRA': 336, 'LHR': 690, 'CDG': 561, 'AMS': 519, 'MAD': 977, 'BCN': 741, 'FCO': 421, 'DUB': 922, 'VIE': 0, 'ZRH': 327}, 'ZRH': {'FRA': 154, 'LHR': 427, 'CDG': 258, 'AMS': 326, 'MAD': 670, 'BCN': 463, 'FCO': 375, 'DUB': 670, 'VIE': 327, 'ZRH': 0}}

	# if not fixed_aircraft:
	# 	objective_function(x, fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands)

if __name__ == "__main__":
    main(sys.argv[1:])



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

