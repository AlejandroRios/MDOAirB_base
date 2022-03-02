"""
MDOAirB

Description:
    - This module configurate the genetic algorithm for the aircraft and
    network optimization. 

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
import multiprocessing


import json
import jsonschema
import linecache

import numpy as np
import os
import random
import subprocess
import getopt 
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Database.Airports.airports_database import AIRPORTS_DATABASE
from framework.Optimization.objective_function import objective_function
from framework.Attributes.Mission.mission_parameters import actual_mission_range

from haversine import haversine, Unit

from jsonschema import validate

from smt.sampling_methods import LHS
import sys
# =============================================================================
# CLASSES
# =============================================================================
class CustomInputsError(Exception):
	def __init__(self, message):
		self.message = f"Custom inputs issue: {message}"
		super().__init__(self.message)

# =============================================================================
# FUNCTIONS
# =============================================================================
CUSTOM_INPUTS_SCHEMA = 'Database/JsonSchema/Custom_Inputs.schema.json'

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

def check_demands(data, fixed_parameters, route_computation_mode):
	airport_keys = list(data.keys())
	for key in data:
		if (key in data[key]):
			raise CustomInputsError(f'Airport {key} exist on both departure and arrival for the same demand')
		airport_keys = airport_keys + list(set(data[key].keys()) - set(airport_keys))

	airports = check_airports(airport_keys)

	check_runways(data, airports)

	market_share = fixed_parameters['operations']['market_share']

	distances = {}

	if route_computation_mode == 0:
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
	else:
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
					distances[departure][arrival] = round(actual_mission_range(departure,arrival))


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

		airports, distances, demands = check_demands(data["demands"], fixed_parameters, route_computation_mode)

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
	
def register_variable(toolbox, lower_bounds, upper_bounds, design_variables, attr_name, key, min, max):
	if (key in design_variables):
		min = design_variables[key]["lower_band"]
		max = design_variables[key]["upper_band"]

	toolbox.register(attr_name, random.randint, min, max)
	lower_bounds.append(min)
	upper_bounds.append(max)

def register_variables(toolbox, design_variables):
	# Init of lower and upper bounds
	lower_bounds = []
	upper_bounds = []

	# Register variables
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_wing_surface", "wing_surface", 72, 130)  # [0] 
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_aspect_ratio", "aspect_ratio", 75, 120)  # [1] - real range 7.5 to 10
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_taper_ratio", "taper_ratio", 25, 50)  # [2] - real range 0.25 to 0.5
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_wing_sweep", "wing_sweep", 15, 30)  # [3]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_twist_angle", "twist_angle", -5, -2)  # [4]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_kink_position", "kink_position", 32, 45)  # [5] - real range 0.32 to 0.4
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_bypass_ratio", "engine_bypass_ratio", 45, 65)  # [6] - real range 4.5 to 6.5
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_fan_diameter", "engine_fan_diameter", 10, 25)  # [7] - real range 1 to 2
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_overall_pressure_ratio", "engine_overall_pressure_ratio", 27, 30)  # [8]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_inlet_turbine_temperature", "engine_inlet_turbine_temperature", 1350, 1500)  # [9] 1350 1500
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_fan_pressure_ratio", "engine_fan_pressure_ratio", 14, 25)  # [10] - real range 1.4 to 2.5
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_pax_number", "pax_number", 70, 220)  # [11]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_number_of_seat_abreast", "number_of_seat_abreast", 4, 6)  # [12]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_aircraft_range", "aircraft_range", 1000, 3500)  # [13]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_design_point_pressure", "engine_design_point_pressure", 39000, 43000)  # [14]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_design_point_mach", "engine_design_point_mach", 78, 82)  # [15] - real range 0.78 to 0.78
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_position", "engine_position", 1, 1)  # [16]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_winglet_presence", "winglet_presence", 1, 1)  # [17]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_slat_presense", "slat_presence", 1, 1)  # [18]
	register_variable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_horizontal_tail_position", "horizontal_tail_position", 1, 1)  # [19]

	toolbox.register("Individual", tools.initCycle, creator.Individual,
		(toolbox.attr_wing_surface, toolbox.attr_aspect_ratio, toolbox.attr_taper_ratio, toolbox.attr_wing_sweep, toolbox.attr_twist_angle, toolbox.attr_kink_position,
		toolbox.attr_engine_bypass_ratio, toolbox.attr_engine_fan_diameter, toolbox.attr_engine_overall_pressure_ratio, toolbox.attr_engine_inlet_turbine_temperature,
		toolbox.attr_engine_fan_pressure_ratio, toolbox.attr_pax_number, toolbox.attr_number_of_seat_abreast, toolbox.attr_aircraft_range, toolbox.attr_engine_design_point_pressure,
		toolbox.attr_engine_design_point_mach, toolbox.attr_engine_position, toolbox.attr_winglet_presence, toolbox.attr_slat_presense, toolbox.attr_horizontal_tail_position),
		n=1)

	return lower_bounds, upper_bounds

def init_population(pcls, ind_init, file):    
    return pcls(ind_init(c) for c in file)

def first_generation_create(individuas_number,lower_bounds,upper_bounds):
	xlimits = np.asarray(list(zip(lower_bounds, upper_bounds)))
	sampling = LHS(xlimits=xlimits)
	Initial_population = sampling(individuas_number)
	Initial_population =  [[round(y) for y in x] for x in Initial_population]

	return Initial_population

def process_optimized_aircraft(design_variables, original_vehicle, computation_mode, route_computation_mode, airports, distances, demands):
	# Declare the kind of optimization (min or max)
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	# Define the individual list
	creator.create("Individual", list, fitness=creator.FitnessMax)

	# Definition of all the atributes (design variables), their type and range
	toolbox = base.Toolbox()
	lower_bounds, upper_bounds = register_variables(toolbox, design_variables)

	individuas_number = 10
	# original_Population = first_generation_create(individuas_number,lower_bounds,upper_bounds)
	original_population = [[79, 89, 39, 25, -4, 26, 62, 18, 27, 1418, 15, 74, 5, 1225, 41000, 78, 1, 1, 1, 1], [121, 79, 26, 26, -3, 36, 64, 12, 28, 1448, 16, 116, 6, 1375, 41000, 78, 1, 1, 1, 1], [109, 84, 34, 29, -3, 38, 54, 16, 29, 1372, 20, 60, 4, 2425, 41000, 78, 1, 1, 1, 1], [127, 99, 46, 19, -4, 27, 46, 20, 30, 1402, 17, 102, 6, 1075, 41000, 78, 1, 1, 1, 1], [85, 96, 36, 16, -5, 29, 50, 12, 30, 1462, 17, 68, 5, 1975, 41000, 78, 1, 1, 1, 1], [115, 91, 29, 20, -2, 39, 48, 14, 27, 1492, 19, 110, 4, 1525, 41000, 78, 1, 1, 1, 1], [91, 86, 31, 17, -2, 32, 58, 14, 28, 1432, 16, 54, 4, 2125, 41000, 78, 1, 1, 1, 1], [73, 94, 44, 22, -5, 33, 60, 18, 28, 1478, 18, 88, 5, 1825, 41000, 78, 1, 1, 1, 1], [103, 81, 41, 28, -4, 35, 56, 16, 29, 1388, 14, 82, 5, 2275, 41000, 78, 1, 1, 1, 1]]

	# Genetic algoritgm configuration
	toolbox.register("evaluate", objective_function, original_vehicle=original_vehicle, computation_mode=computation_mode, route_computation_mode=route_computation_mode, airports=airports, distances=distances, demands=demands)
	toolbox.register('mate', tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutUniformInt,low =lower_bounds,up=upper_bounds,indpb=0.2)
	toolbox.register("select", tools.selNSGA2)
	toolbox.register("population_guess", init_population, list, creator.Individual,original_population)

	with multiprocessing.get_context('spawn').Pool(processes=1) as pool:
		# pool = multiprocessing.Pool(processes=6)
		# toolbox.register("map", pool.map)

		# pop = toolbox.population(n=10)
		pop = toolbox.population_guess()
		hof = tools.HallOfFame(2)
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		logbooks = list()

		pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=10, stats=stats, halloffame=hof)

		pool.close()

	# Save results to txt files
	with open("Database/Results/Optimization/optim_statistics.txt", "w") as file:
		file.write(str(log))

	with open("Database/Results/Optimization/optim_population.txt", "w") as file:
		file.write(str(pop))

	with open("Database/Results/Optimization/optim_hall_of_fame.txt", "w") as file:
		file.write(str(hof))

	best = hof.items[0]

	print("Best Solution = ", best)
	print("Best Score = ", best.fitness.values[0])

	print('# =============================================================================######################')
	print('Score:', best.fitness.values[0])
	print('# =============================================================================######################')

def process_fixed_aircraft(fixed_aircraft, computation_mode, route_computation_mode, airports, distances, demands):
	print("Fixed aircraft processing")

# =============================================================================
# MAIN
# =============================================================================
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
	design_variables = {}
	fixed_parameters = {}
	fixed_aircraft = {}
	customInputsfile = readArgv(argv)
	if not customInputsfile or not os.path.isfile(customInputsfile):
		print(f"Custom file {customInputsfile} does not exist")
		sys.exit(1)

	try:
		computation_mode, route_computation_mode, airports, distances, demands, design_variables, fixed_parameters, fixed_aircraft = read_custom_inputs(CUSTOM_INPUTS_SCHEMA, customInputsfile)
	except Exception as err:
		print(f"Exception ocurred while playing custom inputs file {customInputsfile}")
		print(f"Error: {err}")
		sys.exit(1)

	if not fixed_aircraft:
		process_optimized_aircraft(design_variables, fixed_parameters, computation_mode, route_computation_mode, airports, distances, demands)
	else:
		process_fixed_aircraft(fixed_aircraft, computation_mode, route_computation_mode, airports, distances, demands)

if __name__ == "__main__":
    main(sys.argv[1:])


# =============================================================================
# TEST
# =============================================================================
