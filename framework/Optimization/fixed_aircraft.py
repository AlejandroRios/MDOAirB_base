"""
MDOAirB

Description:
    - This module runs the fixed aircraft case

TODO's:

| Authors: Lionel Guerin
           Alejandro Rios
  
| Email: aarc.88@gmail.com
| Creation: January 2021
| Last modification: July 2021
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""


import os
import json
import jsonschema
import linecache
from jsonschema import validate
from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Optimization.objective_function import objective_function
class AircraftInputDataError(Exception):
	def __init__(self, message):
		self.message = f"Aircraft input data issue: {message}"
		super().__init__(self.message)


AIRCRAFT_INPUT_DATA_SCHEMA = 'Database/JsonSchema/Aircraft_Input_Data.schema.json'
AIRCRAFT_INPUT_DATA_PATH = 'Database/Custom_Inputs/Aircraft_Input_Data.json'

def CheckDesignVariables(data):
	for key in data:
		if (data[key]["lower_band"] > data[key]["upper_band"]):
			raise DesignVariablesError(f'Lower band {data[key]["lower_band"]} is greater than upper band {data[key]["upper_band"]} for {key}')

def ReadJsonFile(schema_path, file_path, design_error_class, custom_check_function = None):
	try:

		with open(schema_path) as f:
			schema = json.load(f)

		with open(file_path) as f:
			data = json.load(f)

		validate(instance=data, schema=schema)

		if (custom_check_function != None):
			custom_check_function(data)
		
	except OSError as os_error:
		raise design_error_class(f"{os_error.strerror} [{os_error.filename}]")
	except json.JSONDecodeError as dec_error:
		raise design_error_class(f"{dec_error.msg} [line {dec_error.lineno} column {dec_error.colno} (char {dec_error.pos})]")
	except jsonschema.exceptions.SchemaError:
		raise design_error_class(f"There is an error with the schema")
	except jsonschema.exceptions.ValidationError as val_error:
		path_error = ""
		for path in val_error.path:
			if (path_error):
				path_error += "."
			path_error += path
		raise design_error_class(f"{val_error.message} in path [{path_error}]")

	return data

if os.path.isfile(AIRCRAFT_INPUT_DATA_PATH):
	aircraft_input_data = ReadJsonFile(AIRCRAFT_INPUT_DATA_SCHEMA, AIRCRAFT_INPUT_DATA_PATH,AircraftInputDataError)

def UpdateVehicle(vehicle, fixed_parameters):
    for key in fixed_parameters:
        if (key in vehicle):
            vehicle[key].update(fixed_parameters[key])
    return vehicle

vehicle = initialize_aircraft_parameters()
vehicle = UpdateVehicle(vehicle, aircraft_input_data)


profit = objective_function(vehicle, x=None)


