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

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure

from framework.Database.Aircrafts.baseline_aircraft_parameters import *
from framework.Optimization.aircraft_family_DD_multiobjective import objective_function

from jsonschema import validate

from smt.sampling_methods import LHS
# =============================================================================
# CLASSES
# =============================================================================

class DesignVariablesError(Exception):
	def __init__(self, message):
		self.message = f"Read Design variables issue: {message}"
		super().__init__(self.message)

class FixedParametersError(Exception):
	def __init__(self, message):
		self.message = f"Fixed parameters issue: {message}"
		super().__init__(self.message)

# =============================================================================
# FUNCTIONS
# =============================================================================
DESIGN_VARIABLES_SCHEMA = 'Database/JsonSchema/Design_Variables_Limits.schema.json'
DESIGN_VARIABLES_PATH = 'Database/Custom_Inputs/Design_Variables_Limits.json'

FIXED_PARAMETERS_SCHEMA = 'Database/JsonSchema/Fixed_Parameters.schema.json'
FIXED_PARAMETERS_PATH = 'Database/Custom_Inputs/Fixed_Parameters.json'


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

def UpdateVehicle(vehicle, fixed_parameters):
    for key in fixed_parameters:
        if (key in vehicle):
            vehicle[key].update(fixed_parameters[key])
    return vehicle
	
def RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, attr_name, key, min, max):
	if (key in design_variables):
		min = design_variables[key]["lower_band"]
		max = design_variables[key]["upper_band"]

	toolbox.register(attr_name, random.randint, min, max)
	lower_bounds.append(min)
	upper_bounds.append(max)

def RegisterVariables(toolbox, design_variables):
	# Init of lower and upper bounds
	lower_bounds = []
	upper_bounds = []

	# Register variables
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_wing_surface", "wing_surface", 72, 130)  # [0] 
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_aspect_ratio", "aspect_ratio", 75, 120)  # [1] - real range 7.5 to 10
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_taper_ratio", "taper_ratio", 25, 50)  # [2] - real range 0.25 to 0.5
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_wing_sweep", "wing_sweep", 15, 30)  # [3]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_twist_angle", "twist_angle", -5, -2)  # [4]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_kink_position", "kink_position", 32, 45)  # [5] - real range 0.32 to 0.4
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_bypass_ratio", "engine_bypass_ratio", 45, 65)  # [6] - real range 4.5 to 6.5
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_fan_diameter", "engine_fan_diameter", 10, 25)  # [7] - real range 1 to 2
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_overall_pressure_ratio", "engine_overall_pressure_ratio", 27, 30)  # [8]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_inlet_turbine_temperature", "engine_inlet_turbine_temperature", 1350, 1500)  # [9] 1350 1500
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_fan_pressure_ratio", "engine_fan_pressure_ratio", 14, 25)  # [10] - real range 1.4 to 2.5
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_pax_number", "pax_number", 70, 220)  # [11]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_number_of_seat_abreast", "number_of_seat_abreast", 4, 6)  # [12]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_aircraft_range", "aircraft_range", 1000, 3500)  # [13]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_design_point_pressure", "engine_design_point_pressure", 39000, 43000)  # [14]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_design_point_mach", "engine_design_point_mach", 78, 82)  # [15] - real range 0.78 to 0.78
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_engine_position", "engine_position", 1, 1)  # [16]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_winglet_presence", "winglet_presence", 1, 1)  # [17]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_slat_presense", "slat_presence", 1, 1)  # [18]
	RegisterVariable(toolbox, lower_bounds, upper_bounds, design_variables, "attr_horizontal_tail_position", "horizontal_tail_position", 1, 1)  # [19]

	toolbox.register("Individual", tools.initCycle, creator.Individual,
		(toolbox.attr_wing_surface, toolbox.attr_aspect_ratio, toolbox.attr_taper_ratio, toolbox.attr_wing_sweep, toolbox.attr_twist_angle, toolbox.attr_kink_position,
		toolbox.attr_engine_bypass_ratio, toolbox.attr_engine_fan_diameter, toolbox.attr_engine_overall_pressure_ratio, toolbox.attr_engine_inlet_turbine_temperature,
		toolbox.attr_engine_fan_pressure_ratio, toolbox.attr_pax_number, toolbox.attr_number_of_seat_abreast, toolbox.attr_aircraft_range, toolbox.attr_engine_design_point_pressure,
		toolbox.attr_engine_design_point_mach, toolbox.attr_engine_position, toolbox.attr_winglet_presence, toolbox.attr_slat_presense, toolbox.attr_horizontal_tail_position),
		n=1)

	return lower_bounds, upper_bounds

# # Declaration of the objective function (network profit)
def obj_function(individual):
    '''
    Description:
        - This function takes as inputs the current individual (vector of design variables) and
          a predefined dictionary with pre-stored information of the vehicle (aircraft)

    Inputs:
        - individual - array containing the design variables of the individual to be analysed
    Outputs:
        - network profit
    '''
    vehicle = initialize_aircraft_parameters()
    vehicle = UpdateVehicle(vehicle, fixed_parameters)
    net_profit,SAR= objective_function(vehicle,individual)
    vehicle.clear()
    return -net_profit, SAR

def initPopulation(pcls, ind_init, file):
    return pcls(ind_init(c) for c in file)

def first_generation_create(individuas_number,lower_bounds,upper_bounds):
    '''
    Description:
        - This function create the first generation to be analysed via latin-hypercube sampling
    Inputs:
        - individual_number - number of individual to compose the first generation
		- lower_bounds - design variables lower bounds
		- upper_bounds - design varaibles upper bounds
    Outputs:
        - Initial_population - array containning the initial population individuals
    '''
    xlimits = np.asarray(list(zip(lower_bounds, upper_bounds)))
    sampling = LHS(xlimits=xlimits)
    Initial_population = sampling(individuas_number)
    Initial_population =  [[round(y) for y in x] for x in Initial_population]

    return Initial_population

# Declare the kind of optimization (min or max)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define the individual list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Get custom design variables
design_variables = {}
if os.path.isfile(DESIGN_VARIABLES_PATH):
	design_variables = ReadJsonFile(DESIGN_VARIABLES_SCHEMA, DESIGN_VARIABLES_PATH, DesignVariablesError, CheckDesignVariables)

# Definition of all the atributes (design variables), their type and range
toolbox = base.Toolbox()
lower_bounds, upper_bounds = RegisterVariables(toolbox, design_variables)

# Get fixed parameters
fixed_parameters = {}
if os.path.isfile(FIXED_PARAMETERS_PATH):
	fixed_parameters = ReadJsonFile(FIXED_PARAMETERS_SCHEMA, FIXED_PARAMETERS_PATH, FixedParametersError)
	
# Update vehicle with fixed parameters
# UpdateVehicle(vehicle, fixed_parameters)
individuas_number = 10
# init_Population = first_generation_create(individuas_number,lower_bounds,upper_bounds)

init_Population = [[97, 76, 49, 23, -3, 30, 52, 10, 29, 1358, 18, 96, 6, 1675, 41000, 78, 1, 1, 1, 1], 
[79, 89, 39, 25, -4, 26, 62, 18, 27, 1418, 15, 74, 5, 1225, 41000, 78, 1, 1, 1, 1], 
[121, 79, 26, 26, -3, 36, 64, 12, 28, 1448, 16, 116, 6, 1375, 41000, 78, 1, 1, 1, 1], 
[109, 84, 34, 29, -3, 38, 54, 16, 29, 1372, 20, 60, 4, 2425, 41000, 78, 1, 1, 1, 1], 
[127, 99, 46, 19, -4, 27, 46, 20, 30, 1402, 17, 102, 6, 1075, 41000, 78, 1, 1, 1, 1], 
[85, 96, 36, 16, -5, 29, 50, 12, 30, 1462, 17, 68, 5, 1975, 41000, 78, 1, 1, 1, 1], 
[115, 91, 29, 20, -2, 39, 48, 14, 27, 1492, 19, 110, 4, 1525, 41000, 78, 1, 1, 1, 1], 
[91, 86, 31, 17, -2, 32, 58, 14, 28, 1432, 16, 54, 4, 2125, 41000, 78, 1, 1, 1, 1], 
[73, 94, 44, 22, -5, 33, 60, 18, 28, 1478, 18, 88, 5, 1825, 41000, 78, 1, 1, 1, 1], 
[103, 81, 41, 28, -4, 35, 56, 16, 29, 1388, 14, 82, 5, 2275, 41000, 78, 1, 1, 1, 1]]

# Genetic algoritgm configuration
toolbox.register("evaluate", obj_function)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt,low =lower_bounds,up=upper_bounds,indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("population_guess", initPopulation, list, creator.Individual,init_Population)
# =============================================================================
# MAIN
# =============================================================================
# if __name__ == '__main__':

# 	with multiprocessing.get_context('spawn').Pool(processes=6) as pool:
# 		# pool = multiprocessing.Pool(processes=6)
# 		toolbox.register("map", pool.map)

# 	# pop = toolbox.population(n=10)
# 		pop = toolbox.population_guess()
# 		hof = tools.HallOfFame(2)
# 		stats = tools.Statistics(lambda ind: ind.fitness.values)
# 		stats.register("avg", np.mean)
# 		stats.register("std", np.std)
# 		stats.register("min", np.min)
# 		stats.register("max", np.max)

# 		logbooks = list()

# 		pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=30, stats=stats, halloffame=hof)

# 		pool.close()

# 	# Save results to txt files
# 	with open("Database/Results/Optimization/optim_statistics.txt", "w") as file:
# 		file.write(str(log))

# 	with open("Database/Results/Optimization/optim_population.txt", "w") as file:
# 		file.write(str(pop))

# 	with open("Database/Results/Optimization/optim_hall_of_fame.txt", "w") as file:
# 		file.write(str(hof))

# 	best = hof.items[0]

# 	print("Best Solution = ", best)
# 	print("Best Score = ", best.fitness.values[0])

# 	print('# =============================================================================######################')
# 	print('Score:', best.fitness.values[0])
# 	print('# =============================================================================######################')

# =============================================================================
# TEST
# =============================================================================


from scipy.optimize import  differential_evolution
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state, MapWrapper
import matplotlib.pyplot as plt
import scipy.optimize

import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import starmap_parallelized_eval
from pymoo.problems.functional import FunctionalProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.misc import stack
from pymoo.factory import get_termination
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric
from pymoo.util.running_metric import RunningMetric

import pandas as pd
from pymoo.core.population import Population


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=3,
                        n_obj=2,
                        n_constr=0,
                        xl=np.array([1,1, 1]),
                        xu=np.array([37,29, 68]),
                        type_var=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1,f2 = obj_function(x)
        out["F"] = [f1, f2]
if __name__ == "__main__":


    mask = ["int", "int", "int"]

    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0)
    })

    n_proccess = 14

    pool = multiprocessing.Pool(n_proccess)
    problem = MyProblem(runner=pool.starmap, func_eval=starmap_parallelized_eval)
    # problem = FunctionalProblem(n_var,
    #                         obj_function,
    #                         xl=np.array([1,1,1]),
    #                         xu=np.array([38,  30,  34]),
    #                         type_var=int,
    #                         runner=pool.starmap, func_eval=starmap_parallelized_eval)

    

    algorithm = NSGA2(pop_size=42,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        eliminate_duplicates=True,
                        )



    res = minimize(problem,algorithm,('n_gen', 50), verbose=True,save_history=True,seed=1)
    print('Processes:', res.exec_time)
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)


    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    pool.close()


    X = res.X
    F = res.F
    xl, xu = problem.bounds()

    all_pop = Population()

    for algorithm in res.history:
        all_pop = Population.merge(all_pop, algorithm.off)

    df = pd.DataFrame(all_pop.get("F"), columns=[f"X{i+1}" for i in range(problem.n_obj)])
    df2 = pd.DataFrame(all_pop.get("X"), columns=[f"X{i+1}" for i in range(problem.n_var)])

    df.to_pickle("functions_opt.pkl") 
    df2.to_pickle("vars_opt.pkl") 

    np.save('results', all_pop)


    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')


    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Profit [US$]')
    ax.set_ylabel('CO2 efficiecy [kg/nm]')

    ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none',marker= '^',edgecolors='blue')
    ax.scatter(df['X1'], df['X2'],s=5, facecolors='none', edgecolors='red')
    ax.set_title("Objective Space")

    plt.savefig("pareto.pdf", format="pdf", bbox_inches="tight")
    plt.show()


    # pf_a, pf_b = problem.pareto_front(use_cache=False, flatten=False)

    # pf = problem.pareto_front(use_cache=False, flatten=True)

    # fig = plt.figure(figsize=(10, 9))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='b', label="Solutions")
    # ax.plot(pf_a[:, 0], pf_a[:, 1], alpha=0.5, linewidth=2.0, color="red", label="Pareto-front")
    # ax.plot(pf_b[:, 0], pf_b[:, 1], alpha=0.5, linewidth=2.0, color="red")
    # ax.set_title("Objective Space")
    # plt.legend()
    # plt.show()



    X, F = res.opt.get("X", "F")

    hist = res.history
    print(len(hist))
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population

    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    k = np.where(np.array(hist_cv) <= 0.0)[0].min()
    print(f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")

    # replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
    vals = hist_cv_avg

    k = np.where(np.array(vals) <= 0.0)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n_evals, vals,  color='black', lw=0.7, label="Avg. CV of Pop")
    ax.scatter(n_evals, vals,  facecolor="none", edgecolor='black', marker="p")
    ax.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
    ax.set_title("Convergence")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Constraint Violation")
    ax.legend()
    plt.show()

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    from pymoo.indicators.hv import Hypervolume

    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                        norm_ref_point=False,
                        zero_to_one=True,
                        ideal=approx_ideal,
                        nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    ax.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    ax.set_title("Convergence")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Hypervolume")
    plt.savefig("hypervolume.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    

    running = RunningMetric(delta_gen=5,
                            n_plots=3,
                            only_if_n_plots=True,
                            key_press=False,
                            do_show=True)

    for algorithm in res.history[:15]:
        running.notify(algorithm)

    running = RunningMetric(delta_gen=10,
                            n_plots=4,
                            only_if_n_plots=True,
                            key_press=False,
                            do_show=True)

    for algorithm in res.history:
        running.notify(algorithm)

    
    plt.savefig("seila.pdf", format="pdf", bbox_inches="tight")

    plt.show()

