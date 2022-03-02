"""
File name : Optimization function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : Dezember 2020
Last edit : February 2021
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    - This function configurate the genetic algorithm for the aircraft and
    network optimization. 
Inputs:
    -
Outputs:
    -
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
import random
import numpy as np
import os
import subprocess
import linecache
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from framework.Optimization.objective_function import objective_function
from framework.Database.Aircrafts.baseline_aircraft_parameters import *
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
# p = figure()

# Declare the kind of optimization (min or max)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define the individual list
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 20  # Define the number of optimization variables

# Definition of all the atributes (design variables), their type and range
toolbox = base.Toolbox()
toolbox.register("attr_wing_surface", random.randint, 72, 130)  # [0] 
toolbox.register("attr_aspect_ratio", random.randint, 75, 100)  # [1] - real range 7.5 to 10
toolbox.register("attr_taper_ratio", random.randint, 25, 50)  # [2] - real range 0.25 to 0.5
toolbox.register("attr_wing_sweep", random.randint, 15, 35)  # [3]
toolbox.register("attr_twist_angle", random.randint, -5, -2)  # [4]
toolbox.register("attr_kink_position", random.randint, 32, 40)  # [5] - real range 0.32 to 0.4
toolbox.register("attr_engine_bypass_ratio", random.randint, 45, 65)  # [6] - real range 4.5 to 6.5
toolbox.register("attr_engine_fan_diameter", random.randint, 10, 20)  # [7] - real range 1 to 2
toolbox.register("attr_engine_overall_pressure_ratio", random.randint, 27, 30)  # [8]
toolbox.register("attr_engine_inlet_turbine_temperature",
                 random.randint, 1350, 1500)  # [9]
toolbox.register("attr_engine_fan_pressure_ratio", random.randint, 14, 25)  # [10] - real range 1.4 to 2.5
toolbox.register("attr_pax_number", random.randint, 50, 120)  # [11]
toolbox.register("attr_number_of_seat_abreast", random.randint, 4, 6)  # [12]
toolbox.register("attr_aircraft_range", random.randint, 1000, 2500)  # [13]
toolbox.register("attr_engine_design_point_pressure",
                 random.randint, 41000, 41000)  # [14]
toolbox.register("attr_engine_design_point_mach", random.randint, 78, 78)  # [15] - real range 0.78 to 0.78
toolbox.register("attr_engine_position", random.randint, 1, 1)  # [16]
toolbox.register("attr_winglet_presence", random.randint, 1, 1)  # [17]
toolbox.register("attr_slat_presense", random.randint, 1, 1)  # [18]
toolbox.register("attr_horizontal_tail_position", random.randint, 1, 1)  # [19]

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_wing_surface, toolbox.attr_aspect_ratio, toolbox.attr_taper_ratio, toolbox.attr_wing_sweep, toolbox.attr_twist_angle, toolbox.attr_kink_position,
                  toolbox.attr_engine_bypass_ratio, toolbox.attr_engine_fan_diameter, toolbox.attr_engine_overall_pressure_ratio, toolbox.attr_engine_inlet_turbine_temperature,
                  toolbox.attr_engine_fan_pressure_ratio, toolbox.attr_pax_number, toolbox.attr_number_of_seat_abreast, toolbox.attr_aircraft_range, toolbox.attr_engine_design_point_pressure,
                  toolbox.attr_engine_design_point_mach, toolbox.attr_engine_position, toolbox.attr_winglet_presence, toolbox.attr_slat_presense, toolbox.attr_horizontal_tail_position),
                 n=1)


# Declaration of the objective function (network profit)


def obj_function(individual,vehicle):
    # This function takes as inputs the current individual (vector of design variavbles) and
    # a predefined dictionary with pre-stored information of the vehicle (aircraft)
    net_profit = objective_function(individual, vehicle)
    return [net_profit, ]

# Declarate the limits for feasible individuals
def initPopulation(pcls, ind_init, file):    
    return pcls(ind_init(c) for c in file)


def feaseGeom(x):
    wing_surface = x[0]
    aspect_ratio = x[1]
    taper_ratio = x[2]
    wing_sweep = x[3]
    twist_angle = x[4]
    kink_position = x[5]
    engine_bypass_ratio = x[6]
    engine_fan_diameter = x[7]
    engine_overall_pressure_ratio = x[8]
    engine_inlet_turbine_temperature = x[9]
    engine_fan_pressure_ratio = x[10]
    pax_number = x[11]
    number_of_seat_abreast = x[12]
    aircraft_range = x[13]
    engine_design_point_pressure = x[14]
    engine_design_point_mach = x[15]
    engine_position = x[16]
    winglet_presence = x[17]
    slat_precense = x[18]
    horizontal_tail_position = x[19]

    if ((wing_surface >= 72 and wing_surface <= 130) and (aspect_ratio >= 75 and aspect_ratio <= 100) and (taper_ratio >= 25 and taper_ratio <= 50) and (wing_sweep >= 15 and wing_sweep <= 35) and (twist_angle >= -5 and twist_angle <= -2) and
        (kink_position >= 32 and kink_position <= 40) and (engine_bypass_ratio >= 45 and engine_bypass_ratio <= 65) and (engine_fan_diameter >= 10 and engine_fan_diameter <= 20) and (engine_overall_pressure_ratio >= 25 and engine_overall_pressure_ratio <= 30) and
        (engine_inlet_turbine_temperature >= 1350 and engine_inlet_turbine_temperature <= 1500) and (engine_fan_pressure_ratio >= 14 and engine_fan_pressure_ratio <= 25) and (pax_number >= 70 and pax_number <= 120) and (number_of_seat_abreast >= 4 and number_of_seat_abreast <= 6) and
        (aircraft_range >= 1000 and aircraft_range <= 2500) and (engine_design_point_pressure >= 41000 and engine_design_point_pressure <= 41000) and (engine_design_point_mach >= 78 and engine_design_point_mach <= 78) and (engine_position >= 1 and engine_position <= 1) and (winglet_presence >= 1 and winglet_presence <= 1) and
            (slat_precense >= 1 and slat_precense <= 1) and (horizontal_tail_position >= 1 and horizontal_tail_position <= 1)):

        return True
    return False

Initial_population = [[72, 91, 30, 33, -4, 31, 45, 13, 28, 1382, 17, 88, 5, 1885, 41000, 78, 1, 1, 1, 1], [73, 94, 42, 16, -4, 34, 63, 14, 28, 1384, 18, 111, 5, 2425, 41000, 78, 1, 1, 1, 1], [98, 78, 31, 16, -4, 40, 61, 20, 28, 1418, 17, 98, 4, 2005, 41000, 78, 1, 1, 1, 1], [90, 90, 49, 33, -2, 34, 48, 16, 29, 1394, 16, 105, 4, 1285, 41000, 78, 1, 1, 1, 1], [75, 92, 41, 23, -4, 30, 53, 18, 29, 1460, 17, 51, 5, 1315, 41000, 78, 1, 1, 1, 1], 
[80, 85, 37, 34, -3, 27, 52, 14, 28, 1498, 17, 87, 6, 1615, 41000, 78, 1, 1, 1, 1], [127, 89, 42, 20, -3, 27, 58, 14, 28, 1466, 18, 107, 6, 1735, 41000, 78, 1, 1, 1, 1], [123, 87, 31, 32, -3, 32, 60, 15, 30, 1442, 17, 108, 5, 1555, 41000, 78, 1, 1, 1, 1], [121, 98, 45, 24, -4, 38, 
51, 10, 27, 1414, 19, 72, 5, 1525, 41000, 78, 1, 1, 1, 1], [116, 75, 34, 16, -5, 29, 48, 16, 28, 1478, 19, 70, 5, 1945, 41000, 78, 1, 1, 1, 1], [102, 78, 39, 17, -4, 36, 47, 16, 27, 1364, 19, 77, 5, 1465, 41000, 78, 1, 1, 1, 1], [83, 90, 33, 20, -2, 33, 50, 10, 29, 1448, 19, 62, 4, 1225, 41000, 78, 1, 1, 1, 1], [84, 95, 39, 28, -3, 25, 60, 18, 27, 1444, 18, 101, 4, 2275, 41000, 78, 1, 1, 1, 1], [126, 99, 49, 28, -3, 25, 63, 20, 29, 1366, 16, 104, 6, 1435, 41000, 78, 1, 1, 1, 1], [114, 92, 30, 26, -3, 37, 52, 16, 27, 1400, 15, 102, 5, 2395, 41000, 78, 1, 1, 1, 1], [129, 95, 47, 25, -4, 39, 65, 20, 29, 1484, 15, 79, 4, 1045, 41000, 78, 1, 1, 1, 1], [77, 88, 40, 15, -3, 28, 50, 13, 29, 1480, 15, 73, 5, 1375, 41000, 78, 1, 1, 1, 1], [117, 77, 43, 22, -3, 37, 61, 15, 29, 1358, 20, 68, 5, 1645, 41000, 78, 1, 1, 1, 1], [125, 93, 27, 35, -3, 
30, 56, 11, 28, 1390, 18, 69, 5, 1495, 41000, 78, 1, 1, 1, 1], [119, 96, 36, 19, -3, 35, 49, 12, 28, 1402, 20, 90, 6, 1105, 41000, 78, 1, 1, 
1, 1], [115, 93, 27, 29, -4, 40, 59, 17, 29, 1492, 19, 82, 5, 2455, 41000, 78, 1, 1, 1, 1], [89, 80, 28, 21, -3, 36, 64, 19, 29, 1486, 20, 76, 5, 2155, 41000, 78, 1, 1, 1, 1], [85, 83, 46, 19, -5, 39, 60, 10, 29, 1450, 18, 116, 5, 2485, 41000, 78, 1, 1, 1, 1], [93, 94, 29, 32, -2, 
32, 53, 13, 30, 1490, 17, 115, 5, 2095, 41000, 78, 1, 1, 1, 1], [74, 91, 34, 24, -4, 33, 64, 12, 27, 1454, 20, 55, 4, 1345, 41000, 78, 1, 1, 
1, 1], [113, 97, 33, 17, -3, 37, 54, 11, 30, 1436, 16, 63, 6, 1915, 41000, 78, 1, 1, 1, 1], [122, 77, 44, 27, -4, 38, 46, 14, 28, 1412, 15, 84, 4, 2065, 41000, 78, 1, 1, 1, 1], [111, 86, 29, 18, -3, 39, 58, 12, 28, 1406, 16, 114, 4, 1675, 41000, 78, 1, 1, 1, 1], [96, 81, 43, 30, -4, 31, 46, 12, 28, 1456, 15, 74, 5, 1855, 41000, 78, 1, 1, 1, 1], [79, 100, 26, 31, -4, 31, 55, 14, 27, 1426, 16, 93, 4, 1075, 41000, 78, 1, 1, 1, 1], [110, 99, 38, 30, -4, 28, 59, 17, 28, 1432, 14, 118, 6, 1015, 41000, 78, 1, 1, 1, 1], [103, 82, 44, 22, -2, 28, 54, 18, 27, 1438, 16, 96, 6, 1405, 41000, 78, 1, 1, 1, 1], [99, 98, 35, 34, -5, 34, 57, 19, 29, 1420, 17, 110, 6, 2305, 41000, 78, 1, 1, 1, 1], [97, 82, 47, 26, 
-5, 35, 49, 19, 29, 1430, 18, 80, 5, 1165, 41000, 78, 1, 1, 1, 1], [109, 81, 37, 21, -3, 27, 62, 11, 30, 1462, 15, 86, 5, 2185, 41000, 78, 1, 1, 1, 1], [81, 88, 48, 28, -5, 35, 57, 17, 29, 1408, 15, 66, 6, 2215, 41000, 78, 1, 1, 1, 1], [71, 87, 48, 30, -2, 26, 46, 18, 29, 1496, 15, 119, 4, 1255, 41000, 78, 1, 1, 1, 1], [95, 89, 32, 20, -5, 34, 56, 16, 29, 1378, 17, 58, 4, 1135, 41000, 78, 1, 1, 1, 1], [108, 85, 45, 31, 
-2, 29, 48, 18, 28, 1472, 18, 52, 5, 2245, 41000, 78, 1, 1, 1, 1], [107, 76, 32, 23, -5, 31, 52, 11, 30, 1396, 19, 59, 5, 1795, 41000, 78, 1, 1, 1, 1], [128, 79, 46, 18, -4, 37, 47, 12, 28, 1372, 18, 91, 6, 1585, 41000, 78, 1, 1, 1, 1], [105, 80, 25, 25, -4, 30, 58, 13, 28, 1388, 19, 54, 5, 2125, 41000, 78, 1, 1, 1, 1], [92, 83, 41, 32, -4, 38, 55, 12, 29, 1354, 14, 112, 4, 1975, 41000, 78, 1, 1, 1, 1], [78, 86, 26, 18, -4, 33, 62, 18, 30, 1468, 16, 83, 4, 2035, 41000, 78, 1, 1, 1, 1], [104, 84, 35, 29, -2, 28, 64, 15, 27, 1474, 14, 56, 5, 1825, 41000, 78, 1, 1, 1, 1], [101, 97, 38, 34, -5, 26, 56, 14, 28, 1360, 14, 94, 5, 2365, 41000, 78, 1, 1, 1, 1], [86, 76, 28, 22, -3, 36, 51, 17, 30, 1376, 18, 65, 5, 1195, 41000, 78, 1, 1, 1, 1], [91, 84, 40, 24, -3, 32, 62, 16, 29, 1424, 16, 100, 6, 2335, 41000, 78, 1, 1, 1, 1], [120, 79, 50, 27, -2, 26, 50, 19, 28, 1352, 16, 60, 6, 1765, 41000, 78, 1, 1, 1, 1], [87, 96, 36, 26, -3, 29, 54, 15, 30, 1370, 19, 97, 6, 1705, 41000, 78, 1, 1, 1, 1]]



# Genetic algoritgm configuration
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", obj_function,vehicle = vehicle)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("population_guess", initPopulation, list, creator.Individual, Initial_population)



# toolbox.register("evaluate", obj_function)
# toolbox.decorate("evaluate", tools.DeltaPenalty(feaseGeom, [1.0, ]))

# The main function that defines the genetic algorithm run


if __name__ == '__main__':

	# random.seed(62)
	# toolbox.decorate("evaluate", tools.DeltaPenalty(feaseGeom, [1.0, ]))

	# Process Pool of 4 workers
	# pool = multiprocessing.Pool(processes=8)
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

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=2, stats=stats, halloffame=hof)

	# pool.close()

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

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
