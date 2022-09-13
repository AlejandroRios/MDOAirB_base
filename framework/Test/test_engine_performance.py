
"""
MDOAirB

Description:
    - This test module performs parametric analysis of the
    engine model

TODO's:

| Authors: Alejandro Rios
           
  
| Email: aarc.88@gmail.com
| Creation: January 2022
| Last modification: 
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Performance.Engine.engine_performance import turbofan

import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# FUNCTIONS
# =============================================================================

# Aircraft initialization
vehicle = initialize_aircraft_parameters()

# Call engine parameters
engine = vehicle['engine']

engine['fan_pressure_ratio'] = 1.1
engine['compressor_pressure_ratio'] = 25
engine['bypass'] = 6
engine['fan_diameter'] = 2
engine['turbine_inlet_temperature'] = 1500

# Single ponint run
start_time = time.time()
engine_thrust, ff , vehicle = turbofan(10000, 0.5, 0.8, vehicle)

time_exe = (time.time() - start_time)
print(time_exe)
# ========