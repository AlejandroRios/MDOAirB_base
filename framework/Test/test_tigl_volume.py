
import threading
from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Performance.Engine.engine_performance import turbofan

import numpy as np
import matplotlib.pyplot as plt
vehicle = initialize_aircraft_parameters()


from framework.CPACS_update.cpacsfunctions import *
import numpy as np
from framework.utilities.plot_tigl_aircraft import plot3d_tigl

plot3d_tigl(vehicle)