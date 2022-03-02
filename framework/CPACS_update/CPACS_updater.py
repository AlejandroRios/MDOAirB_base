"""
MDOAirB

Description:
    - This functions will update the CPACS file to consider the updates during the run.

Reference:
    -

TODO's:
    - To be finished

| Authors: Alejandro Rios
| Email: aarc.88@gmail.com
| Creation: January 2021
| Last modification: July 2021
| Language  : Python 3.8 or >
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import linecache
import subprocess
import numpy as np
from itertools import islice

from framework.Economics.crew_salary import crew_salary
from framework.CPACS_update.cpacsfunctions import *
import cpacsfunctions as cpsf

# =============================================================================
# FUNCTIONS
# =============================================================================
# CPACS XML input and output dir
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(MODULE_DIR)
cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'Aircraft_In.xml')
cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'Aircraft_Out.xml')

tixi = cpsf.open_tixi(cpacs_out_path)
tigl = cpsf.open_tigl(tixi)

# Reference parameters
Cref = tigl.wingGetMAC(tigl.wingGetUID(1))
Sref = tigl.wingGetReferenceArea(1, 1)
b    = tigl.wingGetSpan(tigl.wingGetUID(1))

# print(Sref)
xpath_write = '/cpacs/toolspecific/AVL/save_results/total_forces/'
model_xpath = '/cpacs/vehicles/aircraft/model/'
wing_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[1]/'

# Open and write cpacs xml output file
tixi_out = cpsf.open_tixi(cpacs_out_path)
tixi_out.updateDoubleElement(model_xpath+'reference/area', Sref, '%g')
tixi_out.updateDoubleElement(model_xpath+'reference/length', b, '%g')
tixi_out.updateDoubleElement(wing_xpath+'transformation/translation/x', 20, '%g')

# Close cpacs xml output file
tixi_out = cpsf.close_tixi(tixi_out, cpacs_out_path)