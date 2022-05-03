from framework.CPACS_update.cpacsfunctions import *
import numpy as np
def plot3d_tigl(vehicle):

    MODULE_DIR = 'c:/Users/aarc8/Documents/github\MDOAirB_base/framework/CPACS_update'
    cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'D150_v30.xml')
    cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'D150_v30.xml')
    tixi = open_tixi(cpacs_out_path)
    tigl = open_tigl(tixi)

    tixi_out = open_tixi(cpacs_out_path)
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']
    aircraft = vehicle['aircraft']




    return

import pickle

with open('Database/Family/40_to_100/all_dictionaries/'+str(1)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)


vehicle = all_info_acft1['vehicle']
plot3d_tigl(vehicle)