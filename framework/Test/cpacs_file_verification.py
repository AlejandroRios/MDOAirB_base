from framework.CPACS_update.cpacsfunctions import *
import numpy as np
import os

path = os.getcwd()
MODULE_DIR = path + '/CPACS_update'
cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'baseline.xml')
cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'baseline.xml')
tixi = open_tixi(cpacs_out_path)
tigl = open_tigl(tixi)

# tixi_out = open_tixi(cpacs_out_path)
# wing = vehicle['wing']
# horizontal_tail = vehicle['horizontal_tail']
# vertical_tail = vehicle['vertical_tail']
# fuselage = vehicle['fuselage']
# engine = vehicle['engine']
# nacelle = vehicle['nacelle']
# aircraft = vehicle['aircraft']


# Reference parameters
Cref = tigl.wingGetMAC(tigl.wingGetUID(1))
Sref = tigl.wingGetReferenceArea(1,1)
b    = tigl.wingGetSpan(tigl.wingGetUID(1))

print(Cref)
print(Sref*2)
print(b)