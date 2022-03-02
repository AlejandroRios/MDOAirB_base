"""
Function  :main.py
Title     : main function
Written by: 
Email     : aarc.88@gmail.com
Date      : 
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
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
from framework.Performance.Mission.mission import mission
from framework.Network.network_optimization import network_optimization
from framework.Economics.revenue import revenue

import pandas as pd
import pickle
import numpy as np
from datetime import datetime


# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================

start_time = datetime.now()


global GRAVITY
GRAVITY = 9.80665
gallon_to_liter = 3.7852
feet_to_nautical_miles = 0.000164579

aircraft_data = baseline_aircraft()

market_share = 0.1

df1 = pd.read_csv('Database/distance.csv')
df1 = (df1.T)
# print(df1)
distances = df1.to_dict()

df2 = pd.read_csv('Database/demand.csv')
df2 = round(market_share*(df2.T))
# print(df2)
demand = df2.to_dict()
# print(type(demand))
# df3 = pd.read_csv('Database/doc.csv')
# df3 = (df3.T)
# print(df3)
# doc = df3.to_dict()

pax_capacity = aircraft_data['passenger_capacity']
# rev_mat = revenue(aircraft_data, distances)

departures = ['CD1', 'CD2', 'CD3', 'CD4',
              'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']
arrivals = ['CD1', 'CD2', 'CD3', 'CD4',
            'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']

pax_number = 78
load_factor = pax_number/pax_capacity
revenue_ik = {}
for i in departures:
    for k in arrivals:
        if i != k:
            # revenue_ik[(i, k)] = revenue(distances[i][k], load_factor, pax_capacity, pax_number)
            revenue_ik[(i, k)] = revenue(
                demand[i][k], distances[i][k], pax_capacity, pax_number)
        else:
            revenue_ik[(i, k)] = 0

# print(revenue_ik)

DOC_ik = {}

for i in departures:
    for k in arrivals:
        if i != k:
            DOC_ik[(i, k)] = float(mission(distances[i][k]) * distances[i][k])
            print(DOC_ik[(i, k)])
        else:
            DOC_ik[(i, k)] = 0

# df = pd.DataFrame(data=DOC_ik)
# df = (df.T)
# DOC_ik = np.load('my_file.npy', allow_pickle=True)
# DOC_ik = DOC_ik.item()
# np.save('my_file.npy', DOC_ik)
# print('==================================================================')
# print(DOC_ik)
profit = network_optimization(distances, demand, DOC_ik)
print(profit)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
# =============================================================================
# TEST
# =============================================================================
