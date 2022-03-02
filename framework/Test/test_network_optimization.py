"""
File name : Network optimization
Author    : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : March/2020
Last edit : November/2020
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
"""Importing Modules"""
# =============================================================================


# import cplex
# import pulp
# =============================================================================
# Problem definition
# =============================================================================
# pulp.pulpTestAll()

from framework.Economics.revenue import revenue
from framework.Economics.direct_operational_cost import direct_operational_cost
from collections import defaultdict
import numpy as np
from pulp import *
import pandas as pd
import pprint
aircraft_data = baseline_aircraft()

# Available cities:
# ['CD1', 'CD2', 'CD3', 'CD4', 'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']
departures = ['CD1', 'CD2', 'CD3', 'CD4',
              'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']
arrivals = ['CD1', 'CD2', 'CD3', 'CD4',
            'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']
arrivals_2 = ['CD1', 'CD2', 'CD3', 'CD4',
              'CD5', 'CD6', 'CD7', 'CD8', 'CD9', 'CD10']

# departures =  ['CD1', 'CD2', 'CD3', 'CD4']
# arrivals =  ['CD1', 'CD2', 'CD3', 'CD4']
# arrivals_2 =  ['CD1', 'CD2', 'CD3', 'CD4']

# Define minimization problem
prob = LpProblem("Network", LpMaximize)

df1 = pd.read_csv('Database/distance.csv')
df1 = (df1.T)
print('=============================================================================')
print('Distance matrix:')
print('-----------------------------------------------------------------------------')
print(df1)
distances = df1.to_dict()

df2 = pd.read_csv('Database/demand.csv')
df2 = round((df2.T)*0.1)
print('=============================================================================')
print('Demand matrix:')
print('-----------------------------------------------------------------------------')
print(df2)
print('demand:', np.sum(np.sum(df2)))
df2 = np.round(df2)
demand = df2.to_dict()

df3 = pd.read_csv('Database/doc.csv')
df3 = (df3.T)
print('=============================================================================')
print('DOC matrix:')
print('-----------------------------------------------------------------------------')
print(df3)
doc0 = df3.to_dict()

doc = {}
for i in departures:
    for k in arrivals:
        if i != k:
            doc[(i, k)] = np.round(doc0[i][k])
        else:
            doc[(i, k)] = np.round(doc0[i][k])

print(doc)
pax_number = 78
pax_capacity = 78
load_factor = pax_number/pax_capacity
revenue_ik = defaultdict(dict)

for i in departures:
    for k in arrivals:
        if i != k:
            revenue_ik[(i, k)] = round(
                revenue(demand[i][k], distances[i][k], pax_capacity, pax_number))
        else:
            revenue_ik[(i, k)] = 0

print('=============================================================================')
print('Revemue:')
print('-----------------------------------------------------------------------------')
print(revenue_ik)
print('-----------------------------------------------------------------------------')


planes = {'P1': {'w': 78, 'r': 1063, 'v': 252, 'f': 1481, 'm': 758}}

# =============================================================================
# Decision variables definition
# =============================================================================

# Number of airplanes of a given type flying (i, k):
nika = LpVariable.dicts('nika', [(i, k) for i in departures
                                 for k in arrivals],
                        0, None, LpInteger)

# Number of passengers transported from route (i, j, k)
xijk = LpVariable.dicts('numPac',
                        [(i, j, k) for i in departures
                         for j in arrivals_2
                         for k in arrivals],
                        0, None, LpInteger)


# Route capacity:
'''
Capacidade da rota (i, k) definida pela somatória de numero de aviões do tipo P fazendo a rota (i, k) vezes a capacidade do avião P
'''
G = {}
for i in departures:
    for k in arrivals:
        G[(i, k)] = nika[(i, k)]*planes['P1']['w']

# =============================================================================
# Objective function
# =============================================================================

prob += lpSum(revenue_ik) - lpSum(nika[(i, k)]*2*doc[(i, k)]
                                  for i in departures for k in arrivals if i != k)

# prob += lpSum(revenue) - lpSum(nika[(i, k)]*doc[(i, k)] for i in departures for k in arrivals if i != k)
# =============================================================================
# Constraints
# =============================================================================
# Demand constraint
for i in departures:
    for j in arrivals_2:
        for k in arrivals:
            if i != j:
                prob += lpSum(xijk[(i, j, k)]
                              for k in arrivals) == demand[i][j]

# Capacity constraint
for i in departures:
    for j in arrivals_2:
        for k in arrivals:
            if i != k:
                prob += lpSum(xijk[(i, j, k)] for j in arrivals_2) <= G[(i, k)]

# Capacity constraint
for i in departures:
    for j in arrivals_2:
        for k in arrivals:
            if k != j:
                prob += lpSum(xijk[(i, j, k)] for i in departures) <= G[(j, k)]


# =============================================================================
# Solve problem
# =============================================================================

prob.solve(GLPK(msg=0, timeLimit=60*5))
# prob.solve(solver = GLPK_CMD(timeLimit=60*5))

# prob.solve()
# prob.solve(PULP_CBC_CMD(maxSeconds=60*5))


print('Status:', LpStatus[prob.status])

# Print solution to CONTINUOUS
pax = []
for v in prob.variables():
    print(v.name, "=", v.varValue)
    pax.append(v.varValue)

print(sum(pax))
df4 = pd.DataFrame(pax)
df4.to_csv('PAX.csv')
# Print optimal
print('Total profit [$US]:', value(prob.objective))
