"""
File name : Network optimization function
Authors   : Alejandro Rios
Email     : aarc.88@gmail.com
Date      : June 2020
Last edit : January 2021
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    - This function performs the network optimization using linear programming
    algorithm (1-stop model)
Inputs:
    - Distance matrix
    - Demand matrix
    - DOC matrix
    - Pax capacity
Outputs:
    - Network Profit [USD]
    - Route frequencies 
TODO's:
    -
"""
# =============================================================================
# IMPORTS
# =============================================================================
import copy
import csv
import getopt
import json
from multiprocessing.dummy import Namespace
import os
import pickle
import sys
from collections import defaultdict

import haversine
import jsonschema
import numpy as np
import pandas as pd
from framework.Database.Aircrafts.baseline_aircraft_parameters import \
    initialize_aircraft_parameters
from framework.Database.Airports.airports_database import AIRPORTS_DATABASE
from framework.Economics.revenue import revenue
from framework.utilities.logger import get_logger
from haversine import Unit, haversine
from jsonschema import validate
from pulp import *

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])


def dict_to_list(computation_mode, airports_keys, info_acft):

    distances = info_acft['distances']
    demands = info_acft['demands']
    doc0 = info_acft['DOC_ik']

    distances_list = []
    if (computation_mode == 0):
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i < j:
                    distances_list.append(distances[airports_keys[i]][airports_keys[j]])
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i > j:
                    distances_list.append(distances[airports_keys[i]][airports_keys[j]])
    else:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j:
                    distances_list.append(distances[airports_keys[i]][airports_keys[j]])
        
    demand_list = []
    if (computation_mode == 0):
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i < j:
                    demand_list.append(demands[airports_keys[i]][airports_keys[j]]['demand'])
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i > j:
                    demand_list.append(demands[airports_keys[i]][airports_keys[j]]['demand'])
    else:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j:
                    demand_list.append(demands[airports_keys[i]][airports_keys[j]]['demand'])

    demand_sum = sum(demand_list)

    docs_list = []
    for i in range(len(airports_keys)):
        for j in range(len(airports_keys)):
            if i != j and i < j:
                if (doc0[airports_keys[i]][airports_keys[j]] > 10000000) or (demands[airports_keys[i]][airports_keys[j]]['demand']==0):
                    docs_list.append(100000000000000000000000000000)
                else:
                    docs_list.append(doc0[airports_keys[i]][airports_keys[j]])
    for i in range(len(airports_keys)):
        for j in range(len(airports_keys)):
            if i != j and i > j:
                if (doc0[airports_keys[i]][airports_keys[j]] > 10000000) or (demands[airports_keys[i]][airports_keys[j]]['demand']==0):
                    docs_list.append(100000000000000000000000000000)
                else:
                    docs_list.append(doc0[airports_keys[i]][airports_keys[j]])

    froms_list = []
    if (computation_mode == 0):
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i < j:
                    froms_list.append(i)
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i > j:
                    froms_list.append(i)
    else:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j:
                    froms_list.append(i)

    tos_list = []
    if (computation_mode == 0):
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i < j:
                    tos_list.append(j)
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j and i > j:
                    tos_list.append(j)
    else:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i != j:
                    tos_list.append(j)


    return distances_list, demand_list, demand_sum, docs_list, froms_list, tos_list

def family_network_optimization(computation_mode, airports_keys, acft1, acft2, acft3):
    log.info('==== Start network optimization module ====')


    distances_list, demand_list, demand_sum, docs_list1, froms_list, tos_list = dict_to_list(computation_mode, airports_keys, acft1)
    _, _, _, docs_list2, _, _ = dict_to_list(computation_mode, airports_keys, acft2)
    _, _, _, docs_list3, _, _ = dict_to_list(computation_mode, airports_keys, acft3)


    vehicle01= acft1['vehicle']
    vehicle02= acft2['vehicle']
    vehicle03= acft3['vehicle']


    demands = acft1['demands']
    doc0_01 = acft1['DOC_ik']
    doc0_02 = acft2['DOC_ik']
    doc0_03 = acft3['DOC_ik']
    distances= acft1['distances']
    
    # Definition of cities to be considered as departure_airport, first stop, final airport
    aircraft01 = vehicle01['aircraft']
    aircraft02 = vehicle02['aircraft']
    aircraft03 = vehicle03['aircraft']

    operations01 = vehicle01['operations']
    operations02 = vehicle02['operations']
    operations03 = vehicle03['operations']

    results01 = vehicle01['results']
    results02 = vehicle02['results']
    results03 = vehicle03['results']
    # doc0 = np.load('Database/DOC/DOC.npy',allow_pickle=True)
    # doc0 = doc0.tolist() 

    pax_capacity01 = np.floor(aircraft01['passenger_capacity'])  # Passenger capacity
    pax_capacity02 = np.floor(aircraft02['passenger_capacity'])  # Passenger capacity
    pax_capacity03 = np.floor(aircraft03['passenger_capacity'])  # Passenger capacity

    # Define minimization problem
    # prob = LpProblem("Network", LpMaximize)
    prob = LpProblem("Network", LpMinimize)

    pax_number01 = int(operations01['reference_load_factor']*pax_capacity01)
    pax_number02 = int(operations02['reference_load_factor']*pax_capacity02)
    pax_number03 = int(operations03['reference_load_factor']*pax_capacity03)

    average_ticket_price01 = operations01['average_ticket_price']
    average_ticket_price02 = operations02['average_ticket_price']
    average_ticket_price03 = operations03['average_ticket_price']




    arcs = list(range(len(froms_list)))
    planes = {'P1': {'w': pax_number01}, 'P2': {'w': pax_number02} ,'P3': {'w': pax_number03}}

    avg_capacity1 = pax_number01
    avg_capacity2 = pax_number02
    avg_capacity3 = pax_number03

    avg_vel = 400
    avg_vel = [avg_vel]*len(froms_list)
    avg_grnd_time = 0.5
    time_allowed = 13
    allowed_planes = [round(13/((distances_list[i]/avg_vel[i])+avg_grnd_time)) for i in range(len(arcs))]

    ############################################################################################
    def flatten_dict(dd, separator ='_', prefix =''):
        return { prefix + separator + k if prefix else k : v
                    for kk, vv in dd.items()
                    for k, v in flatten_dict(vv, separator, kk).items()
                } if isinstance(dd, dict) else { prefix : dd }

    def restructure_data(aux_mat,n):
        aux_mat = np.reshape(aux_mat, (n,n-1))
        new_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    new_mat[i][j] = 0
                elif j<=i:
                    new_mat[i][j] = aux_mat[i][j]
                else:
                    new_mat[i][j] = aux_mat[i][j-1]
        return new_mat

    if (computation_mode == 0):
        nodes = list(range(len(airports_keys)*2))

        demand_aux = []
        supply_aux = []
        for i in range(len(airports_keys)):
            aux1 = [demands[airports_keys[i]][airports_keys[j]]['demand'] for j in range(len(airports_keys)) if i != j and i > j ]
            demand_aux.append(sum(aux1))
            aux2 = [demands[airports_keys[i]][airports_keys[j]]['demand'] for j in range(len(airports_keys)) if i != j and i < j ]
            supply_aux.append(sum(aux2))

        sup_dem_ij = [x - y for x, y in zip(demand_aux, supply_aux)]
        sup_dem_ji = [x - y for x, y in zip(supply_aux, demand_aux)]

        prob = LpProblem("NetOptMin", LpMinimize)
        # prob = LpProblem("NetOptMax", LpMaximize)

        flow = LpVariable.dicts("flow",(arcs),0,None,LpInteger)
        aircrafts01 = LpVariable.dicts("aircrafts1",(arcs),0,10,LpInteger)
        aircrafts02 = LpVariable.dicts("aircrafts2",(arcs),0,10,LpInteger)
        aircrafts03 = LpVariable.dicts("aircrafts3",(arcs),0,10,LpInteger)



        # prob += lpSum([aircrafts01[i]*docs_list1[i] for i in range(len(arcs))])
        # prob += lpSum([aircrafts02[i]*docs_list2[i] for i in range(len(arcs))])
        # prob += lpSum([aircrafts03[i]*docs_list3[i] for i in range(len(arcs))])

        prob += lpSum([aircrafts01[i]*docs_list1[i] + aircrafts02[i]*docs_list2[i] + aircrafts03[i]*docs_list3[i] for i in range(len(arcs))])

        

        # prob += lpSum([flow[i]*average_ticket_price for i in range(len(arcs))]) - lpSum([aircrafts[i]*docs_list[i] for i in range(len(arcs))])
        # prob += lpSum([demand_list[i]*distances_list[i]*((flow[i]*average_ticket_price)/(flow[i]*distances_list[i])) for i in range(len(arcs))])  - lpSum([aircrafts[i]*docs_list[i] for i in range(len(arcs))])


        prob += lpSum([flow[i] for i in range(len(arcs))]) == demand_sum

        for i in range(0,len(nodes)//2):
                prob += lpSum(flow[j] for j in range(0,len(arcs)//2) if tos_list[j] == i) - lpSum(flow[j] for j in range(0,len(arcs)//2) if froms_list[j] == i) == sup_dem_ij[i]

        for i in range(0,len(nodes)//2):
                prob += lpSum(flow[j] for j in range(len(arcs)//2,len(arcs)) if tos_list[j] == i) - lpSum(flow[j] for j in range(len(arcs)//2,len(arcs)) if froms_list[j] == i) == sup_dem_ji[i]
            
        for i in arcs:
            prob += flow[i] <= aircrafts01[i]*avg_capacity1  + aircrafts02[i]*avg_capacity2 + aircrafts03[i]*avg_capacity3



        # =============================================================================
        # Solve linear programming problem (Network optimization)
        # =============================================================================
        log.info('==== Start PuLP optimization ====')
        # prob.solve(GLPK(timeLimit=60*5, msg = 0))
        prob.solve(COIN_CMD(timeLimit=60*5, msg = 0))
        # prob.solve(COIN_CMD(timeLimit=60*5, msg = 0))
        # prob.solve(PULP_CBC_CMD(timeLimit=60*6, msg = 0))

        log.info('==== Start PuLP optimization ====')
        print('Problem solution:',value(prob.objective))
        
        names = []
        values = []
        for v in prob.variables():
            # print(v.name, "=", v.varValue)
            names.append(v.name)
            values.append(v.varValue)

        combined = zip(names, values)
        zipped_sorted = sorted(combined, key=lambda x: int(x[0].split("_")[-1].split(".")[0]))
        names, values = map(list, zip(*zipped_sorted))
     

        log.info('Network optimization status: {}'.format(LpStatus[prob.status]))
        try:
            condition = LpStatus[prob.status]
            if condition != 'Optimal':
                raise ValueError('Optimal network solution NOT found')
        except (ValueError, IndexError):
            exit('Could not complete network optimization')
        


        list_aircrafts1 = []
        list_aircrafts2 = []
        list_aircrafts3 = []
        list_of_pax = []
        for v in range(len(names)):
            variable_name = names[v]         
            if variable_name.find('aircrafts1') != -1:
                list_aircrafts1.append(values[v])
            if variable_name.find('aircrafts2') != -1:
                list_aircrafts2.append(values[v])
            if variable_name.find('aircrafts3') != -1:
                list_aircrafts3.append(values[v])
            if variable_name.find('flow') != -1:
                # print(v.name, "=", v.varValue)
                list_of_pax.append(values[v])

        # print('flow',sum(list_of_pax))

        idx = 0
        fraction = np.zeros((len(airports_keys),len(airports_keys)))
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if i==j:
                    fraction[i][j] = 0
                else:
                    fraction[i][j] = list_of_pax[idx]
                    idx = idx+1
    else:
        aircrafts01 = [demand_list[i]/pax_number01 for i in range(len(arcs))]
        # aircrafts = pax_number
        list_of_pax = [aircrafts01[i]*pax_number01 for i in range(len(arcs))]

        n = len(airports_keys)

        fraction = restructure_data(list_of_pax,n)

    # print('Flow matrix:',fraction)


    list_size = len(airports_keys)**2 - len(airports_keys)
    aircrafts1= np.zeros((len(airports_keys),len(airports_keys)))
    aircrafts2 =  np.zeros((len(airports_keys),len(airports_keys)))
    aircrafts3 = np.zeros((len(airports_keys),len(airports_keys)))
    idx = 0
    while idx<list_size/2:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if j>i:
                    aircrafts1[i][j] = list_aircrafts1[idx]
                    aircrafts2[i][j] = list_aircrafts2[idx]
                    aircrafts3[i][j] = list_aircrafts3[idx]
                    idx = idx+1
    while idx<list_size:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if j<i:
                    aircrafts1[i][j] = list_aircrafts1[idx]
                    aircrafts2[i][j] = list_aircrafts2[idx]
                    aircrafts3[i][j] = list_aircrafts3[idx]
                    idx = idx+1



    print('===============================')
    print('Aircraft1 matrix:',aircrafts1)
    print('===============================')
    print('Aircraft2 matrix:',aircrafts2)
    print('===============================')
    print('Aircraft3 matrix:',aircrafts3)
    print('===============================')

    np.save('Database/Results_FamOpt/acft1', aircrafts1)
    np.save('Database/Results_FamOpt/acft2', aircrafts2)
    np.save('Database/Results_FamOpt/acft3', aircrafts3)

    revenue_1_list = []
    for i in range(len(list_of_pax)):
        if (list_of_pax[i] <= 0):
            revenue_1_list.append(0)
        else:
            revenue_1_list.append(list_of_pax[i]*distances_list[i]*((list_of_pax[i]*average_ticket_price01)/(list_of_pax[i]*distances_list[i])))
            # revenue_1_list.append(1.1*list_of_pax[i]*average_ticket_price01)
            

    revenue_1_list = [0 if x != x else x for x in revenue_1_list]

    revenue_tot= sum(revenue_1_list)

    revenue_mat = np.zeros((len(airports_keys),len(airports_keys)))


    doc_1_list = []
    for i in range(len(list_of_pax)):
        if (list_of_pax[i] <= 0):
            doc_1_list.append(0)
        else:
            doc_1_list.append(docs_list1[i]*list_aircrafts1[i] + docs_list2[i]*list_aircrafts2[i] +docs_list3[i]*list_aircrafts3[i])
        # revenue_1_list.append(1.1*list_of_pax[i]*average_ticket_price01)

    doc_tot= sum(doc_1_list)

    # profit = np.int(1.0*revenue_tot - 1.0*doc_tot)
    # results01['profit'] = np.round(profit)
    # results01['total_cost'] = np.round(doc_tot)

    idx = 0
    while idx<list_size/2:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if j>i:
                    revenue_mat[i][j] = revenue_1_list[idx]
                    idx = idx+1
    while idx<list_size:
        for i in range(len(airports_keys)):
            for j in range(len(airports_keys)):
                if j<i:
                    revenue_mat[i][j] = revenue_1_list[idx]
                    idx = idx+1


    list_of_airplanes_processed = np.zeros((len(airports_keys),len(airports_keys)))
    for i in range(len(airports_keys)):
        for j in range(len(airports_keys)):
            list_of_airplanes_processed[i][j]= aircrafts1[i][j] +aircrafts2[i][j] + aircrafts3[i][j]

    # print('Aircraft matrix:',list_of_airplanes_processed)

    # print('Aircraft matrix:',aircrafts1)

    DOCmat1 =  np.zeros((len(airports_keys),len(airports_keys)))
    DOCmat2 =  np.zeros((len(airports_keys),len(airports_keys)))
    DOCmat3 =  np.zeros((len(airports_keys),len(airports_keys)))
    for i in range(len(airports_keys)):
        for j in range(len(airports_keys)):
            if i != j:
                DOCmat1[i][j] = np.round(doc0_01[airports_keys[i]][airports_keys[j]])
                DOCmat2[i][j] = np.round(doc0_02[airports_keys[i]][airports_keys[j]])
                DOCmat3[i][j] = np.round(doc0_03[airports_keys[i]][airports_keys[j]])
            else:
                DOCmat1[i][j] = 0
                DOCmat2[i][j] = 0
                DOCmat3[i][j] = 0


    DOC_proccessed1 = np.zeros((len(airports_keys),len(airports_keys)))
    DOC_proccessed2 = np.zeros((len(airports_keys),len(airports_keys)))
    DOC_proccessed3 = np.zeros((len(airports_keys),len(airports_keys)))
    for i in range(len(airports_keys)):
        for j in range(len(airports_keys)):
            DOC_proccessed1[i][j] = DOCmat1[i][j]*aircrafts1[i][j]
            DOC_proccessed2[i][j] = DOCmat2[i][j]*aircrafts2[i][j]
            DOC_proccessed3[i][j] = DOCmat3[i][j]*aircrafts3[i][j]



    results01['aircrafts_used']= np.sum(aircrafts1)
    results01['covered_demand'] = np.sum(list_of_pax)
    results01['total_revenue'] = revenue_tot
    airplanes_ik = {}
    n = 0
    for i in range(len(airports_keys)):
        for k in range(len(airports_keys)):
            # print(list_airplanes[n])
            airplanes_ik[(airports_keys[i],airports_keys[k])] = aircrafts1[i][k] + aircrafts2[i][k] +aircrafts3[i][k] 

    list_airplanes_db = pd.DataFrame(aircrafts1)
    list_airplanes_db.to_csv('Database/Network/frequencies.csv')
    
    airplanes_flatt = flatten_dict(airplanes_ik)
    np.save('Database/Network/frequencies.npy', airplanes_flatt) 

    # airplanes_flatt = flatten_dict(airplanes_ik)
    
    # np.save('Database/Network/frequencies.npy', airplanes_flatt) 

    # list_of_pax_db = pd.DataFrame(list_pax_processed)

    # list_of_pax_db = list_of_pax_db.loc[~(list_of_pax_db==0).all(axis=1)]
    # print(list_of_pax)

    # list_of_pax_db.to_csv('Database/Network/pax.csv')

    DOC_tot = np.sum(DOC_proccessed1+DOC_proccessed2+DOC_proccessed3)

    
    profit = np.int(1.0*revenue_tot - 1.2*DOC_tot)

    results01['profit'] = np.round(profit)
    results01['total_cost'] = np.round(DOC_tot)

    print('margin',profit/revenue_tot)
    print('profit',profit)


    pax_number_flatt = aircrafts1.flatten() 
    
    pax_number_df = pd.DataFrame({'pax_number':pax_number_flatt})
    kpi_df1 = pd.DataFrame()
    # print(pax_number_df)
    kpi_df1['pax_number'] = pax_number_df['pax_number'].values
    # print(kpi_df1["pax_number"])

    # kpi_df1.drop(columns=["variable_object"], inplace=True)
    kpi_df1.to_csv("Test/optimization_solution01.csv")

    n = len(airports_keys)

    if (computation_mode == 0):
        # aircrafts_aux = np.reshape(aircrafts, (n,n-1))

        kpi_df2 = pd.DataFrame.from_dict(aircrafts01, orient="index", 
                                    columns = ["variable_object"])
        # kpi_df2.idx =  pd.MultiIndex.from_tuples(kpi_df2.idx, 
        #                             names=["origin", "destination"])
        kpi_df2.reset_index(inplace=True)

        kpi_df2["aircraft_number"] =  kpi_df2["variable_object"].apply(lambda item: item.varValue)

        kpi_df2.drop(columns=["variable_object"], inplace=True)
    else:
        kpi_df2 = pd.DataFrame(aircrafts01, columns = ["aircraft_number"])


    distances_flatt = flatten_dict(distances)
    # doc_flatt = flatten_dict(DOC)
    demand_flatt = flatten_dict(demands)
    revenue_flatt = revenue_mat.flatten()
    doc_flatt = DOC_proccessed1.flatten()

    doc_df = pd.DataFrame({'doc':doc_flatt})
    revenue_df = pd.DataFrame({'revenue':revenue_flatt})


    distance_df =  pd.DataFrame.from_dict(distances_flatt,orient="index",columns=['distances'])
    # doc_df =  pd.DataFrame.from_dict(doc_flatt,orient="idx",columns=['doc'])
    demand_df =  pd.DataFrame.from_dict(demand_flatt,orient="index",columns=['demand'])
    # revenue_df =  pd.DataFrame.from_dict(revenue_flatt,orient="idx",columns=['revenue'])

    kpi_df2['distances'] = distances_list
    kpi_df2['doc'] = docs_list1
    kpi_df2['demand'] = demand_list
    # kpi_df2['revenue'] = revenue_df['revenue'].values
    
    kpi_df2['active_arcs'] = np.where(kpi_df2["aircraft_number"] > 0, 1, 0)
    X = kpi_df2['active_arcs'].to_numpy()
    X = restructure_data(X,n)

    Distances = kpi_df2['distances'].to_numpy()
    Distances = restructure_data(Distances,n)

    Demand = kpi_df2['demand'].to_numpy()
    Demand= restructure_data(Demand,n)

    N = 0
    for i,j in np.ndindex(X.shape):
        if X[i,j] == 1:
            N = N+1

    DON = np.zeros(n)
    for i in range(n):
        DON[i] = 0
        for j in range(n):
            if i != n:
                if X[i,j] == 1:
                    DON[i] = DON[i]+1
    
    results01['avg_degree_nodes'] = np.mean(DON)

    R = 500
    C = np.zeros(n)
    for i in range(n):
        CON =0
        MAXCON = 0
        for j in range(n):
            if i != j:
                if Distances[i,j] <= R:
                    MAXCON = MAXCON + 1
                    if X[i,j] == 1:
                        CON = CON+1
        if MAXCON>0:
            C[i] = CON/MAXCON
        else:
            C[i] = 0

    results01['average_clustering'] = np.mean(C)


    LF = np.ones((n,n))
    FREQ = X

    results01['number_of_frequencies'] = np.sum(list_of_airplanes_processed)

    log.info('==== End network optimization module ====')
    return profit, vehicle01, kpi_df1, kpi_df2, airplanes_ik

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# # Load origin-destination distance matrix [nm]
# distances_db = pd.read_csv('Database/Distance/distance.csv')
# distances_db = (distances_db.T)
# distances = distances_db.to_dict()  # Convert to dictionaty

# # Load daily demand matrix and multiply by market share (10%)
# demand_db = pd.read_csv('Database//Demand/demand.csv')
# demand_db = round(market_share*(demand_db.T))
# demand = demand_db.to_dict()
# from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters

# vehicle = initialize_aircraft_parameters()
# operations = vehicle['operations']
# departures = ['CD0', 'CD1', 'CD2', 'CD3',
#                 'CD4', 'CD5', 'CD6', 'CD7', 'CD8', 'CD9']
# arrivals = ['CD0', 'CD1', 'CD2', 'CD3',
#                 'CD4', 'CD5', 'CD6', 'CD7', 'CD8', 'CD9']

# # departures = ['CD0', 'CD1', 'CD2', 'CD3',
# #                 'CD4']
# # arrivals = ['CD0', 'CD1', 'CD2', 'CD3',
# #             'CD4']


# # Load origin-destination distance matrix [nm]
# distances_db = pd.read_csv('Database/Distance/distance.csv')
# distances_db = (distances_db)
# distances = distances_db.to_dict()  # Convert to dictionaty

# market_share = operations['market_share']
# # # Load dai
# demand_db= pd.read_csv('Database/Demand/demand.csv')
# demand_db= round(market_share*(demand_db.T))
# demand = demand_db.to_dict()

# df3 = pd.read_csv('Database/DOC/DOC_test5.csv')
# df3 = (df3.T)
# doc0 = df3.to_dict()

# active_airports_db = pd.read_csv('Database/Demand/switch_matrix_full.csv')
# active_airports_db = active_airports_db
# active_airports = active_airports_db .to_dict()


# demand_in = {}
# for i in range(len(departures)):
#     demand_in[departures[i]] = {}
#     for k in range(len(arrivals)):
#         if i != k:
#             demand_in[departures[i]][arrivals[k]] = demand[departures[i]][arrivals[k]]*active_airports[departures[i]][arrivals[k]]
#         else:
#             demand_in[departures[i]][arrivals[k]] =  demand[departures[i]][arrivals[k]]*active_airports[departures[i]][arrivals[k]]
# pax_capacity = 144



# network_optimization(arrivals, departures, distances, demand_in, active_airports, doc0, pax_capacity, vehicle)

CUSTOM_INPUTS_SCHEMA = 'Database/JsonSchema/Custom_Inputs.schema.json'

class CustomInputsError(Exception):
	def __init__(self, message):
		self.message = f"Custom inputs issue: {message}"
		super().__init__(self.message)

def check_runways(demands, airports):
	for departure in demands:
		for arrival in demands[departure]:
			takeoff_runway = demands[departure][arrival]["takeoff_runway"]
			if takeoff_runway not in airports[departure]["runways"]:
				raise CustomInputsError(f'Take-of runway {takeoff_runway} do not exit for airport {departure} in MDO database')
			landing_runway = demands[departure][arrival]["landing_runway"]
			if landing_runway not in airports[arrival]["runways"]:
				raise CustomInputsError(f'Landing runway {landing_runway} do not exit for airport {arrival} in MDO database')

def check_airports(airport_keys):
	try:
		airports = { k: AIRPORTS_DATABASE[k] for k in airport_keys }
	except KeyError as key_error:
		raise CustomInputsError(f'Airports {key_error.args} do not exit in MDO database')

	return airports

def haversine_distance(coordinates_departure,coordinates_arrival):
    # Perform haversine distance calculation in nautical miles
    distance = float(haversine(coordinates_departure,coordinates_arrival,unit='nmi'))
    return distance

def check_demands(data, fixed_parameters):
	airport_keys = list(data.keys())
	for key in data:
		if (key in data[key]):
			raise CustomInputsError(f'Airport {key} exist on both departure and arrival for the same demand')
		airport_keys = airport_keys + list(set(data[key].keys()) - set(airport_keys))

	airports = check_airports(airport_keys)

	check_runways(data, airports)

	market_share = fixed_parameters['operations']['market_share']

	distances = {}
	for departure in airport_keys:
		distances[departure] = {}
		if (departure not in data):
			data[departure] = {}
		for arrival in airport_keys:
			if (arrival not in data[departure]):
				data[departure][arrival] = {}
				data[departure][arrival]['demand'] = 0
				distances[departure][arrival] = 0
			else:
				data[departure][arrival]['demand'] = np.round(market_share * data[departure][arrival]['demand'])

				coordinates_departure = (airports[departure]["latitude"],airports[departure]["longitude"])
				coordinates_arrival = (airports[arrival]["latitude"],airports[arrival]["longitude"])
				distances[departure][arrival] = round(haversine_distance(coordinates_departure,coordinates_arrival))

	return airports, distances, data

def check_design_variables(data):
	for key in data:
		if (data[key]["lower_band"] > data[key]["upper_band"]):
			raise CustomInputsError(f'Lower band {data[key]["lower_band"]} is greater than upper band {data[key]["upper_band"]} for {key}')

def read_custom_inputs(schema_path, file_path):
	computation_mode = 0
	route_computation_mode = 0
	design_variables = {}
	custom_fixed_parameters = {}
	fixed_aircraft = {}

	try:
		with open(schema_path) as f:
			schema = json.load(f)

		with open(file_path) as f:
			data = json.load(f)

		validate(instance=data, schema=schema)

		if ("computation_mode" in data):
			computation_mode = data["computation_mode"]

		if ("route_computation_mode" in data):
			route_computation_mode = data["route_computation_mode"]

		if ("demands" not in data):
			raise CustomInputsError(f"Demands is mandatory in custom inputs")

		if ("design_variables" in data):
			design_variables = data["design_variables"]
			check_design_variables(design_variables)

		if ("fixed_parameters" in data):
			custom_fixed_parameters = data["fixed_parameters"]

		# Update vehicle with fixed parameters
		fixed_parameters = initialize_aircraft_parameters()
		fixed_parameters = update_vehicle(fixed_parameters, custom_fixed_parameters)

		if ("fixed_aircraft" in data):
			fixed_aircraft = data["fixed_aircraft"]

		airports, distances, demands = check_demands(data["demands"], fixed_parameters)

	except OSError as os_error:
		raise CustomInputsError(f"{os_error.strerror} [{os_error.filename}]")
	except json.JSONDecodeError as dec_error:
		raise CustomInputsError(f"{dec_error.msg} [line {dec_error.lineno} column {dec_error.colno} (char {dec_error.pos})]")
	except jsonschema.exceptions.SchemaError:
		raise CustomInputsError(f"There is an error with the schema")
	except jsonschema.exceptions.ValidationError as val_error:
		path_error = ""
		for path in val_error.path:
			if (path_error):
				path_error += "."
			path_error += path
		raise CustomInputsError(f"{val_error.message} in path [{path_error}]")
    
	return computation_mode, route_computation_mode, airports, distances, demands, design_variables, fixed_parameters, fixed_aircraft

def update_vehicle(vehicle, fixed_parameters):
    for key in fixed_parameters:
        if (key in vehicle):
            vehicle[key].update(fixed_parameters[key])
    return vehicle

def usage():
	print("This is the usage function")
	print(f"Usage: {sys.argv[0]} -f <custom inputs file>")

def readArgv(argv):
	customInputsfile = ""
	try:                                
		opts, _ = getopt.getopt(argv, "hf:", ["help", "file="])
	except getopt.GetoptError:          
		usage()                         
		sys.exit(2)                     
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()                     
			sys.exit()                  
		elif opt in ("-f", "--file"):
			customInputsfile = arg               
	return customInputsfile

def main(argv):
	fixed_aircraft = {}
	customInputsfile = readArgv(argv)
	if not customInputsfile or not os.path.isfile(customInputsfile):
		print(f"Custom file {customInputsfile} does not exist")
		sys.exit(1)

	try:
		computation_mode, _, airports, distances, demands, _, _, fixed_aircraft = read_custom_inputs(CUSTOM_INPUTS_SCHEMA, customInputsfile)
	except Exception as err:
		print(f"Exception ocurred while playing custom inputs file {customInputsfile}")
		print(f"Error: {err}")
		sys.exit(1)

	with open('Database/DOC/Vehicle.pkl', 'rb') as f:
		vehicle = pickle.load(f)

	with open('Database/DOC/DOC_ori.pkl', 'rb') as f:
		doc0 = pickle.load(f)
	doc0 = {'FRA': {'FRA': 0, 'LHR': 4810, 'CDG': 3806, 'AMS': 3449, 'MAD': 8713, 'BCN': 7118, 'FCO': 6370, 'DUB': 7098, 'VIE': 4626, 'ZRH': 2977}, 'LHR': {'FRA': 4821, 'LHR': 0, 'CDG': 3343, 'AMS': 3470, 'MAD': 7792, 'BCN': 7417, 'FCO': 9010, 'DUB': 3847, 'VIE': 8107, 'ZRH': 5461}, 'CDG': {'FRA': 3790, 'LHR': 3338, 'CDG': 0, 'AMS': 3528, 'MAD': 6896, 'BCN': 5828, 'FCO': 7131, 'DUB': 5467, 'VIE': 6798, 'ZRH': 3916}, 'AMS': {'FRA': 3423, 'LHR': 3415, 'CDG': 3600, 'AMS': 0, 'MAD': 9083, 'BCN': 7780, 'FCO': 8162, 'DUB': 5299, 'VIE': 6397, 'ZRH': 4549}, 'MAD': {'FRA': 8777, 'LHR': 7825, 'CDG': 6875, 'AMS': 8930, 'MAD': 0, 'BCN': 3930, 'FCO': 8300, 'DUB': 8817, 'VIE': 10855, 'ZRH': 7805}, 'BCN': {'FRA': 7000, 'LHR': 7418, 'CDG': 5854, 'AMS': 7865, 'MAD': 3948, 'BCN': 0, 'FCO': 5816, 'DUB': 9056, 'VIE': 8568, 'ZRH': 5854}, 'FCO': {'FRA': 6352, 'LHR': 9011, 'CDG': 7170, 'AMS': 8070, 'MAD': 8266, 'BCN': 5798, 'FCO': 0, 'DUB': 11410, 'VIE': 5450, 'ZRH': 5005}, 'DUB': {'FRA': 7017, 'LHR': 3829, 'CDG': 5485, 'AMS': 5303, 'MAD': 8944, 'BCN': 9105, 'FCO': 11320, 'DUB': 0, 'VIE': 10353, 'ZRH': 7812}, 'VIE': {'FRA': 4667, 'LHR': 8009, 'CDG': 6762, 'AMS': 6404, 'MAD': 10896, 'BCN': 8657, 'FCO': 5467, 'DUB': 10349, 'VIE': 0, 'ZRH': 4583}, 'ZRH': {'FRA': 2983, 'LHR': 5447, 'CDG': 3875, 'AMS': 4504, 'MAD': 7734, 'BCN': 5793, 'FCO': 4972, 'DUB': 7734, 'VIE': 4514, 'ZRH': 0}}
	distances = {'FRA': {'FRA': 0, 'LHR': 355, 'CDG': 243, 'AMS': 198, 'MAD': 768, 'BCN': 591, 'FCO': 517, 'DUB': 589, 'VIE': 336, 'ZRH': 154}, 'LHR': {'FRA': 355, 'LHR': 0, 'CDG': 188, 'AMS': 200, 'MAD': 672, 'BCN': 620, 'FCO': 781, 'DUB': 243, 'VIE': 690, 'ZRH': 427}, 'CDG': {'FRA': 243, 'LHR': 188, 'CDG': 0, 'AMS': 215, 'MAD': 574, 'BCN': 463, 'FCO': 595, 'DUB': 425, 'VIE': 561, 'ZRH': 258}, 'AMS': {'FRA': 198, 'LHR': 200, 'CDG': 215, 'AMS': 0, 'MAD': 788, 'BCN': 670, 'FCO': 700, 'DUB': 406, 'VIE': 519, 'ZRH': 326}, 'MAD': {'FRA': 768, 'LHR': 672, 'CDG': 574, 'AMS': 788, 'MAD': 0, 'BCN': 261, 'FCO': 720, 'DUB': 784, 'VIE': 977, 'ZRH': 670}, 'BCN': {'FRA': 591, 'LHR': 620, 'CDG': 463, 'AMS': 670, 'MAD': 261, 'BCN': 0, 'FCO': 459, 'DUB': 802, 'VIE': 741, 'ZRH': 463}, 'FCO': {'FRA': 517, 'LHR': 781, 'CDG': 595, 'AMS': 700, 'MAD': 720, 'BCN': 459, 'FCO': 0, 'DUB': 1020, 'VIE': 421, 'ZRH': 375}, 'DUB': {'FRA': 589, 'LHR': 243, 'CDG': 425, 'AMS': 406, 'MAD': 784, 'BCN': 802, 'FCO': 1020, 'DUB': 0, 'VIE': 922, 'ZRH': 670}, 'VIE': {'FRA': 336, 'LHR': 690, 'CDG': 561, 'AMS': 519, 'MAD': 977, 'BCN': 741, 'FCO': 421, 'DUB': 922, 'VIE': 0, 'ZRH': 327}, 'ZRH': {'FRA': 154, 'LHR': 427, 'CDG': 258, 'AMS': 326, 'MAD': 670, 'BCN': 463, 'FCO': 375, 'DUB': 670, 'VIE': 327, 'ZRH': 0}}

	if not fixed_aircraft:
		family_network_optimization(computation_mode, list(airports.keys()), distances, demands, doc0, vehicle)

if __name__ == "__main__":
    main(sys.argv[1:])
