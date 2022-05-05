"""
MDOAirB

Description:
    - This module takes the information of the missions obtained from
    ADS-B database (Database/Routes) and tranform them into vectors that
    are used as inputs for the climb and descent integration.
    - The calaculation of the actual horizontal distance from lat and lot
    is performed as well

Reference: 
    - 

TODO's:
    -

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

import os, fnmatch
from types import new_class
import pandas as pd
import numpy as np
import scipy as sp
from scipy import interpolate
import haversine
from haversine import haversine, Unit
import matplotlib.pyplot as plt

from framework.Attributes.Airspeed.airspeed import mach_to_V_cas, V_tas_to_mach

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def plot_mission(flight):

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend',fontsize=12) # using a size in points
    plt.rc('legend',fontsize='medium') # using a named size
    plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

    print(flight.head())


    flight_cli = flight.loc[flight['flight_phase'] == 'CL']
    flight_des = flight.loc[flight['flight_phase'] == 'DE']
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(flight['times'], flight['alt'],'-',label='altitude')
    ax.plot(flight_cli['times'], flight_cli['alt'],'x',label='altitude')
    ax.plot(flight_des['times'], flight_des['alt'],'o',label='altitude')

    ax.set_xlabel('time [s]')
    ax.set_ylabel('altitude [ft]')
    ax2 = ax.twinx()
    ax2.plot(flight['times'], flight['tas'],'r-',label='altitude')


def actual_mission_range(departure,arrival):
    """
    Description:
        - This function calculates the mission range based in actual lat lon flight data
    Inputs:
        - departure - departure airport IATA name
        - arrival - arrival airport IATA name
    Outputs:
        - distance - actual distance [nm]
    """
    listOfFiles = os.listdir('Database/Routes/'+ departure+'_'+arrival+'/.')  
    pattern = "*.csv"

    list_of_altitudes = []
    for entry in listOfFiles:  
        if fnmatch.fnmatch(entry, pattern):
            read_File_as_DF=pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+entry)
            just_name = os.path.splitext(entry)[0]
            list_of_altitudes.append(just_name)

    sorted_altitudes = sorted(list_of_altitudes,reverse=True)

    for i in sorted_altitudes:
        flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+i+'.csv', header=0, delimiter=',')
        plot_mission(flight)
    
    return 


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

max_altitude = 42000

departure= 'AMS'
arrival = 'IST'

departures = ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']
             #  0     1     2     3     4     5      6     7    8     9     10   11    12    13     14   
arrivals =   ['FRA','LHR','CDG','AMS','MAD','BCN','FCO','DUB','VIE','ZRH','ARN','DME','HEL','IST','KBP']

distance = actual_mission_range(departures[5],arrivals[10])
print(distance)



plt.show()
