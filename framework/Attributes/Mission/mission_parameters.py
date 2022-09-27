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

from framework.Attributes.Airspeed.airspeed import mach_to_V_cas,V_tas_to_mach
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
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
    flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+sorted_altitudes[0]+'.csv', header=0, delimiter=',')
    climb_flight = flight.query('flight_phase == "CL" & alt >= 1500')

    chunk_size = 100

    # # Resizing vector of flights lat, lon and alt
    lat = flight['lat']
    xlat = np.arange(lat.size)
    new_xlat = np.linspace(xlat.min(), xlat.max(), chunk_size)
    lat_rz = sp.interpolate.interp1d(xlat, lat, kind='linear')(new_xlat)
    # lat_rz = sp.signal.medfilt(lat_rz,51)

    lon = flight['lon']
    xlon = np.arange(lon.size)
    new_xlon = np.linspace(xlon.min(), xlon.max(), chunk_size)
    lon_rz = sp.interpolate.interp1d(xlon, lon, kind='linear')(new_xlon)

    distances_pp = []
    distances_pp_f = []
    for j in range(len(lon_rz)-1):
        # Defining cordinates of two points to messure distance      
        coordinates0 = (lat_rz[j],lon_rz[j])
        coordinates1 = (lat_rz[j+1],lon_rz[j+1])
        # Calculating haversine distance between two points in nautical miles
        distance_pp = float(haversine(coordinates0,coordinates1,unit='nmi'))
        # Storing calculated point to point distances into a vector
        distances_pp.append(distance_pp)
        # Sum of point to point distances to obtain total distance of a flight
        distance = sum(distances_pp)

    return distance

def climb_altitudes_vector(departure,arrival,max_altitude):
    """
    Description:
        - This function take actual information of fligth (ADS-B) and create a vector
        that will be used into the climb mission integration function 
    Inputs:
        - departure - departure airport IATA name
        - arrival - arrival airport IATA name
        - max_altitude - maximum computed flight altitude [ft]
    Outputs:
        - alt_rz - vector containing altitudes [ft]
        - cas_spds_rz - vectort containing calibrated airspeed [kt]
        - mach_rz - vectort containing mach numbers
        - time_rz - vector containing times [s]
    """
    listOfFiles = os.listdir('Database/Routes/'+ departure+'_'+arrival+'/.')  
    pattern = "*.csv"

    list_of_altitudes = []
    for entry in listOfFiles:  
        if fnmatch.fnmatch(entry, pattern):
            read_File_as_DF=pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+entry)
            just_name = os.path.splitext(entry)[0]
            list_of_altitudes.append(int(just_name))


    sorted_altitudes = sorted(list_of_altitudes,reverse=True)
    
    try:
        usable_altitude = [x for x in sorted_altitudes if x<max_altitude]
        flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+str(usable_altitude[0])+'.csv', header=0, delimiter=',')
    except:
        flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+str(sorted_altitudes[0])+'.csv', header=0, delimiter=',')

    climb_flight = flight.query('flight_phase == "CL" & alt >= 1500')


    chunk_size = 30
    time= climb_flight['times']
    xtime= np.arange(time.size)
    new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
    time_rz_1 = sp.interpolate.interp1d(xtime, time, kind='linear')(new_xtime)
    time_rz = time_rz_1

    alt = climb_flight['alt']
    xalt = np.arange(alt.size)
    new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
    alt_rz = sp.interpolate.interp1d(xalt, alt, kind='linear')(new_xalt)

    spds= climb_flight['speed']
    xspds= np.arange(spds.size)
    new_xspds = np.linspace(xspds.min(), xspds.max(), chunk_size)
    spds_rz = sp.interpolate.interp1d(xspds, spds, kind='linear')(new_xspds)

    tas_spds= climb_flight['tas']
    xtas_spds= np.arange(tas_spds.size)
    new_xtas_spds = np.linspace(xtas_spds.min(), xtas_spds.max(), chunk_size)
    tas_spds_rz = sp.interpolate.interp1d(xtas_spds, tas_spds, kind='linear')(new_xtas_spds)

    machs= climb_flight['mach']
    xmachs= np.arange(machs.size)
    new_xmachs = np.linspace(xmachs.min(), xmachs.max(), chunk_size)
    mach_rz = sp.interpolate.interp1d(xmachs, machs, kind='linear')(new_xmachs)
    
    cas_spds_rz = []
    mach_new_rz = []
    for i in range(len(tas_spds_rz)):
        aux1 = mach_to_V_cas(mach_rz[i], alt_rz[i], 0)

        aux2 = V_tas_to_mach(spds_rz[i],alt_rz[i], 0)
        mach_new_rz.append(aux2)

        aux1 = mach_to_V_cas(aux2, alt_rz[i], 0)
        cas_spds_rz.append(aux1)
        cas_spds_rz.append(aux1)


    x = alt_rz
    y = cas_spds_rz
    f = interpolate.interp1d(x, y)
    xnew = np.linspace(min(alt_rz),max(alt_rz), num=30, endpoint=True)
    ynew = f(xnew)

    alt_rz = xnew
    cas_spds_rz = ynew

    y = mach_rz
    f = interpolate.interp1d(x, y)
    ynew = f(xnew)
    mach_rz = ynew



    return alt_rz,cas_spds_rz,mach_new_rz,time_rz

def descent_altitudes_vector(departure,arrival,max_altitude):
    """
    Description:
        - This function take actual information of fligth (ADS-B) and create a vector
        that will be used into the descent mission integration function 
    Inputs:
        - departure - departure airport IATA name
        - arrival - arrival airport IATA name
        - max_altitude - maximum computed flight altitude [ft]
    Outputs:
        - alt_rz - vector containing altitudes [ft]
        - cas_spds_rz - vectort containing calibrated airspeed [kt]
        - mach_rz - vectort containing mach numbers
        - time_rz - vector containing times [s]
    """
    listOfFiles = os.listdir('Database/Routes/'+ departure+'_'+arrival+'/.')  
    pattern = "*.csv"

    list_of_altitudes = []
    for entry in listOfFiles:  
        if fnmatch.fnmatch(entry, pattern):
            read_File_as_DF=pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+entry)
            just_name = os.path.splitext(entry)[0]
            list_of_altitudes.append(int(just_name))


    sorted_altitudes = sorted(list_of_altitudes,reverse=True)
    
    try:
        usable_altitude = [x for x in sorted_altitudes if x<max_altitude]
        flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+str(usable_altitude[0])+'.csv', header=0, delimiter=',')
    except:
        flight = pd.read_csv('Database/Routes/'+ departure+'_'+arrival+'/'+str(sorted_altitudes[0])+'.csv', header=0, delimiter=',')

    descent_flight = flight.query('flight_phase == "DE" & alt >= 1500')


    chunk_size = 30
    time= descent_flight['times']
    xtime= np.arange(time.size)
    new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
    time_rz_1 = sp.interpolate.interp1d(xtime, time, kind='linear')(new_xtime)
    time_rz = time_rz_1

    alt = descent_flight['alt']
    xalt = np.arange(alt.size)
    new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
    alt_rz = sp.interpolate.interp1d(xalt, alt, kind='linear')(new_xalt)

    spds= descent_flight['speed']
    xspds= np.arange(spds.size)
    new_xspds = np.linspace(xspds.min(), xspds.max(), chunk_size)
    spds_rz = sp.interpolate.interp1d(xspds, spds, kind='linear')(new_xspds)

    tas_spds= descent_flight['tas']
    xtas_spds= np.arange(tas_spds.size)
    new_xtas_spds = np.linspace(xtas_spds.min(), xtas_spds.max(), chunk_size)
    tas_spds_rz = sp.interpolate.interp1d(xtas_spds, tas_spds, kind='linear')(new_xtas_spds)

    machs= descent_flight['mach']
    xmachs= np.arange(machs.size)
    new_xmachs = np.linspace(xmachs.min(), xmachs.max(), chunk_size)
    mach_rz = sp.interpolate.interp1d(xmachs, machs, kind='linear')(new_xmachs)
    
    cas_spds_rz = []
    mach_new_rz = []
    for i in range(len(tas_spds_rz)):


        aux2 = V_tas_to_mach(spds_rz[i],alt_rz[i], 0)
        mach_new_rz.append(aux2)

        aux1 = mach_to_V_cas(aux2, alt_rz[i], 0)
        cas_spds_rz.append(aux1)

    # fig1, ax1 = plt.subplots()
    # ax1.plot(time_rz, cas_spds_rz,'-x')
    # ax1.legend(['data'], loc='best')

    x = alt_rz
    y = cas_spds_rz
    f = interpolate.interp1d(x, y)
    xnew = np.linspace(max(alt_rz),min(alt_rz), num=30, endpoint=True)
    ynew = f(xnew)

    alt_rz = xnew
    cas_spds_rz = ynew

    y = mach_rz
    f = interpolate.interp1d(x, y)
    ynew = f(xnew)
    mach_rz = ynew

    return alt_rz,cas_spds_rz,mach_new_rz,time_rz

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# max_altitude = 37000

# departure= 'ZRH'
# arrival = 'LHR'

# distance = actual_mission_range(departure,arrival)


# alt_rz, spds_rz, mach_rz, time_rz = climb_altitudes_vector(departure,arrival,max_altitude)

# x = alt_rz
# y = mach_rz
# f = interpolate.interp1d(x, y)
# xnew = np.linspace(min(alt_rz),max(alt_rz), num=30, endpoint=True)
# ynew = f(xnew)
# fig1, ax1 = plt.subplots()
# ax1.plot(alt_rz, mach_rz,'-x')
# ax1.legend(['data'], loc='best')



# fig1, ax1 = plt.subplots()
# ax1.plot(alt_rz, spds_rz,'-x')
# ax1.legend(['data'], loc='best')

# alt_rz, spds_rz, mach_rz, time_rz = descent_altitudes_vector(departure,arrival,max_altitude)

# # x = alt_rz
# # y = mach_rz
# # f = interpolate.interp1d(x, y)
# # xnew = np.linspace(min(alt_rz),max(alt_rz), num=30, endpoint=True)
# # fig2, ax2 = plt.subplots()
# # ax2.plot(x, y, 'o', xnew, f(xnew), '-x')
# # ax2.legend(['data', 'linear', 'cubic'], loc='best')

# fig2, ax2 = plt.subplots()
# ax2.plot(alt_rz, spds_rz,'-x')
# ax2.legend(['data'], loc='best')

# plt.show()
