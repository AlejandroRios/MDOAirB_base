"""
MDOAirB

Description:
    - This module contains functions for the creation of output files for
    the evaluated aicraft

Reference:
    -

TODO's:
    -

| Authors: Alejandro Rios
| Email: aarc.88@gmail.com
| Creation: January 2021
| Last modification: July 2021
| Language  : Python 3.8 or >v
| Aeronautical Institute of Technology - Airbus Brazil

"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import json
from datetime import datetime
from framework.utilities.logger import get_logger
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])

def write_optimal_results(individual, airports_keys, distances, demands, profit, DOC_ik, vehicle, kpi_df2, airplanes_ik):
    """
    Description:
        - This function create a txt file containing principal results
    Inputs:
        - profit - [US$]
        - DOC_ik - doc matrix [US$]
        - vehicle - dictionary containing aircraft parameters
        - kpi_df2 - dictionary containing network oprimization parameters
    Outputs:
        - txt file - output information text file
    """

    log.info('==== Start writing aircraft results ====')

    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    fuselage = vehicle['fuselage']
    cabine = vehicle['cabine']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    winglet = vehicle['winglet']

    engine = vehicle['engine']
    pylon = vehicle['pylon']


    results = vehicle['results']
    performance = vehicle['performance']
    operations = vehicle['operations']

    # Creating data for output
    active_arcs = kpi_df2['active_arcs'].sum()
    number_aircraft = kpi_df2['aircraft_number'].sum()
    average_cruise_mach = kpi_df2['mach_tot_aircraft'].sum()/number_aircraft
    total_fuel = kpi_df2['total_fuel'].sum()
    total_CO2 = total_fuel*3.15
    total_distance = kpi_df2['total_distance'].sum()
    total_pax = results['covered_demand']
    CO2_efficiency = 3.15*total_fuel/(total_pax*total_distance*1.852)
    total_cost = results['total_cost']
    total_revenue = results['total_revenue']
    total_profit = results['profit']
    margin_percent = 100*(total_profit/total_revenue)
    average_DOC = kpi_df2['DOC_nd']
    average_DOC = average_DOC[average_DOC > 0].mean()
    average_distance = kpi_df2['active_arcs']*kpi_df2['distances']
    average_distance = average_distance[average_distance > 0].mean()

    average_CEMV = kpi_df2['total_CEMV']
    average_CEMV = average_CEMV[average_CEMV > 0].mean()
    
    number_aircraft2 = np.round(((kpi_df2['total_time'].sum()))/(operations['maximum_daily_utilization']*60))
    
    REV = 1.1*total_pax*operations['average_ticket_price']
    COST = 1.2*total_cost
    RASK = REV/(total_pax*total_distance)
    CASK = COST/(total_pax*total_distance)
    NP = RASK-CASK

    # write string one by one adding newline
    with open(r'Database/Results/Aircrafts/acft_' + str(profit) + '_' + str(start_time) +'.txt','w') as output:
    # with open('Database/Results/Aircrafts/acft_' + str(profit) +'.txt','a') as output:
        output.write(
            '======== Aircraft and network optimization results ========')
        output.write('\n\n')

        output.write('Individual variables: ')
        output.write('\n')
        output.write(str(individual) + "\n")

        # ===============================================================================
        output.write('\n ----- Aircraft parameters ----- \n')

        output.write(
            'Operational Empty Weight: ' + str("{:.2f}".format(float(aircraft['operational_empty_weight']))) + ' [kg] \n')
        output.write(
            'Maximum Takeoff Weight: ' + str("{:.2f}".format(float(aircraft['maximum_takeoff_weight']))) + ' [kg] \n')
        output.write(
            'Maximum Landing Weight: ' + str("{:.2f}".format(float(aircraft['maximum_landing_weight']))) + ' [kg] \n')
        output.write(
            'Maximum Zero Fuel Weight: ' + str("{:.2f}".format(float(aircraft['maximum_zero_fuel_weight']))) + ' [kg] \n')
        output.write(
            'Maximum Fuel Weight: ' + str("{:.2f}".format(float(wing['fuel_capacity']))) + ' [kg] \n')

        output.write('\n Performance: \n')

        output.write(
            'RANGE: ' + str("{:.2f}".format(performance['range'])) + ' [nm] \n')
        output.write(
            'MMO: ' + str("{:.2f}".format(operations['mach_maximum_operating'])) + '\n')
        output.write(
            'VMO: ' + str("{:.2f}".format(operations['max_operating_speed'])) + ' [kts] \n')
        output.write(
            'Ceiling: ' + str("{:.2f}".format(operations['max_ceiling'])) + ' [ft] \n')

        output.write('\n Fuselage: \n')

        output.write(
            'Pax: ' + str("{:.2f}".format(aircraft['passenger_capacity'])) + '\n')
        output.write(
            'Crew: ' + str("{:.2f}".format(aircraft['crew_number'])) + '\n')
        output.write('Aisles number: ' +
                     str("{:.2f}".format(fuselage['aisles_number'])) + '\n')
        output.write('Seats number: ' +
                     str("{:.2f}".format(fuselage['seat_abreast_number'])) + '\n')
        output.write('Seat width: ' +
                     str("{:.2f}".format(cabine['seat_width'])) + ' [m] \n')
        output.write('Seat pitch: ' +
                     str("{:.2f}".format(fuselage['seat_pitch'])) + ' [m] \n')
        output.write('Cabine height: ' +
                     str("{:.2f}".format(fuselage['cabine_height'])) + ' [m] \n')
        output.write('Fuselage length: ' +
                     str("{:.2f}".format(fuselage['length'])) + ' [m] \n')
        output.write('Cockpit length: ' +
                     str("{:.2f}".format(fuselage['cockpit_length'])) + ' [m] \n')
        output.write('Tail length: ' +
                     str("{:.2f}".format(fuselage['tail_length'])) + ' [m] \n')
        output.write(
            'Width: ' + str("{:.2f}".format(fuselage['width'])) + ' [m] \n')
        output.write('Diameter: ' +
                     str("{:.2f}".format(fuselage['diameter'])) + ' [m] \n')
        output.write(
            'Height: ' + str("{:.2f}".format(fuselage['height'])) + ' [m] \n')
        output.write('Dz floor: ' +
                     str("{:.2f}".format(fuselage['Dz_floor'])) + ' [m] \n')
        output.write('Wetted area: ' +
                     str("{:.2f}".format(fuselage['wetted_area'])) + ' [m2] \n')
        output.write(
            'Weight: ' + str(fuselage['weight']) + ' [kg] \n')

        output.write('\n Aerodynamics: \n')

        output.write(
            'CLmax: ' + str("{:.2f}".format(aircraft['CL_maximum_clean'])) + '\n')
        output.write(
            'CLmax TO: ' + str("{:.2f}".format(aircraft['CL_maximum_takeoff'])) + '\n')
        output.write(
            'CLmax LD: ' + str("{:.2f}".format(aircraft['CL_maximum_landing'])) + '\n')

        output.write('\n Wing: \n')

        output.write(
            'Area: ' + str("{:.2f}".format(wing['area'])) + ' [m2] \n')
        output.write('Span: ' + str("{:.2f}".format(wing['span'])) + ' [m] \n')
        output.write('Aspect Ratio: ' +
                     str("{:.2f}".format(wing['aspect_ratio'])) + '\n')
        output.write('Taper Ratio: ' +
                     str("{:.2f}".format(wing['taper_ratio'])) + '\n')
        output.write('Sweep c/4: ' +
                     str("{:.2f}".format(wing['sweep_c_4'])) + ' [deg] \n')
        output.write(
            'Sweep LE: ' + str("{:.2f}".format(wing['sweep_leading_edge'])) + ' [deg] \n')
        output.write(
            'Twist: ' + str("{:.2f}".format(wing['twist'])) + ' [deg] \n')
        output.write('Wetted area: ' +
                     str("{:.2f}".format(wing['wetted_area'])) + ' [m2] \n')
        output.write('Kink position: ' +
                     str("{:.2f}".format(wing['semi_span_kink'])) + ' [%] \n')
        output.write('Root incidence: ' +
                     str("{:.2f}".format(wing['root_incidence'])) + ' [deg] \n')
        output.write('Kink incidence: ' +
                     str("{:.2f}".format(wing['kink_incidence'])) + ' [deg] \n')
        output.write('Tip incidence: ' +
                     str("{:.2f}".format(wing['tip_incidence'])) + ' [deg] \n')
        output.write('Root t/c: ' +
                     str("{:.2f}".format(wing['thickness_ratio'][0])) + '\n')
        output.write('Kink t/c: ' +
                     str("{:.2f}".format(wing['thickness_ratio'][1])) + '\n')
        output.write(
            'Tip t/c: ' + str("{:.2f}".format(wing['thickness_ratio'][2])) + '\n')
        output.write('Center chord: ' +
                     str("{:.2f}".format(wing['center_chord'])) + ' [m] \n')
        output.write('Root chord: ' +
                     str("{:.2f}".format(wing['root_chord'])) + ' [m] \n')
        output.write('Kink chord: ' +
                     str("{:.2f}".format(wing['kink_chord'])) + ' [m] \n')
        output.write('Tip chord: ' +
                     str("{:.2f}".format(wing['tip_chord'])) + ' [m] \n')
        output.write(
            'MAC: ' + str("{:.2f}".format(wing['mean_aerodynamic_chord'])) + ' [m] \n')
        output.write('Leading edge xposition: ' +
                     str(wing['leading_edge_xposition']) + ' [m] \n')
        output.write('Slat presence: ' +
                     str("{:.2f}".format(aircraft['slat_presence'])) + '\n')
        output.write('Flap span: ' +
                     str("{:.2f}".format(wing['flap_span'])) + ' [%] \n')
        output.write('Flap area: ' +
                     str("{:.2f}".format(wing['flap_area'])) + ' [m2] \n')
        output.write('Flap def. TO: ' +
                     str("{:.2f}".format(wing['flap_deflection_takeoff'])) + ' [deg] \n')
        output.write('Flap def. LD: ' +
                     str("{:.2f}".format(wing['flap_deflection_landing'])) + ' [deg] \n')
        output.write('Aileron position: ' +
                     str("{:.2f}".format(wing['aileron_position'])) + ' [%] \n')
        output.write('Rear spar position: ' +
                     str("{:.2f}".format(wing['rear_spar_ref'])) + ' [%] \n')

        output.write('\n Vertical tail: \n')

        output.write(
            'Area: ' + str(vertical_tail['area']) + ' [m2] \n')
        output.write('Aspect Ratio: ' +
                     str("{:.2f}".format(vertical_tail['aspect_ratio'])) + '\n')
        output.write('Taper Ratio: ' +
                     str("{:.2f}".format(vertical_tail['taper_ratio'])) + '\n')
        output.write(
            'Sweep c/4: ' + str("{:.2f}".format(vertical_tail['sweep_c_4'])) + ' [deg] \n')

        output.write('\n Horizontal tail: \n')

        output.write(
            'Area: ' + str(horizontal_tail['area']) + ' [m2] \n')
        output.write('Aspect Ratio: ' +
                     str("{:.2f}".format(horizontal_tail['aspect_ratio'])) + '\n')
        output.write(
            'Taper Ratio: ' + str("{:.2f}".format(horizontal_tail['taper_ratio'])) + '\n')
        output.write(
            'Sweep c/4: ' + str("{:.2f}".format(horizontal_tail['sweep_c_4'])) + ' [deg] \n')

        output.write('\n Winglet: \n')

        output.write('Aspect Ratio: ' +
                     str("{:.2f}".format(winglet['aspect_ratio'])) + '\n')
        output.write('Taper Ratio: ' +
                     str("{:.2f}".format(winglet['taper_ratio'])) + '\n')
        output.write('Sweep leading edge: ' +
                     str("{:.2f}".format(winglet['sweep_leading_edge'])) + ' [deg] \n')

        output.write('\n Engine: \n')
        output.write('Maximum thrust: ' +
                     str(engine['maximum_thrust']*aircraft['number_of_engines']) + ' [N] \n')
        output.write('Bypass ratio: ' +
                     str("{:.2f}".format(engine['bypass'])) + '\n')
        output.write('Fan diameter: ' +
                     str("{:.2f}".format(engine['fan_diameter'])) + ' [m] \n')
        output.write('Fan pressure ratio: ' +
                     str("{:.2f}".format(engine['fan_pressure_ratio'])) + '\n')
        output.write('Compressor pressure ratio: ' +
                     str("{:.2f}".format(engine['compressor_pressure_ratio'])) + '\n')
        output.write('Turbine inlet temperature: ' +
                     str("{:.2f}".format(engine['turbine_inlet_temperature'])) + ' [deg C] \n')
        output.write('Engine length: ' +
                     str("{:.2f}".format(engine['length'])) + ' [m] \n')

        output.write('\n Pylon: \n')

        output.write('Wetted area: ' +
                     str("{:.2f}".format(pylon['wetted_area'])) + ' [m2] \n')

        output.write('\n Aircraft: \n')

        output.write('Wing position: ' +
                     str("{:.2f}".format(wing['position'])) + '\n')
        output.write('Horizontal tail position: ' +
                     str("{:.2f}".format(horizontal_tail['position'])) + '\n')

        output.write('Engine position: ' +
                     str("{:.2f}".format(engine['position'])) + '\n')
        output.write(
            'Wetted area: ' + str(aircraft['wetted_area']) + ' [m2] \n')
    

        output.write('\n ----- Network parameters ----- \n')

        output.write(
            'Number of nodes: ' + str("{:.2f}".format(results['nodes_number'])) + ' \n')
        output.write(
            'Number of arcs: ' + str("{:.2f}".format(float(active_arcs))) + ' \n')
        output.write(
            'Average degree of nodes: ' + str("{:.2f}".format(float(results['avg_degree_nodes']))) + ' \n')
        output.write(
            'Average path length: ' + str("{:.2f}".format(float(average_distance))) + ' \n')
        output.write(
            'Network density: ' + str("{:.2f}".format(float(results['network_density']))) + ' \n')
        output.write(
            'Average clustering: ' + str("{:.2f}".format(float(results['average_clustering']))) + '\n')


        output.write('\nReferemce values: \n')

        # output.write(
        #     'Mach: ' + str("{:.2f}".format(0)) + ' \n')
        # output.write(
        #     'Range: ' + str("{:.2f}".format(0)) + ' [nm] \n')
        # output.write(
        #     'DOC: ' + str("{:.2f}".format(0)) + ' [$/nm] \n')
        # output.write(
        #     'Passengers: ' + str("{:.2f}".format(0)) + ' \n')
        # output.write(
        #     'Net present value: ' + str("{:.2f}".format(0)) + ' [$] \n')
        # output.write(
        #     'Price: ' + str("{:.2f}".format(0)) + ' [$] \n')
        # output.write(
        #     'Average fare: ' + str("{:.2f}".format(0)) + ' [$] \n')
        # output.write(
        #     'Average load factor: ' + str("{:.2f}".format(0)) + ' \n')
        # output.write(
        #     'Average market share: ' + str("{:.2f}".format(0)) + ' [%] \n')

        # output.write('Airports array: ' + str(airport_departure['array']) + "\n")

        # market_share = operations['market_share']

        # demand_db = pd.read_csv('Database/Demand/demand.csv')
        # demand_db = round(market_share*(demand_db.T))
        # output.write('\nDaily demand: \n')
        # np.savetxt(output, demand_db.values, fmt='%d')

        # distances_db = pd.read_csv('Database/Distance/distance.csv')
        # distances_db = (distances_db.T)
        # output.write('\nDistances: \n')
        # np.savetxt(output, distances_db.values, fmt='%d')
        # demand_db = pd.read_csv('Database/Demand/demand.csv')
        # demand_db = round(market_share*(demand_db.T))
        demand_array1 = []
        for i in range(len(airports_keys)):
        	demand_array2 = []
        	for j in range(len(airports_keys)):
        		demand_array2.append(demands[airports_keys[i]][airports_keys[j]]['demand'])
        	demand_array1.append(demand_array2)
        output.write('\nDaily demand: \n')
        np.savetxt(output, demand_array1, fmt='%d')

        # distances_db = pd.read_csv('Database/Distance/distance.csv')
        # distances_db = (distances_db.T)
        distances_array1 = []
        for i in range(len(airports_keys)):
        	distances_array2 = []
        	for j in range(len(airports_keys)):
        		distances_array2.append(distances[airports_keys[i]][airports_keys[j]])
        	distances_array1.append(distances_array2)

        output.write('\nDistances: \n')
        np.savetxt(output, distances_array1, fmt='%d')

        # headings_db = pd.read_csv('Database/distabces/distance.csv')
        # headings_db = (headings_db.T)
        output.write('\nHeadings: \n')
        # np.savetxt(output, headings_db.values, fmt='%d')

        output.write('\nDOC: \n')
        for key in DOC_ik.keys():
            output.write("%s,%s\n\n"%(key,DOC_ik[key]))

        frequencies_db = np.load('Database/Network/frequencies.npy',allow_pickle='TRUE').item()
        # frequencies = pd.DataFrame(frequencies_db, index=False, header=False )

        frequencies = np.array(frequencies_db)
        
        # frequencies_db = pd.read_csv('Database/Network/frequencies.npy')
        # frequencies_db = (frequencies_db.T)
        output.write('\n\nFrequencies: \n')
        # np.savetxt(output, frequencies_db.values, fmt='%d')

        output.write(str(airplanes_ik) + "\n")

        output.write('\nNetwork Results: \n')

        output.write(
            'Average Cruise Mach: ' + str("{:.2f}".format(average_cruise_mach)) + ' \n')
        output.write(
            'Total fuel [kg]: ' + str("{:.2f}".format(total_fuel)) + ' \n')
        output.write(
            'Total CO2 [kg]: ' + str("{:.2f}".format(total_CO2)) + ' \n')
        output.write(
            'CO2 efficiency [kg/pax.nm]: ' + str("{:.8f}".format(CO2_efficiency)) + ' \n')
        output.write(
            'Average CEMV [kg/nm]: ' + str("{:.8f}".format(average_CEMV)) + ' \n')
        output.write(
            'Total distance [nm]: ' + str("{:.2f}".format(total_distance)) + ' \n')
        output.write(
            'Total pax: ' + str("{:.2f}".format(total_pax)) + ' \n')
        output.write(
            'Total cost [$]: ' + str("{:.2f}".format(total_cost)) + ' \n')
        output.write(
            'Total revenue [$]: ' + str("{:.2f}".format(total_revenue)) + ' \n')
        output.write(
            'Total profit [$]: ' + str("{:.2f}".format(total_profit)) + ' \n')
        output.write(
            'Margin percent [%]: ' + str("{:.2f}".format(margin_percent)) + ' \n')
        output.write(
            'Average DOC [$/nm]: ' + str("{:.2f}".format(average_DOC)) + ' \n')
        output.write(
            'NRASK [$/pax.nm]x1E-4: ' + str("{:.2f}".format(RASK*1E4)) + ' \n')
        output.write(
            'NCASK [$/pax.nm]x1E-4: ' + str("{:.2f}".format(CASK*1E4)) + ' \n')
        output.write(
            'NP [$/pax.nm]x1E-4: ' + str("{:.2f}".format(NP*1E4)) + ' \n')
        
        output.write(
            'Number of frequencies: ' + str("{:.2f}".format(results['number_of_frequencies'])) + ' \n')
        # output.write(
        #     'Number of used aircraft: ' + str("{:.2f}".format(number_aircraft)) + ' \n')
        output.write(
            'Number of used aircraft: ' + str("{:.2f}".format(number_aircraft2)) + ' \n')
        output.write(
            'Sectors per aircraft: ' + str("{:.2f}".format(results['number_of_frequencies']/number_aircraft2)) + ' \n')    

    log.info('==== End writing aircraft results ====')

    return


def write_kml_results(airports, profit, airplanes_ik):
    """
    Description:
        - This function create a kml file containing the optimal network
    Inputs:
        - arrivals - list containing the ICAO code for all the cosidered airports
        - departures - list containing the ICAO code for all the cosidered airports
        - profit - [US$]
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - kml file - kml file to be open in google earth showing the network connections
    """
    log.info('==== Start writing klm results ====')
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    # departures = departures
    # arrivals = arrivals

    # data_airports = pd.read_csv("Database/Airports/airports.csv")
    # frequencies_db = np.load('Database/Network/frequencies.npy',allow_pickle='TRUE').item()
    with open(r'Database/Results/Kml/acft_' + str(profit) + '_' + str(start_time) +'.kml','w') as output:
    # with open(r'Database/Results/Klm/acft_' + str(profit) + '.kml','w') as output:

        output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        output.write('<Document>\n')
        output.write('        <Folder>\n')
        output.write('          <name>NETWORK01</name>\n')
        output.write('              <Style>\n')
        output.write('                <LineStyle>\n')
        output.write('                   <color>ffff0055</color> <width>5</width>\n')
        output.write('                </LineStyle>\n')
        output.write('             </Style>\n')
        # output.write('      <Placemark>\n')

        n = 0
        airports_keys = list(airports.keys())
        for i in range(len(airports)):
            for k in range(len(airports)):
                if (i != k) and (airplanes_ik[(airports_keys[i],airports_keys[k])] > 0):
                    output.write('      <Placemark>\n')
                    output.write('             <LineString>\n')

                    dep_latitude = airports[airports_keys[i]]['latitude']
                    dep_longitude = airports[airports_keys[i]]['longitude']
                    des_latitude = airports[airports_keys[k]]['latitude']
                    des_longitude = airports[airports_keys[k]]['longitude']
                    output.write('                <coordinates>' + str("{:.2f},".format(dep_longitude)) + str("{:.2f},".format(dep_latitude)) + '0,' + str("{:.2f},".format(des_longitude)) + str("{:.2f},".format(des_latitude))+ '0' +'</coordinates>\n')
                    output.write('             </LineString>\n')
                    output.write('     	</Placemark>\n')
                    n = n+1

        output.write('       </Folder>\n')
        output.write('</Document>\n')
        output.write('</kml>\n')

    log.info('==== End writing klm results ====')

    return


def write_newtork_results(profit,dataframe01,dataframe02):
    """
    Description:
        - This function create csv files relating to network results
    Inputs:
        - profit - [US$]
        - dataframe01 - dictionary containg network optimization results
        - dataframe02- dictionary containg network optimization results
    Outputs:
        - csv files - csv containing network results
    """
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    dataframe01.to_csv(r'Database/Results/Network/acft_' + str(profit) + '_' + str(start_time) +'01.csv')
    dataframe02.to_csv(r'Database/Results/Network/acft_' + str(profit) + '_' + str(start_time) +'02.csv')

    return

def write_unfeasible_results(flags,x=None):
    """
    Description:
        - This function a txt file with the results of unfeasible aircrafts (aircrafts
        that didt pass the performance and noise checks)
    Inputs:
        - flags
        - x - vector defining the design variables of the aircraft
    Outputs:
        - txt file
    """
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    with open(r'Database/Results/Aircrafts_unfeasible/acft_' + str(start_time) +'.txt','w') as output:
    # with open('Database/Results/Aircrafts/acft_' + str(profit) +'.txt','a') as output:
        output.write(
            '======== Aircraft parameters ========')
        output.write('\n\n')
        # ===============================================================================
        output.write('\n ----- Aircraft parameters ----- \n')
        output.write(str(x) + "\n")

        output.write('\n ----- Flags ----- \n')
        output.write('\n landing, takeoff, climb second segment, missed approach, cruise, fuel, noise \n')
        output.write(str(flags) + "\n")
    return


def write_bad_results(error,x=None):
    """
    Description:
        - This function a txt file with the results of aircraft that produced
        any kind of error
    Inputs:
        - error
        - x - vector defining the design variables of the aircraft
    Outputs:
        - txt file
    """
    start_time = datetime.today().strftime('%Y-%m-%d-%H%M')

    with open(r'Database/Results/Aircrafts_with_error/acft_' + str(start_time) +'.txt','w') as output:
    # with open('Database/Results/Aircrafts/acft_' + str(profit) +'.txt','a') as output:
        output.write(
            '======== Aircraft parameters ========')
        output.write('\n\n')
        # ===============================================================================
        output.write('\n ----- Aircraft parameters ----- \n')
        output.write(str(x) + "\n")

        output.write('\n ----- Error message ----- \n')
        output.write(str(error) + "\n")
    return
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# from framework.Database.Aircrafts.baseline_aircraft_parameters import *

# write_optimal_results(vehicle,150000)
