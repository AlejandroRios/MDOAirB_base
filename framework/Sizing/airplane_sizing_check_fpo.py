"""
MDOAirB: MDO 

Description:
    - This module performs the sizing and checks of the aicraft. It start calculationg the wetted area of the indivudual,
    then calculates the wing structural layout which provide the wing fuel capacity. A while loop is executed to iterate the 
    dimensions of the tail to adjust the máximum takeoff weight considering the mission with the máximum range. Finally, the 
    regulated takeoff and landing weight is calculated. In order to pass for the next step of the framework, the aircraft 
    should pass the checks of the regulated weights as well as the fuel capacity check.

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
import numpy as np
import pandas as pd
import copy
from framework.Performance.Mission.mission import mission
from framework.Sizing.Geometry.wing_structural_layout_fuel_storage import wing_structural_layout
from framework.Sizing.Geometry.fuselage_sizing import fuselage_cross_section
from framework.Sizing.Geometry.wetted_area import wetted_area
from framework.Sizing.Geometry.fuel_capacity_zero_fidelity import zero_fidelity_fuel_capacity
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN
from framework.Aerodynamics.drag_coefficient_flap import drag_coefficient_flap
from framework.Aerodynamics.drag_coefficient_landing_gear import drag_coefficient_landing_gear
from framework.Aerodynamics.airfoil_parameters import airfoil_parameters
from framework.Performance.Mission.mission_sizing import mission_sizing
from framework.Performance.Engine.engine_performance import turbofan
from framework.Performance.Analysis.second_segment_climb import second_segment_climb
from framework.Performance.Analysis.missed_approach_climb import missed_approach_climb_AEO, missed_approach_climb_OEI
from framework.Performance.Analysis.residual_rate_of_climb import residual_rate_of_climb
from framework.Sizing.Geometry.sizing_landing_gear import sizing_landing_gear
from framework.Sizing.performance_constraints import *
from framework.Noise.Noise_Smith.noise_calculation import noise_calculation
from framework.utilities.plot_simple_aircraft import plot3d
#from framework.utilities.plot_tigl_aircraft import plot3d_tigl
from framework.utilities.logger import get_logger

from datetime import datetime
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])

# Globals
global GRAVITY, m2_to_ft2, lb_to_kg, friction_coefficient

# Constants
GRAVITY = 9.80665


# Convertion factors
m2_to_ft2 = 10.7639
lb_to_kg = 0.453592
deg_to_rad = np.pi/180
ft_to_m = 0.3048
lbf_to_N = 4.448


def airplane_sizing(vehicle,x=None):
    """
    Description:
        - This function performs the sizing and checks of the aicraft.
    Inputs:
        - x - design variables vector
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - status - flag check status: 0 = passed all checks | 1 = one or more checks not passed
        - flags - flags corresponding to performance and noise constraints
        - vehicle - dictionary containing aircraft parameters
    """
    log.info('---- Start aircraft sizing module ----')

    if not isinstance(x, list):
        x = x.tolist()

    # Load nested dictionary vehicle
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    winglet = vehicle['winglet']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    pylon = vehicle['pylon']
    nose_landing_gear = vehicle['nose_landing_gear']
    main_landing_gear = vehicle['main_landing_gear']
    performance = vehicle['performance']
    operations = vehicle['operations']
    airport_departure = vehicle['airport_departure']
    airport_destination = vehicle['airport_destination']

    friction_coefficient = wing['friction_coefficient']

    # Upload dictionary variables with optimization variables input vector x
    if x != None:
        log.info('Current individual vairables: {}'.format(x))
        wing['area'] = x[0]
        wing['aspect_ratio'] = x[1]/10
        wing['taper_ratio'] = x[2]/100
        wing['sweep_c_4'] = x[3]
        wing['twist'] = x[4]
        wing['semi_span_kink'] = x[5]/100
        aircraft['passenger_capacity'] = x[6]
        fuselage['seat_abreast_number'] = x[7]
        performance['range'] = x[8]
        aircraft['winglet_presence'] = 1
        aircraft['slat_presence'] = 1
        horizontal_tail['position'] = 1

        engine['bypass'] = x[9]/10
        engine['fan_diameter'] = x[10]/10
        engine['compressor_pressure_ratio'] = x[11]
        engine['turbine_inlet_temperature'] = x[12]
        engine['fan_pressure_ratio'] = x[13]/10
        engine['design_point_pressure'] = x[14]
        engine['design_point_mach'] = x[15]/100

        engine['position'] = 1
    else:
        wing['area'] = wing['area']
        wing['aspect_ratio'] = wing['aspect_ratio']/10
        wing['taper_ratio'] = wing['taper_ratio']/100
        wing['sweep_c_4'] = wing['sweep_c_4']
        wing['twist'] = wing['twist']
        wing['semi_span_kink'] = wing['semi_span_kink']/100
        aircraft['passenger_capacity'] = aircraft['passenger_capacity']
        fuselage['seat_abreast_number'] = fuselage['seat_abreast_number']
        performance['range'] = performance['range'] 
        # aircraft['winglet_presence'] = aircraft['winglet_presence']
        aircraft['winglet_presence'] = 1
        aircraft['slat_presence'] = aircraft['slat_presence']
        # aircraft['slat_presence'] = 1
        # horizontal_tail['position'] =horizontal_tail['position']
        horizontal_tail['position'] = 1

        engine['bypass'] = engine['bypass']/10
        engine['diameter'] = engine['diameter']/10
        engine['compressor_pressure_ratio'] = engine['compressor_pressure_ratio']
        engine['turbine_inlet_temperature'] = engine['turbine_inlet_temperature'] 
        engine['fan_pressure_ratio'] = engine['fan_pressure_ratio']/10

        engine['position'] = engine['position']

    


    operations['mach_cruise'] = operations['mach_maximum_operating'] - 0.02
    engine['diameter'] = engine['diameter']*1.02  # [m]



    ceiling =  operations['max_ceiling']
    # Aerodynamics parameters
    Cl_max = wing['max_2D_lift_coefficient']
    wing['tip_incidence'] = wing['root_incidence'] + wing['twist']

    # Operations parameters
    aircraft['payload_weight'] = aircraft['passenger_capacity'] * \
        operations['passenger_mass']  # [kg]
    proceed = 0

    # Airframe parameters
    wing['span'] = np.sqrt(
        wing['aspect_ratio']*wing['area'])
    CL_max = 0.9*Cl_max*np.cos(wing['sweep_c_4']*np.pi/180)
    CL_max_clean = CL_max

    # Engine paramters
    if engine['position'] == 0:
        aircraft['number_of_engines'] = 2
        engines_under_wing = 0
    elif engine['position'] == 1:
        aircraft['number_of_engines'] = 2
        engines_under_wing = 2

    if wing['position'] == 2 or engine['position'] == 2:
        horizontal_tail['position'] == 2

    classification = 0
    for i in range(1, fuselage['pax_transitions']-1):
        track = aircraft['passenger_capacity'] - \
            fuselage['transition_points'][i-1]
        if track < 0:
            classification = i
            break

    # Selection of container type
    if classification == 0:
        classification = fuselage['pax_transitions']
    if classification == 1:
        fuselage['container_type'] = 'None'
    elif classification == 2:
        fuselage['container_type'] = 'LD3-45W'
    elif classification == 3:
        fuselage['container_type'] = 'LD3-45'

    # start_time =datetime.now()
    # Fuselage cross section sizing
# try:
    vehicle = fuselage_cross_section(vehicle)
# except:
    # log.error("Error at fuselage_cross_section", exc_info = True)
    
    # end_time = datetime.now()
    # print('Fuselage sizing time: {}'.format(end_time - start_time))
    
    # start_time =datetime.now()
    # Airfoil geometry coefficients
# try:
    vehicle = airfoil_parameters(vehicle)
# except:
    # log.error("Error at airfoil_parameters", exc_info = True)

    
    # end_time = datetime.now()
    # print('airfoil coefficients time: {}'.format(end_time - start_time))


    # Wetted area calculation

    wing['mean_thickness'] = np.mean(wing['thickness_ratio'])
    

    # start_time =datetime.now()
# try:
    (vehicle,
        xutip,
        yutip,
        xltip,
        yltip,
        xukink,
        yukink,
        xlkink,
        ylkink,
        xuroot,
        yuroot,
        xlroot,
        ylroot) = wetted_area(vehicle)

# except:
    # log.error("Error at wetted_area", exc_info = True)

    # end_time = datetime.now()
    # print('wetted area time: {}'.format(end_time - start_time))


    
    # start_time =datetime.now()
    # Wing structural layout sizing
# try:
    vehicle = wing_structural_layout(
        vehicle,
        xutip,
        yutip,
        yltip,
        xukink,
        xlkink,
        yukink,
        ylkink,
        xuroot,
        xlroot,
        yuroot,
        ylroot)
# except:
#     log.error("Error at wing_structural_layout", exc_info = True)

    # end_time = datetime.now()
    # print('wing structural time: {}'.format(end_time - start_time))

    # Estimation of MTOW [kg] by Class I methodology
    # Wetted area in [ft]
    aircraft_wetted_area_ft2 = aircraft['wetted_area'] * m2_to_ft2
    aux1 = (np.log10(aircraft_wetted_area_ft2) - 0.0199)/0.7531
    wt0_initial = 10**aux1  # Initial maximum takeoff weight [lb]
    aircraft['maximum_takeoff_weight'] = wt0_initial * \
        lb_to_kg  # Initial maximum takeoff weiight in  [kg]
    fuselage['diameter'] = np.sqrt(fuselage['width']*fuselage['height'])
    wing['leading_edge_xposition'] = 0.45*fuselage['length']
    fuselage['cabine_length'] = fuselage['length'] - \
        (fuselage['tail_length']+fuselage['cockpit_length'])

    # engine['diameter'] = engine['fan_diameter']*1.1
    engine['maximum_thrust'], _ , vehicle = turbofan(
        0, 0.01, 1, vehicle)
    
    engine_static_trhust = engine['maximum_thrust']*0.95

    # engine['maximum_thrust'] = aircraft['number_of_engines'] * \
    #     0.95 * 16206 * (1**0.8) * lbf_to_N  # Rolls-Royce Tay 650 Thrust[N]

    aircraft['maximum_engine_thrust'] = aircraft['number_of_engines'] * \
        0.95 * engine['maximum_thrust'] * (1**0.8)  # Rolls-Royce Tay 650 Thrust[N]
    aircraft['average_thrust'] = 0.75*aircraft['maximum_engine_thrust'] * \
        ((5 + engine['bypass']) /
            (4 + engine['bypass']))  # [N]  

    # Estimation of wings CDO and induced drag factor k
    CL_1 = 0.4
    # Input for neural network: 0 for CL | 1 for alpha
    switch_neural_network = 0
    alpha_deg = 0
    altitude = 100
    mach = 0.15

    # start_time =datetime.now()
    CD_1, _ = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL_1, alpha_deg, switch_neural_network)
    # end_time = datetime.now()
    # print('neural net call time: {}'.format(end_time - start_time))

    CL_2 = 0.5
    CD_2, _ = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL_2, alpha_deg, switch_neural_network)

    K_coefficient = (CD_1 - CD_2)/(CL_1**2 - CL_2**2)
    wing_CD0 = CD_1 - K_coefficient*(CL_1**2)

    altitude = 0
    CL = 0.45

    # CLalpha derivative estimation
    switch_neural_network = 1
    alpha_deg_1 = 1
    _, CL_out_1 = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL, alpha_deg_1, switch_neural_network)

    alpha_deg_2 = 2
    _, CL_out_2 = aerodynamic_coefficients_ANN(
        vehicle, altitude*ft_to_m, mach, CL, alpha_deg_2, switch_neural_network)

    CL_alpha_deg = (CL_out_2 - CL_out_1)/(alpha_deg_2 - alpha_deg_1)
    CL_alpha_rad = CL_alpha_deg/(np.pi/180)

    # Divergence mach check
    mach = 0.7
    CL_1 = 0.4
    alga_deg = 1
    CD_max = -1000
    CD_min = 1000

    while mach <= operations['mach_maximum_operating']:
        mach = mach + 0.01
        switch_neural_network = 0
        CD_wing, _ = aerodynamic_coefficients_ANN(
            vehicle, ceiling*ft_to_m, mach, CL_1, alpha_deg, switch_neural_network)
        CD_ubrige = friction_coefficient * \
            (aircraft['wetted_area'] - wing['wetted_area']) / \
            wing['area']
        CD_total = CD_wing + CD_ubrige
        CD_max = max(CD_max, CD_total)
        CD_min = min(CD_min, CD_total)

    delta_CD = CD_max - CD_min

    # Definition of takeoff and lanfing maximum CL values
    if aircraft['slat_presence'] == 1:
        aircraft['CL_maximum_takeoff'] = CL_max_clean + 0.6
        aircraft['CL_maximum_landing'] = CL_max_clean + 1.2
    else:
        aircraft['CL_maximum_takeoff'] = CL_max_clean + 0.4
        aircraft['CL_maximum_landing'] = CL_max_clean + 1.0

    nose_landing_gear['xpostion'] = fuselage['cockpit_length'] - 0.4
    MTOW_count = 0
    delta_MTOW = 1E6
    max_MTOW_count = 25
    status = 0

    log.info('---- Start sizing loop for tail sizing ----')

    while (delta_MTOW > 100) and (MTOW_count < max_MTOW_count):
        # start_time =datetime.now()
        # Mission evaluation and tail sizing
    # try:
        vehicle, MTOW_calculated, fuel_mass, landing_weight = mission_sizing(
            vehicle, airport_departure, airport_destination)
    # except:
        # log.error("Error at mission_sizing", exc_info = True)
    # end_time = datetime.now()
    # print('mission evaluation tail sizing call time: {}'.format(end_time - start_time))
        
        delta_MTOW = np.abs(
            MTOW_calculated - aircraft['maximum_takeoff_weight'])

        # This calculation was performed inside mission function and again here. Is that correct?
        aircraft['maximum_takeoff_weight'] = 0.20 * \
            aircraft['maximum_takeoff_weight'] + 0.8*MTOW_calculated

        aircraft['maximum_landing_weight'] = landing_weight
        # engine['maximum_thrust'] = (
        #     0.3264*aircraft['maximum_takeoff_weight'] + 2134.8)*0.95 # Test this

        MTOW_count = MTOW_count + 1  # Counter of number of iterations

    log.info('---- End sizing loop for tail sizing ----')

    # Sizing Checks
    if MTOW_count > max_MTOW_count:
        print('MTOW calculation not converged')
        flag_requirements = 1
        DOC = 100
        delta_fuel = -10000
        takeoff_noise = 400
        status = 1
    else:
        aircraft['zCG'] = -0.80
        flag_requirements = 0

    # try:
        vehicle = sizing_landing_gear(vehicle)
    # except:
    #     log.error("Error at sizing_landing_gear", exc_info = True)
        # fuel deficit

        other_elements_fuel = zero_fidelity_fuel_capacity(vehicle)
        delta_fuel = (other_elements_fuel + wing['fuel_capacity']) - 1.005*fuel_mass

    if delta_fuel < 0:
        flag_fuel = 1
    else:
        flag_fuel = 0



    # Simple plot check
    # plot3d(vehicle)
    # plot3d_tigl(vehicle)

    # Regulated takeoff and landing checks
    maximum_takeoff_weight = aircraft['maximum_takeoff_weight']
    maximum_landing_weight = landing_weight
    aircraft['maximum_zero_fuel_weight'] = maximum_landing_weight*0.98

    # engine['maximum_thrust'] = (
    #     aircraft['number_of_engines']*engine['maximum_thrust']*lb_to_kg)*GRAVITY  # Test this

    ToW = (aircraft['number_of_engines']*engine['maximum_thrust'])/(aircraft['maximum_takeoff_weight']*GRAVITY)
    WoS = aircraft['maximum_takeoff_weight']/wing['area']

    # regulated_takeoff_weight_required = regulated_takeoff_weight(vehicle)
    # regulated_landing_weight_required = regulated_landing_weight(vehicle)


    # Landing field length check
    landing_field_length_required = airport_destination['lda']

    k_L = 0.107

    WtoS_landing = (k_L*airport_destination['lda']*aircraft['CL_maximum_landing'])/(aircraft['maximum_landing_weight']/aircraft['maximum_takeoff_weight'])
    #WtoS_landing = (k_L*airport_destination['lda']*aircraft['CL_maximum_landing'])
    #WtoS_takeoff= WtoS_landing/(maximum_landing_weight/maximum_takeoff_weight)     

    if  WoS > WtoS_landing:
        flag_landing = 1
    else:
        flag_landing = 0

    
    # Takeoff field length check
    k_TO = 2.34

    ToW_takeoff = (k_TO/(airport_departure['tora']*aircraft['CL_maximum_takeoff']))*WoS

    if ToW < ToW_takeoff :
        flag_takeoff = 1
    else:
        flag_takeoff = 0
    
        
    # Climb gradient in the Second segment check
    ToW_second_segment = second_segment_climb(vehicle, airport_departure, aircraft['maximum_takeoff_weight']*GRAVITY)


    if ToW < ToW_second_segment: 
        flag_climb_second_segment = 1
    else:
        flag_climb_second_segment = 0

    
    # Climb gradient during missed approach check
    ToW_missed_approach = missed_approach_climb_OEI(vehicle, airport_destination, aircraft['maximum_takeoff_weight']*GRAVITY,aircraft['maximum_takeoff_weight']*GRAVITY)

    if ToW < ToW_missed_approach:
        flag_missed_approach = 1
    else:
        flag_missed_approach = 0

    
    # Cruise check

    engine_cruise_thrust, _ , vehicle = turbofan(
        ceiling,operations['mach_cruise'], 0.98, vehicle)

    ToW_cruise = residual_rate_of_climb(vehicle, airport_departure, aircraft['maximum_takeoff_weight']*GRAVITY,engine_cruise_thrust)

    if ToW < ToW_cruise:
        flag_cruise = 1
    else:
        flag_cruise = 0



    # Noise check
    delta_CD_flap = drag_coefficient_flap(vehicle)
    CD_ubrige = friction_coefficient * (aircraft['wetted_area'] - wing['wetted_area'])/wing['area']
    CD_main, CD_nose = drag_coefficient_landing_gear(vehicle)
    delta_CD_landing_gear = CD_main+CD_nose
    CD0_landing = wing_CD0 + CD_ubrige + delta_CD_flap + delta_CD_landing_gear

    aircraft['CD0_landing'] = CD0_landing

# try:
#    takeoff_noise, sideline_noise, landing_noise = noise_calculation(vehicle, airport_departure)
# except:
#         log.error("Error at noise_calculation", exc_info = True)

 #   total_noise = takeoff_noise + sideline_noise + landing_noise

    #if total_noise > 280:
    #    flag_noise = 1
    #else:
    #    flag_noise = 0

    flag_noise = 0
    flags = [flag_requirements, flag_landing, flag_takeoff, flag_climb_second_segment, flag_missed_approach, flag_cruise, flag_fuel, flag_noise]

    # flags = [flag_takeoff, flag_landing, flag_fuel]

    # If any of the flags = 1 then the status = 1 and aircraft will not be feasible
    # to continue with the optimization loop
    if max(flags) > 0:
        status = 1
    else:
        status = 0
        
    log.info('Aircraft flags: {}'.format(flags))
    log.info('Aircraft status (pass = 0, no pass =1): {}'.format(status))
    log.info('---- End aircraft sizing module ----')

    return status, flags, vehicle
# =============================================================================
# MAIN
# =============================================================================


def objective_function_FPO(x, original_vehicle, mission_range,
                           computation_mode, airports, distances, demands):

    # Do a copy of original vehicle
    vehicle = copy.deepcopy(original_vehicle)
    # with open('Database/DOC/Vehicle.pkl', 'rb') as f:
    #     vehicle = pickle.load(f)

    print(x)

    DOC_ik = 0
    status = 0
    flags = 2*np.ones(8)
    # =============================================================================
    # Airplane sizing and
    try:
        status, flags, vehicle = airplane_sizing(vehicle, x)
    except:
        log.error(">>>>>>>>>> Error at <<<<<<<<<<<< airplane_sizing",
                  exc_info=True)

    # status = 0
    performance = vehicle['performance']

    fuel_mass = 0
    total_mission_flight_time = 0
    mach = 0
    passenger_capacity = 0
    SAR = 0
    landing_field_length_computed = 0
    takeoff_field_length_computed = 0
    app_speed = 0
    aircraft = vehicle['aircraft']
    distance = 0 
    altitude = 0 
    mass = 0 
    time = 0
    sfc = 0
    thrust = 0
    machv = 0
    CL = 0
    CD = 0
    LoD = 0
    throttle = 0
    vcas = 0

    # =============================================================================
    # If airplane pass checks, status = 0, else status = 1 and profit = 0
    # if status == 0:
    log.info('Aircraft passed sizing and checks status: {}'.format(status))

    # =============================================================================
    log.info('---- Start DOC calculation ----')
     # The DOC is estimated for each city pair and stored in the DOC dictionary

    airports_keys = list(airports.keys())
    try:
            if (distances[airports_keys[0]][airports_keys[1]] <=
                    performance['range']):
                # Update information about orign-destination pair airports:
                airport_departure = {}
                airport_departure['latitude'] = airports[
                    airports_keys[0]]['latitude']
                airport_departure['longitude'] = airports[
                    airports_keys[0]]['longitude']
                #airport_departure['elevation'] = 6000
                airport_departure['elevation'] = airports[
                     airports_keys[0]]['elevation']
                airport_departure['dmg'] = airports[airports_keys[0]]['dmg']
                airport_departure['tref'] = airports[airports_keys[0]]['tref']

                takeoff_runway = airports[airports_keys[0]]['runways'][demands[
                    airports_keys[0]][airports_keys[1]]['takeoff_runway']]

                airport_destination = {}
                airport_destination['latitude'] = airports[
                    airports_keys[1]]['latitude']
                airport_destination['longitude'] = airports[
                    airports_keys[1]]['longitude']
                airport_destination['elevation'] = airports[
                    airports_keys[1]]['elevation']
                airport_destination['dmg'] = airports[airports_keys[1]]['dmg']
                airport_destination['tref'] = airports[
                    airports_keys[1]]['tref']

                landing_runway = airports[airports_keys[1]]['runways'][demands[
                    airports_keys[0]][airports_keys[1]]['landing_runway']]

                # Calculate DOC and mission parameters for origin-destination airports pair:
                mission_range = distances[airports_keys[0]][airports_keys[1]]
                print('Mission range = ', mission_range)
                fuel_mass, total_mission_flight_time, DOC, mach, passenger_capacity, SAR, landing_field_length_computed, takeoff_field_length_computed, app_speed, distance, altitude, mass, time, sfc, thrust, machv, CL, CD, LoD, throttle, vcas = mission(
                    vehicle, airport_departure, takeoff_runway,
                    airport_destination, landing_runway, mission_range)
                DOC_nd = DOC
                DOC_ik = DOC * mission_range

            elif computation_mode == 0:
                DOC_ik = 100000
                log.info(
                    ">>>>>>>>>> Range computed by design is lower than the distance between airports >>>>>>>>>> DOC calculation failed! Performance range =",
                    performance['range'])

            log.info('---- End DOC calculation ----')
    except:
           log.error(">>>>>>>>>> Error at <<<<<<<<<<<< DOC calculation ",
                      exc_info=True)
        # =============================================================================

    return aircraft['maximum_takeoff_weight'], DOC_ik, fuel_mass, total_mission_flight_time, mach, passenger_capacity, SAR, landing_field_length_computed, takeoff_field_length_computed, app_speed, status, flags[0], distance, altitude, mass, time, sfc, thrust, machv, CL, CD, LoD, throttle, vcas, aircraft['operational_empty_weight']
# =============================================================================
# TEST
# =============================================================================

