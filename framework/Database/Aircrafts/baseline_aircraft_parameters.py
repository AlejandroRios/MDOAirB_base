"""
MDOAirB

Description:
    - This function descrive baseline aircraft properties which is used to pass
    information of the aircraft and operations through modules

Outputs:
    - vehicle - dictionary containing aircraft parameters

TODO's:

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def initialize_aircraft_parameters():
    aircraft = {}

    aircraft['aerodynamic_centers_arm_wing_horizontal'] = 0
    aircraft['after_center_of_gravity_xposition'] = 0
    aircraft['xCG'] = 0
    aircraft['yCG'] = 0
    aircraft['zCG'] = 0
    aircraft['CG_position'] = np.array(
        [aircraft['xCG'], aircraft['yCG'], aircraft['zCG']]).transpose()
    aircraft['CL_maximum_clean'] = 1.65
    aircraft['CL_maximum_landing'] = 2.0
    aircraft['CL_maximum_takeoff'] = 2.20
    aircraft['CD0_landing'] = 0
    aircraft['crew_number'] = 5
    aircraft['fixed_equipment_weight'] = 0
    aircraft['forward_center_of_gravity_xposition'] = 0
    aircraft['Ixx'] = 821466
    aircraft['Iyy'] = 3343669
    aircraft['Izz'] = 4056813
    aircraft['Ixy'] = 0
    aircraft['Ixz'] = 178919
    aircraft['Iyz'] = 0
    aircraft['inertia_matrix'] = np.array([[aircraft['Ixx'], -aircraft['Ixy'], -aircraft['Ixz']],
                                        [-aircraft['Ixy'],
                                            aircraft['Iyy'], -aircraft['Iyz']],
                                        [-aircraft['Ixz'], -aircraft['Iyz'], aircraft['Izz']]])

    aircraft['maximum_takeoff_weight'] = 60000
    aircraft['maximum_landing_weight'] = 60000
    aircraft['maximum_zero_fuel_weight'] = 31700 
    aircraft['maximum_fuel_capacity'] = 9428
    aircraft['number_of_engines'] = 2
    aircraft['neutral_point_xposition'] = 0
    aircraft['operational_empty_weight'] = 0
    aircraft['payload_weight'] = 5000
    aircraft['passenger_capacity'] = 78
    aircraft['power_plant_weight'] = 0
    aircraft['static_margin'] = 0.15
    aircraft['structural_weight'] = 0
    aircraft['slat_presence'] = 1
    aircraft['spoiler_presence'] = 1
    aircraft['wetted_area'] = 589.7500  # [m2]
    aircraft['winglet_presence'] = 0
    aircraft['year_of_technology'] = 2017

    wing = {}
    wing['area'] = 100
    wing['aspect_ratio'] = 12  # Fokker = 8.43
    wing['aerodynamic_center_xposition'] = 0
    wing['aerodynamic_center_ref'] = 0.25
    wing['aileron_chord'] = 0
    wing['aileron_surface'] = 0
    wing['aileron_position'] = 0
    wing['center_chord'] = 3.53
    wing['camber_line_angle_leading_edge'] = [0.0787, -0.0295, 0.1000]
    wing['camber_line_angle_trailing_edge'] = [-0.0549, -0.2101, -0.0258]
    wing['camber_at_maximum_thickness_chordwise_position'] = [-0.0006, 0.0028, 0.0109]
    wing['center_of_gravity_xposition'] = 0
    wing['dihedral'] = 3  # [deg]
    wing['engine_position_chord'] = 0
    wing['friction_coefficient'] = 0.003
    wing['flap_deflection_takeoff'] = 35  # [deg]
    wing['flap_deflection_landing'] = 45  # [deg]
    wing['flap_deflection_approach'] =  15  # [deg]
    wing['flap_span'] = 0.75
    wing['flap_area'] = 0
    wing['flap_chord'] = 0
    wing['flap_slots_number'] = 2
    wing['fuel_capacity'] = 0  # [kg]
    wing['kink_chord'] = 3.53
    wing['kink_chord_yposition'] = 0
    wing['kink_incidence'] = 0  # [deg]
    wing['leading_edge_xposition'] = 0
    wing['leading_edge_radius'] = [0.0153, 0.0150, 0.0150]
    wing['mean_aerodynamic_chord_yposition'] = 0
    wing['maximum_thickness_chordwise_position'] = [0.3738, 0.3585, 0.3590]
    wing['maximum_camber'] = [-0.0004, 0.0185, 0.0104]
    wing['maximum_camber_chordwise_position'] = [0.6188, 0.7870, 0.5567]
    wing['mean_aerodynamic_chord'] = 3.53
    wing['mean_thickness'] = 0.11
    wing['max_2D_lift_coefficient'] = 1.9
    wing['position'] = 1
    wing['pylon_position_chord'] = 0
    wing['root_chord'] = 3.53
    wing['root_incidence'] = 2  # [deg]
    wing['root_thickness'] = 0
    wing['rear_spar_ref'] = 0.75
    wing['root_chord_yposition'] = 0
    wing['ribs_spacing'] = 22
    wing['span'] = 30.3579
    wing['semi_span'] = 0
    wing['sweep_c_4'] = 22.6  # [deg]
    wing['sweep_leading_edge'] = 22.6  # [deg]
    wing['sweep_c_2'] = 22.6  # [deg]
    wing['sweep_trailing_edge'] = 22.6  # [deg]
    wing['semi_span_kink'] = 0.34
    wing['tip_chord'] = 3.53
    wing['taper_ratio'] = 0.38
    wing['tip_incidence'] = -2.5  # [deg]
    wing['thickness_ratio'] = [0.12, 0.12, 0.12]
    wing['thickness_line_angle_trailing_edge'] = [-0.0799, -0.1025, -0.1553]
    wing['thickness_to_chord_average_ratio'] = 0.11
    wing['trunnion_xposition'] = 0.75
    wing['trunnion_length'] = 0
    wing['twist'] = 0
    # yc_trunnion
    wing['tank_center_of_gravity_xposition'] = 0
    wing['slat_span'] = 0.75
    wing['slat_area'] = 0
    wing['slat_chord'] = 0
    wing['slat_slots_number'] = 1
    wing['wetted_area'] = 168.6500  # [m2]
    wing['weight'] = 0


    horizontal_tail = {}
    horizontal_tail['position'] = 1
    horizontal_tail['area'] = 35  # [m2]
    horizontal_tail['aspect_ratio'] = 4.35
    horizontal_tail['taper_ratio'] = 0.4
    horizontal_tail['sweep_c_4']  = 1
    horizontal_tail['sweep_c_2']  = 1
    horizontal_tail['sweep_leading_edge'] = 0
    horizontal_tail['sweep_trailing_edge'] = 0
    horizontal_tail['volume'] = 0.9
    horizontal_tail['aerodynamic_center'] = 0.25
    horizontal_tail['aerodynamic_center_ref'] = 0.25
    horizontal_tail['aerodynamic_center_xposition'] = 0
    horizontal_tail['mean_chord'] = 1
    horizontal_tail['tip_chord'] = 1
    horizontal_tail['root_chord'] = 1
    horizontal_tail['thickness_root_chord'] = 0.1
    horizontal_tail['thickness_tip_chord'] = 0.1
    horizontal_tail['mean_chord_thickness']  = 0.1
    horizontal_tail['center_chord'] = 1
    horizontal_tail['tail_to_wing_area_ratio'] = 0
    horizontal_tail['twist'] = 0
    horizontal_tail['span'] = 0
    horizontal_tail['dihedral'] = 1
    horizontal_tail['mean_aerodynamic_chord'] = 1
    horizontal_tail['mean_geometrical_chord'] = 1
    horizontal_tail['mean_aerodynamic_chord_yposition'] = 0
    horizontal_tail['tau'] = 1
    horizontal_tail['weight'] = 0
    horizontal_tail['wetted_area'] = 0 
    horizontal_tail['center_of_gravity_xposition'] = 0
    horizontal_tail['leading_edge_xposition'] = 0


    vertical_tail = {}
    vertical_tail['area'] = 25  # [m2]
    vertical_tail['aspect_ratio'] = 1.2
    vertical_tail['taper_ratio'] = 0.5
    vertical_tail['sweep_c_4']  = 41
    vertical_tail['volume'] = 0.09
    vertical_tail['aerodynamic_center'] = 0.25
    vertical_tail['aerodynamic_center_ref'] = 0.25
    vertical_tail['aerodynamic_center_xposition'] = 0
    vertical_tail['dorsalfin_wetted_area'] = 0.1
    vertical_tail['twist'] = 0
    vertical_tail['dihedral'] = 90
    vertical_tail['center_chord'] = 1
    vertical_tail['tip_chord'] = 1
    vertical_tail['root_chord'] = 1
    vertical_tail['span'] = 1
    vertical_tail['mean_aerodynamic_chord'] = 1
    vertical_tail['mean_geometrical_chord'] = 1
    vertical_tail['sweep_c_2'] = 0
    vertical_tail['sweep_leading_edge'] = 0
    vertical_tail['sweep_trailing_edge'] = 0
    vertical_tail['thickness_ratio'] = [0.11, 0.11]
    vertical_tail['weight'] = 0
    vertical_tail['wetted_area'] = 0
    vertical_tail['mean_aerodynamic_chord_yposition'] = 0
    vertical_tail['mean_thickness'] = 0
    vertical_tail['center_of_gravity_xposition'] = 0
    vertical_tail['leading_edge_xposition'] = 0

    winglet = {}
    winglet['aspect_ratio'] = 2.75
    winglet['taper_ratio'] = 0.25
    winglet['sweep_leading_edge'] = 35
    winglet['cant_angle'] = 75
    winglet['cant_angle'] = 75
    winglet['weight'] = 0
    winglet['wetted_area'] = 0
    winglet['root_chord'] = 1
    winglet['span'] = 0
    winglet['thickess'] = 0
    winglet['area'] = 0
    winglet['tau'] = 0
    winglet['center_of_gravity_xposition'] = 0

    fuselage = {}
    fuselage['aisles_number'] = 1
    fuselage['seat_abreast_number'] = 4
    fuselage['cabine_height'] = 2
    fuselage['aisle_width'] = 0.5
    fuselage['seat_pitch'] = 0.8128
    fuselage['height_to_width_ratio'] = 1.1
    fuselage['pax_transitions'] = 3
    fuselage['transition_points'] = [75, 95]
    fuselage['width'] = 4 
    fuselage['height'] = 4 
    fuselage['length'] = 0
    fuselage['cabine_length'] = 0
    fuselage['nose_length'] = 1.64
    fuselage['cockpit_length'] = 3.7  # [m]
    fuselage['tail_length'] = 0
    fuselage['Dz_floor'] = 4
    fuselage['minimum_width'] = 3
    fuselage['container_type'] = 'LD3-45'
    fuselage['wetted_area'] = 0
    fuselage['diameter'] = 0
    fuselage['weight'] = 0
    fuselage['center_of_gravity_xposition'] = 0
    fuselage['af_ellipse'] = 0.25  # [m]
    fuselage['bf_ellipse'] = 0.30  # [m]


    cabine = {}
    cabine['armrest_top'] = 32  # [inch]
    cabine['armrest_bottom'] = 7  # [inch]
    cabine['armrest_width'] = 2  # [inch]
    cabine['seat_cushion_thickness_YC'] = 0.14  # [m]
    cabine['seat_width'] = 0.46  # [m]
    cabine['backrest_height'] = 0.59  # [m]
    cabine['floor_thickness'] = 0.117  # [m]
    cabine['pax_distance_head_wall'] = 0.06  # [m]
    cabine['pax_distance_shoulder_wall'] = 0.04  # [m]
    cabine['pax_shoulder_breadth'] = 0.53  # [m]
    cabine['pax_eye_height'] = 0.87  # [m]
    cabine['pax_midshoulder_height'] = 0.70  # [m]
    cabine['delta_z_symmetry_inferior'] = -1
    cabine['delta_z_symmetry_superior'] = 2
    cabine['seat_delta_width_floor'] = 0.025
    cabine['seat_prof'] = 0.48
    cabine['toillet_prof'] = 1.5
    cabine['galley_prof'] = 1.1

    engine = {}
    engine['diameter'] = 1
    engine['fan_diameter'] = 1

    engine['bypass'] = 5.0
    engine['fan_pressure_ratio'] = 1.46
    engine['compressor_pressure_ratio'] = 28.5
    engine['turbine_inlet_temperature'] = 1450
    engine['design_point_pressure'] = 33000
    engine['design_point_mach'] = 0.82
    engine['position'] = 1
    engine['yposition'] = 1
    engine['maximum_thrust'] = 0
    engine['wetted_area'] = 0
    engine['length'] = 0
    engine['weight'] = 0
    engine['center_of_gravity_xposition'] = 0
    engine['fan_rotation_ref'] = 4952
    engine['compressor_rotation_ref'] = 14950
    engine['fan_rotation'] = 0
    engine['compressor_rotation'] = 0
    engine['T0'] = 0
    engine['T1'] = 0
    engine['T2'] = 0
    engine['type'] = 0 # 0 turbofan | 1 Turboprop


    engine['performance_parameters'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,0])
    engine['total_pressures'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    engine['total_temperatures'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    engine['exit_areas'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    engine['fuel_flows'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    engine['gas_exit_speeds'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    engine['rotation_speeds'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    nacelle = {}
    nacelle['weight'] = 0
    nacelle['center_of_gravity_xposition'] = 0

    pylon = {}
    pylon['wetted_area'] = 0
    pylon['thickness_ratio'] = [0.1, 0.1]
    pylon['area'] = 0
    pylon['taper_ratio'] = 0
    pylon['length'] = 0
    pylon['mean_geometrical_chord'] = 0
    pylon['mean_aerodynamic_chord'] = 0
    pylon['span'] = 0
    pylon['xposition'] = 0
    pylon['sweep_leading_edge'] = 0

    pylon['sweep_leading_c_4'] = 0
    pylon['aspect_ratio'] = 2
    pylon['center_of_gravity_xposition'] = 0


    nose_landing_gear = {}
    nose_landing_gear['pressure'] = 190
    nose_landing_gear['xpostion'] = 0
    nose_landing_gear['weight'] = 0
    nose_landing_gear['tyre_diameter'] = 0.80
    nose_landing_gear['tyre_width'] = 0
    nose_landing_gear['tyre_height'] = 0.25
    nose_landing_gear['trunnion_length'] = 1.3  # [m]
    nose_landing_gear['piston_length'] = 0
    nose_landing_gear['piston_diameter'] = 0
    nose_landing_gear['center_of_gravity_xposition'] = 0
    nose_landing_gear['unit_wheels_number'] = 2

    main_landing_gear = {}
    main_landing_gear['pressure'] = 200
    main_landing_gear['xpostion'] = 0
    main_landing_gear['weight'] = 0
    main_landing_gear['tyre_diameter'] = 0
    main_landing_gear['tyre_width'] = 0
    main_landing_gear['piston_length'] = 0
    main_landing_gear['piston_diameter'] = 0
    main_landing_gear['center_of_gravity_xposition'] = 0
    main_landing_gear['unit_wheels_number'] = 2

    systems = {}
    systems['fuel_weight'] = 0
    systems['propulsion_weight'] = 0
    systems['flight_control_weight'] = 0
    systems['fixed_equipment_weight'] = 0
    systems['hydraulic_weight'] = 0
    systems['electrical_weight']  = 0
    systems['avionics_weight'] = 0
    systems['air_weight'] = 0 
    systems['oxygen_weight'] = 0
    systems['APU_weight'] = 0
    systems['furnishing_weight'] = 0
    systems['paint_weight'] = 0
    systems['handling_gear'] = 0
    systems['safety'] = 0


    performance = {}
    performance['range'] = 1600  # Aircraft range [nm]
    performance['residual_rate_of_climb'] = 300

    operations = {}
    operations['computation_mode'] = 0
    operations['route_computation_mode'] = 0
    operations['takeoff_field_length'] = 2000
    operations['landing_field_length'] = 1500
    operations['descent_altitude'] = 1500
    operations['time_between_overhaul'] = 2500
    operations['taxi_fuel_flow_reference'] = 5
    operations['cruise_altitude'] = 30000

    operations['climb_V_cas'] = 280
    operations['mach_climb'] = 0.78
    operations['cruise_V_cas'] = 310
    operations['descent_V_cas'] = 310
    operations['mach_descent'] = 0.78
    operations['mach_cruise_alternative'] = 0.78

    operations['mach_maximum_operating'] = 0.82
    operations['mach_cruise'] = 0.72
    operations['max_operating_speed'] = 340
    operations['holding_time'] = 30  # [min]
    operations['alternative_airport_distance'] = 100  # [nm]
    operations['max_ceiling'] = 41000
    operations['passenger_mass'] = 110  # [kg]
    operations['reference_load_factor'] = 0.85
    operations['buffet_margin'] = 1.3
    operations['fuel_density'] = 0.81
    operations['contingency_fuel_percent'] = 0.1
    operations['min_cruise_time'] = 3
    operations['fuel_price_per_kg'] = 2.8039
    operations['average_ticket_price'] = 120
    operations['market_share'] = 0.1
    operations['go_around_allowance'] = 300
    operations['takeoff_allowance'] = 300
    operations['approach_allowance_mass'] = 150
    operations['average_taxi_in_time'] = 5
    operations['average_taxi_out_time'] = 10
    operations['landing_time_allowance'] = 3
    operations['takeoff_time_allowance'] = 2
    operations['turn_around_time'] = 45
    operations['maximum_daily_utilization'] = 13
    operations['flight_planning_delta_ISA'] = 0  # [deg C]

    operations['average_departure_delay'] = 28
    operations['average_arrival_delay'] = 29.9

    noise = {}
    noise['takeoff_lambda'] = 0
    noise['takeoff_k1'] = 1.1
    noise['takeoff_k2'] = 1.2
    noise['takeoff_time_1'] = 3.0
    noise['takeoff_obstacle_altitude'] = 35
    noise['takeoff_time_step'] = 0.5
    noise['takeoff_time_2'] = 3.0
    noise['takeoff_trajectory_max_distance'] = 10000

    noise['landing_gamma'] = -3
    noise['landing_CL_3P'] = 0.3
    noise['landing_CD_3P'] = 0.08
    noise['landing_mu_roll'] = 0.03
    noise['landing_mu_brake'] = 0.3
    noise['landing_transition_time'] = 1.0
    noise['landing_load_factor_flare'] = 1.1

    noise['aircraft_parameters_CL_3P'] = 0.3
    noise['aircraft_parameters_CL_air'] = 1.65
    noise['aircraft_parameters_CD_3P'] = 0.08
    noise['aircraft_parameters_CD_air_LG_down'] = 0.11
    noise['aircraft_parameters_CD_air_LG_up'] = 0.081

    noise['aircraft_geometry_fuselage_surface'] = 21.1
    noise['aircraft_geometry_fuselage_length'] = 22.28
    noise['aircraft_geometry_main_landing_gear_number'] = 2 
    noise['aircraft_geometry_nose_landing_gear_number'] = 1

    noise['aircraft_geometry_main_landing_gear_length'] = 1.88
    noise['aircraft_geometry_nose_landing_gear_length'] = 1.21
    noise['aircraft_geometry_main_landing_gear_wheels'] = 2
    noise['aircraft_geometry_nose_landing_gear_wheels'] = 2
    noise['aircraft_geometry_wing_flap_type1_position'] = 1
    noise['aircraft_geometry_wing_flap_type2_position'] = 0
    noise['aircraft_geometry_slats_position'] = 1
    noise['aircraft_geometry_slots_number'] = 2
    noise['aircraft_geometry_main_landing_gear_position'] = 1
    noise['aircraft_geometry_nose_landing_gear_position'] = 1
    noise['aircraft_geometry_altitude_retracted'] = 0
    noise['aircraft_geometry_delta_ISA_retracted'] = 0

    noise['engine_parameters_throttle_position'] = 1
    noise['engine_parameters_fan_rotation'] = 4952
    noise['engine_parameters_compressor_rotation'] = 14950

    noise['runaway_parameters_mu_roll'] = 0.03
    noise['runaway_parameters_mu_brake'] = 0.3

    noise['relative_humidity'] = 70
    noise['landing_lateral_distance_mic'] = 1
    noise['sideline_lateral_distance_mic'] = 450
    noise['takeoff_lateral_distance_mic'] = 1
    noise['landing_longitudinal_distance_mic'] = 2000
    noise['sideline_longitudinal_distance_mic'] = 0
    noise['takeoff_longitudinal_distance_mic'] = 6500

    airport_departure = {}
    airport_departure['elevation'] = 0*3.28084  # [m]
    airport_departure['tora'] = 2500  # [m]
    airport_departure['tref'] = 0  # [deg C]
 
    airport_destination = {}
    airport_destination['elevation'] = 0*3.28084  # [m]
    airport_destination['lda'] = 2000  # [m]
    airport_destination['tref'] = 0  # [deg C]

    aircraft['maximum_engine_thrust'] = 67189  # Rolls-Royce Tay 650 Thrust[N]
    aircraft['average_thrust'] = 67189

    results = {}
    results['profit'] = 0
    results['total_cost'] = 0
    results['total_revenue'] = 0
    results['nodes_number'] = 10
    results['arcs_number'] = 0
    results['avg_degree_nodes'] = 0
    results['network_density'] = 0
    results['average_clustering'] = 0
    results['covered_demand'] = 0
    results['aircrafts_used'] = 0
    results['number_of_frequencies'] = 0

    vehicle = {}
    vehicle['aircraft'] = aircraft
    vehicle['wing'] = wing
    vehicle['horizontal_tail'] = horizontal_tail
    vehicle['vertical_tail'] = vertical_tail
    vehicle['winglet'] = winglet
    vehicle['fuselage'] = fuselage
    vehicle['cabine'] = cabine
    vehicle['engine'] = engine
    vehicle['pylon'] = pylon
    vehicle['nacelle'] = nacelle
    vehicle['nose_landing_gear'] = nose_landing_gear
    vehicle['main_landing_gear'] = main_landing_gear
    vehicle['systems'] = systems
    vehicle['performance'] = performance
    vehicle['operations'] = operations
    vehicle['noise'] = noise
    vehicle['airport_departure'] = airport_departure
    vehicle['airport_destination'] = airport_destination
    vehicle['results'] = results

    return(vehicle)

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# print(initialize_aircraft_parameters())