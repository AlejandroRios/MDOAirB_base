"""
MDOAirB

Description:
    - This module computes the center of gravity x position

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
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global deg_to_rad
deg_to_rad = np.pi/180
GRAVITY = 9.80665


def center_of_gravity(vehicle):
    """
    Description:
        - This mfunction computes the center of gravity x position
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    # == wing ==
    # Wing mean aerodynamic chord 37 - 42 percent

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']

    nose_landing_gear = vehicle['nose_landing_gear']
    main_landing_gear = vehicle['main_landing_gear']

    performance = vehicle['performance']
    operations = vehicle['operations']
    systems = vehicle['systems']

    if aircraft['slat_presence'] == 0:
        delta_xw = 0.5*(0.1 + wing['rear_spar_ref'])
    else:
        delta_xw = 0.5*(0.15 + wing['rear_spar_ref'])

    wing['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + \
        delta_xw*wing['mean_aerodynamic_chord']
    wing_moment = wing['weight']*wing['center_of_gravity_xposition']

    # == horizontal tail ==
    if horizontal_tail['position'] == 1:
        horizontal_tail['center_of_gravity_xposition'] = 0.95*fuselage['length'] - horizontal_tail['center_chord'] + \
            horizontal_tail['mean_aerodynamic_chord_yposition'] * \
            np.tan(horizontal_tail['sweep_leading_edge']*deg_to_rad) + \
            0.3*horizontal_tail['mean_aerodynamic_chord']
    else:
        horizontal_tail['center_of_gravity_xposition'] = 0.95*fuselage['length'] - vertical_tail['center_chord'] + \
            vertical_tail['span'] * \
            np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad) + 0.3*horizontal_tail['mean_aerodynamic_chord'] + \
            horizontal_tail['mean_aerodynamic_chord_yposition'] * \
            np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad)

    horizontal_tail_moment = horizontal_tail['weight'] * \
        horizontal_tail['center_of_gravity_xposition']

    # == vertical tail ==
    vertical_tail['center_of_gravity_xposition'] = 0.98*fuselage['length'] - (vertical_tail['center_chord'] - (0.3*vertical_tail['mean_aerodynamic_chord'] + (
        vertical_tail['mean_aerodynamic_chord_yposition']*np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad))))
    vertical_tail_moment = vertical_tail['weight'] * \
        vertical_tail['center_of_gravity_xposition']

    # == fuselage ==
    if engine['position'] == 1:
        fuselage['center_of_gravity_xposition'] = 0.43*fuselage['length']
    elif engine['position'] == 2:
        fuselage['center_of_gravity_xposition'] = 0.47*fuselage['length']
    elif engine['position'] == 3:
        fuselage['center_of_gravity_xposition'] = 0.48*fuselage['length']
    elif engine['position'] == 4:
        fuselage['center_of_gravity_xposition'] = 0.43*fuselage['length']

    fuselage_moment = fuselage['weight'] * \
        fuselage['center_of_gravity_xposition']

    # == engines ==
    aircraft['number_of_engines'] = max(2, engine['position'])
    if engine['position'] == 1:
        engine['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge'] *
                                    deg_to_rad) - 0.1*engine['length']
        engine_2_center_of_gravity_xposition = 0
        engine_moment = engine['weight']*engine['center_of_gravity_xposition']
        engine_2_moment = 0

        # == propulsion system ==
        propulsion_system_center_of_gravity_xposition = wing['leading_edge_xposition'] + \
            wing['semi_span_kink']*(wing['span']/2) * \
            np.tan(wing['sweep_leading_edge']*deg_to_rad)
        propulsion_system_moment = systems['propulsion_weight'] * \
            propulsion_system_center_of_gravity_xposition
        propulsion_system_2_moment = 0

    elif engine['position'] == 2:
        engine['center_of_gravity_xposition'] = 0.98*fuselage['length'] - \
            vertical_tail['center_chord'] - \
            engine['length'] + 0.3*engine['length']
        engine_2_center_of_gravity_xposition = 0
        engine_moment = aircraft['number_of_engines'] * \
            engine['weight']*engine['center_of_gravity_xposition']
        engine_2_moment = 0

        # == propulsion system ==
        propulsion_system_center_of_gravity_xposition = engine['center_of_gravity_xposition']
        propulsion_system_moment = systems['propulsion_weight'] * \
            propulsion_system_center_of_gravity_xposition
        propulsion_system_2_moment = 0

    elif engine['position'] == 3:
        engine['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge']*deg_to_rad) - \
            0.25*engine['length'] + 0.3*engine['length']
        engine_2_center_of_gravity_xposition = fuselage['length'] - \
            vertical_tail['center_chord'] - \
            engine['length'] + 0.3*engine['length']
        engine_moment = (2/3)*aircraft['number_of_engines'] * engine['weight'] * \
            engine['center_of_gravity_xposition']
        engine_2_moment = (1/3)*aircraft['number_of_engines'] * engine['weight'] * \
            engine_2_center_of_gravity_xposition

        # == propulsion system ==
        propulsion_system_moment = (
            2/3)*systems['propulsion_weight']*engine['center_of_gravity_xposition']
        propulsion_system_2_moment = (
            1/3)*systems['propulsion_weight']*engine_2_center_of_gravity_xposition

    elif engine['position'] == 4:
        engine['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge']*deg_to_rad) - \
            0.25*engine['length'] + 0.3*engine['length']
        engine_2_center_of_gravity_xposition = wing['leading_edge_xposition'] + 0.7 * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge']*deg_to_rad) - \
            0.25*engine['length'] + 0.3*engine['length']
        engine_moment = 0.5*aircraft['number_of_engines'] * engine['weight'] * \
            engine['center_of_gravity_xposition']
        engine_2_moment = 0.5*aircraft['number_of_engines'] * engine['weight'] * \
            engine_2_center_of_gravity_xposition

        # == propulsion system ==
        propulsion_system_moment = 0.5*systems['propulsion_weight'] * \
            engine['center_of_gravity_xposition']
        propulsion_system_moment2 = 0.5*systems['propulsion_weight'] * \
            engine_2_center_of_gravity_xposition

    # == nacelles==

    if engine['position'] == 1:
        nacelle['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge'] *
                                    deg_to_rad) + 0.4*engine['length']
        nacelle_moment = nacelle['weight'] * \
            nacelle['center_of_gravity_xposition']
    elif engine['position'] == 2:
        nacelle['center_of_gravity_xposition'] = 0.97*fuselage['length'] - \
            vertical_tail['center_chord'] - \
            engine['length'] + 0.35*engine['length']
        nacelle_moment = nacelle['weight'] * \
            nacelle['center_of_gravity_xposition']
    elif engine['position'] == 3:
        nacelle['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge'] *
                                    deg_to_rad) + 0.4*engine['length']
        nacelle_moment = nacelle['weight'] * \
            nacelle['center_of_gravity_xposition']
    elif engine['position'] == 4:  # CHECK THIS ONE!!!!
        nacelle['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + wing['semi_span_kink'] * \
            (wing['span']/2)*np.tan(wing['sweep_leading_edge'] *
                                    deg_to_rad) + 0.4*engine['length']
        nacelle_moment = nacelle['weight'] * \
            nacelle['center_of_gravity_xposition']

    # == landing gear ==

    # nose landing gear
    nose_landing_gear['center_of_gravity_xposition'] = 0.5 * \
        fuselage['cockpit_length']
    nose_landing_gear_moment = nose_landing_gear['weight'] * \
        nose_landing_gear['center_of_gravity_xposition']

    # main landing gear
    if wing['position'] == 1:
        main_landing_gear['center_of_gravity_xposition'] = wing['leading_edge_xposition'] + \
            wing['trunnion_xposition']
    else:
        main_landing_gear['center_of_gravity_xposition'] = wing['center_of_gravity_xposition'] + \
            0.20*wing['mean_aerodynamic_chord']

    main_landig_gear_moment = main_landing_gear['weight'] * \
        main_landing_gear['center_of_gravity_xposition']

    # == fuel system ==
    # fuel system
    fuel_system_center_of_gravity_xposition = wing['leading_edge_xposition'] - \
        wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + \
        wing['center_chord'] * 0.5
    fuel_system_moment = systems['fuel_weight'] * \
        fuel_system_center_of_gravity_xposition

    # flight control
    flight_control_system_wing_center_of_gravity_xposition = wing['leading_edge_xposition'] - wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + (wing['span'] *
                                                         np.tan(wing['sweep_leading_edge']*deg_to_rad) + wing['tip_chord'])/2
    flight_control_system_wing_moment = 0.5*systems['flight_control_weight'] * \
        flight_control_system_wing_center_of_gravity_xposition

    flight_control_system_tail_center_of_gravity_xposition = vertical_tail[
        'center_of_gravity_xposition']
    flight_control_tail_moment = 0.5*systems['flight_control_weight'] * \
        flight_control_system_tail_center_of_gravity_xposition

    # hydraulic system
    hydraulic_system_center_of_gravity_xposition = wing['leading_edge_xposition'] + \
        0.6*wing['center_chord']
    hydraulic_system_moment = systems['hydraulic_weight'] * \
        hydraulic_system_center_of_gravity_xposition

    # electrical system
    electrical_system_center_of_gravity_xposition = wing['leading_edge_xposition'] - \
        wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + wing['center_chord']
    electrical_system_moment = systems['electrical_weight'] * \
        electrical_system_center_of_gravity_xposition

    # avionics
    avionics_system_center_of_gravity_xposition = 0.4 * \
        fuselage['cockpit_length']
    avionics_system_moment = systems['avionics_weight'] * \
        avionics_system_center_of_gravity_xposition

    # air system
    air_system_center_of_gravity_xposition = wing['leading_edge_xposition'] - wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + \
        (wing['mean_aerodynamic_chord_yposition'] *
         np.tan(wing['sweep_leading_edge']*deg_to_rad)/2)
    air_system_moment = systems['air_weight'] * \
        air_system_center_of_gravity_xposition

    # oxygen system
    oxygen_system_center_of_gravity_xposition = fuselage['cockpit_length'] + \
        (wing['leading_edge_xposition'] - wing['mean_aerodynamic_chord_yposition'] *
         np.tan(wing['sweep_leading_edge']*deg_to_rad) - fuselage['cockpit_length'])/2
    oxygen_system_moment = systems['oxygen_weight'] * \
        oxygen_system_center_of_gravity_xposition

    # auxiliar power unit
    apu_center_of_gravity_xposition = fuselage['length'] - 2.0
    apu_moment = systems['APU_weight']*apu_center_of_gravity_xposition

    # furnishing
    furnishing_center_of_gravity_xposition = fuselage['cockpit_length'] + \
        (fuselage['cabine_length']/2)
    furnishing_moment = systems['furnishing_weight'] * \
        furnishing_center_of_gravity_xposition

    # paint
    paint_center_of_gravity_xposition = 0.51*fuselage['length']
    paint_moment = systems['paint_weight']*paint_center_of_gravity_xposition

    # wing fuel
    wing_tank_center_of_gravity_xposition = wing['leading_edge_xposition'] + \
        wing['tank_center_of_gravity_xposition']

    wing_fuel_moment = systems['fuel_weight'] * \
        wing_tank_center_of_gravity_xposition

    aircraft['operational_empty_weight'] = (wing['weight'] + horizontal_tail['weight'] + vertical_tail['weight'] + fuselage['weight'] + aircraft['power_plant_weight'] + nacelle['weight'] + main_landing_gear['weight'] + nose_landing_gear['weight'] + systems['hydraulic_weight'] +
                                            systems['flight_control_weight'] + systems['electrical_weight'] + systems['oxygen_weight'] + systems['APU_weight'] + systems['furnishing_weight'] + systems['paint_weight'] + systems['avionics_weight'] + systems['air_weight'] + systems['safety'] + systems['handling_gear'])

    aircraft_empty_weight_center_of_gravity_xposition = (wing_moment+horizontal_tail_moment+vertical_tail_moment+fuselage_moment+engine_moment+engine_2_moment+propulsion_system_moment+propulsion_system_2_moment+nacelle_moment+nose_landing_gear_moment+main_landig_gear_moment + hydraulic_system_moment +
                                                         fuel_system_moment+flight_control_system_wing_moment+flight_control_tail_moment+electrical_system_moment+avionics_system_moment+air_system_moment+oxygen_system_moment+apu_moment+furnishing_moment+paint_moment)/aircraft['operational_empty_weight']

    aircraft_empty_weight_center_of_gravity_mean_aerodynamic_chord_xposition = aircraft_empty_weight_center_of_gravity_xposition / \
        wing['mean_aerodynamic_chord']

    # == crew ==
    # cockpit
    crew_cockpit_weight = 2*75
    crew_cockpit_center_of_gravity_xposition = fuselage['cockpit_length']/2
    crew_cockpit_moment = crew_cockpit_weight * \
        crew_cockpit_center_of_gravity_xposition

    # cabine

    crew_cabine_weight = 3*75
    crew_cabine_center_of_gravity_xposition = fuselage['cockpit_length'] + \
        fuselage['cabine_length']
    crew_cabine_moment = crew_cabine_weight*crew_cabine_center_of_gravity_xposition

    residual_fuel_weight = 0.005*wing['fuel_capacity']
    residual_fuel_center_of_gravity_xposition = wing['leading_edge_xposition'] + \
        wing['tank_center_of_gravity_xposition']
    residual_fuel_moment = residual_fuel_weight * \
        residual_fuel_center_of_gravity_xposition

    # print(aircraft['operational_empty_weight'])

    aircraft_operating_empty_weight_moment = aircraft['operational_empty_weight']*aircraft_empty_weight_center_of_gravity_xposition + \
        crew_cockpit_moment + crew_cabine_moment + residual_fuel_moment

    aircraft_operating_empty_weight = aircraft['operational_empty_weight'] + \
        crew_cockpit_weight + crew_cabine_weight + residual_fuel_weight

    wing_fuel_weight = wing['fuel_capacity']

    fuel_tanks_moment = wing_fuel_weight * \
        wing_tank_center_of_gravity_xposition

    pax_weight = aircraft['passenger_capacity']*100
    pax_center_of_gravity_xposition = fuselage['cockpit_length'] + \
        fuselage['cabine_length']/2
    pax_moment = pax_weight*pax_center_of_gravity_xposition

    # CG shift positions
    configuration_1 = aircraft_operating_empty_weight_moment / \
        aircraft_operating_empty_weight

    configuration_2 = (aircraft_operating_empty_weight_moment +
                       fuel_tanks_moment)/(aircraft_operating_empty_weight + wing_fuel_weight)

    configuration_3 = (aircraft_operating_empty_weight_moment + fuel_tanks_moment +
                       pax_moment)/(aircraft_operating_empty_weight + wing_fuel_weight + pax_weight)

    configuration_4 = (aircraft_operating_empty_weight_moment +
                       pax_moment)/(aircraft_operating_empty_weight + pax_weight)

    aircraft['forward_center_of_gravity_xposition'] = min(
        configuration_1, configuration_2, configuration_3, configuration_4)

    aircraft['after_center_of_gravity_xposition'] = max(
        configuration_1, configuration_2, configuration_3, configuration_4)

    aircraft_center_of_gravity_shift_range = (
        (aircraft['after_center_of_gravity_xposition'] - aircraft['forward_center_of_gravity_xposition'])/wing['mean_aerodynamic_chord'])*100
    return vehicle
# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================
