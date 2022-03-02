"""
MDOAirB

Description:
    - This module calculate the regulated takeoff and landing weight according to
    sizing restrictions
Reference:
    - Torenbeek

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
from framework.Performance.Analysis.balanced_field_length import balanced_field_length
from framework.Performance.Analysis.landing_field_length import landing_field_length
from framework.Performance.Analysis.second_segment_climb import second_segment_climb
from framework.Performance.Analysis.missed_approach_climb import missed_approach_climb_AEO, missed_approach_climb_OEI
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Analysis.cruise_performance import cruise_performance
from framework.Performance.Analysis.residual_rate_of_climb import residual_rate_of_climb
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global GRAVITY
GRAVITY = 9.80665
lb_to_kg = 0.453592


def takeoff_field_length_check(vehicle, airport_departure, takeoff_runway, weight_takeoff,gamma_2):
    """
    Description:
        - This function performs the takeoff field length check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - takeoff_runway - selected takeoff runway
        - weight_takeoff - takeoff weight [N]
        - gamma_2 - second segment climb gradient
    Outputs:
        - weight_takeoff - takeoff weight [N]
    """

    aircraft = vehicle['aircraft']

    takeoff_field_length_required = takeoff_runway['tora']
    # weight_takeoff = aircraft['maximum_takeoff_weight']

    flag = 0
    while flag == 0:

        takeoff_field_length_computed = balanced_field_length(
            vehicle, airport_departure, weight_takeoff,gamma_2)

        if takeoff_field_length_computed > takeoff_field_length_required:
            weight_takeoff = weight_takeoff - (10*GRAVITY)
        else:
            flag = 1
    return weight_takeoff


def second_segment_climb_check(vehicle, airport_departure, weight_takeoff):
    """
    Description:
        - This function performs the second segment climb check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - weight_takeoff - takeoff weight [N]
    Outputs:
        - weight_takeoff - takeoff weight [N]
        - thrust_to_weight_takeoff
    """

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    # weight_takeoff = aircraft['maximum_takeoff_weight']
    thrust_takeoff = engine['maximum_thrust']*0.98
    engines_number = aircraft['number_of_engines']

    flag = 0
    while flag == 0:
        thrust_to_weight_takeoff_required = second_segment_climb(
            vehicle, airport_departure, weight_takeoff)
        thrust_to_weight_takeoff =(engines_number*thrust_takeoff)/weight_takeoff  # Second segment climb shouldnt use only one engine?

        if thrust_to_weight_takeoff < thrust_to_weight_takeoff_required:
            weight_takeoff = weight_takeoff-(10*GRAVITY)
        else:
            flag = 2
    return weight_takeoff, thrust_to_weight_takeoff


def landing_field_length_check(vehicle, airport_destination, landing_runway, maximum_takeoff_weight, weight_landing):
    """
    Description:
        - This function performs the landing field length check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - landing_runway
        - maximum_takeoff_weight - [N]
        - weight_landing - [N]
    Outputs:
        - weight_landing - [N]
    """

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    landing_field_length_required = landing_runway['lda']
    wing_surface = wing['area']

    flag = 0
    while flag == 0:
        # aircraft['maximum_landing_weight'] = weight_landing
        landing_field_length_computed = landing_field_length(
            vehicle, airport_destination, weight_landing)

        maximum_takeoff_mass = maximum_takeoff_weight/GRAVITY
        maximum_landing_mass = weight_landing/GRAVITY

        maximum_takeoff_mass_to_wing_surface_requirement = (
            (maximum_landing_mass/wing_surface)/(maximum_landing_mass/maximum_takeoff_mass))
        maximum_takeoff_mass_to_wing_surface = maximum_landing_mass/wing_surface

        if (landing_field_length_computed > landing_field_length_required or maximum_takeoff_mass_to_wing_surface > maximum_takeoff_mass_to_wing_surface_requirement):
            weight_landing = weight_landing-(10*GRAVITY)
        else:
            flag = 2

    return weight_landing


def landing_climb_check(vehicle, airport_destination, maximum_takeoff_weight, weight_landing):
    """
    Description:
        - This function performs the landing climb check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - maximum_takeoff_weight - [N]
        - weight_landing [N]
    Outputs:
        - weight_landing - [N]
    """

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']

    thrust_landing = engine['maximum_thrust'] * 0.98
    engines_number = aircraft['number_of_engines']

    flag = 0
    while flag == 0:
        thrust_to_weight_landing_required = missed_approach_climb_AEO(
            vehicle, airport_destination, maximum_takeoff_weight, weight_landing)
        thrust_to_weight_landing = (thrust_landing*engines_number)/weight_landing

        if thrust_to_weight_landing < thrust_to_weight_landing_required:
            weight_landing = weight_landing-(10*GRAVITY)
        else:
            flag = 2
    return weight_landing


def missed_approach_climb_check(vehicle, airport_destination, maximum_takeoff_weight, weight_landing):
    """
    Description:
        - This function performs the missed approach climb check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - maximum_takeoff_weight - [N]
        - weight_landing - [N]
    Outputs:
        - weight_landing - [N]
    """

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']

    thrust_landing = engine['maximum_thrust'] * 0.98
    engines_number = aircraft['number_of_engines']

    flag = 0
    while flag == 0:
        thrust_to_weight_landing_required = missed_approach_climb_OEI(
            vehicle, airport_destination, maximum_takeoff_weight, weight_landing)
        thrust_to_weight_landing = (thrust_landing*engines_number)/weight_landing

        if thrust_to_weight_landing < thrust_to_weight_landing_required:
            weight_landing = weight_landing-(10*GRAVITY)

        else:
            flag = 2
    return weight_landing
    


def residual_rate_of_climb_check(vehicle, airport_departure, weight_takeoff,engine_cruise_thrust):
    """
    Description:
        - This function performs the residual rate of climb check
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - weight_takeoff - takeoff weight [N]
        - engine_cruise_thrust
    Outputs:
        - weight_takeoff - takeoff weight [N]
        - thrust_to_weight_takeoff
    """

    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    # weight_takeoff = aircraft['maximum_takeoff_weight']
    thrust_takeoff = engine['maximum_thrust']*0.98
    engines_number = aircraft['number_of_engines']

    flag = 0
    while flag == 0:
        thrust_to_weight_takeoff_required = residual_rate_of_climb(
            vehicle, airport_departure, weight_takeoff,engine_cruise_thrust)
        thrust_to_weight_takeoff =(engines_number*thrust_takeoff)/weight_takeoff  # Second segment climb shouldnt use only one engine?

        if thrust_to_weight_takeoff < thrust_to_weight_takeoff_required:
            weight_takeoff = weight_takeoff-(10*GRAVITY)
        else:
            flag = 2
    return weight_takeoff, thrust_to_weight_takeoff




def maximum_cruise_speed_check():

    return


def drag_divergence_check():

    return


def regulated_takeoff_weight(vehicle, airport_departure, takeoff_runway):
    """
    Description:
        - This function performs the calculation of the regulated takeoff weight
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_departure
        - takeoff_runway
    Outputs:
        - regulated takeoff weight - [kg]
    """

    aircraft = vehicle['aircraft']

    weight_takeoff = aircraft['maximum_takeoff_weight']*GRAVITY


    second_segment_climb_weight,gamma_2 = second_segment_climb_check(
        vehicle, airport_departure, weight_takeoff)

    takeoff_field_length_weight = takeoff_field_length_check(
        vehicle, airport_departure, takeoff_runway, weight_takeoff,gamma_2)



    maximum_takeoff_weight = min(
        takeoff_field_length_weight, second_segment_climb_weight)
    return maximum_takeoff_weight/GRAVITY  # [Kg]


def regulated_landing_weight(vehicle, airport_destination, landing_runway):
    """
    Description:
        - This function performs the calculation of the regulated landing weight
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - airport_destination
        - landing_runway
    Outputs:
        - regulated landing weight - [kg]
    """

    aircraft = vehicle['aircraft']
    weight_landing = aircraft['maximum_landing_weight']*GRAVITY

    maximum_takeoff_weight = aircraft['maximum_takeoff_weight']*GRAVITY

    landing_field_length_weight = landing_field_length_check(
        vehicle, airport_destination, landing_runway, maximum_takeoff_weight, weight_landing)

    landing_climb = landing_climb_check(
        vehicle, airport_destination, maximum_takeoff_weight, weight_landing)

    missed_approach = missed_approach_climb_check(
        vehicle, airport_destination, maximum_takeoff_weight, weight_landing)

    maximum_landing_weight = min(
        landing_field_length_weight, landing_climb, missed_approach)
    return maximum_landing_weight/GRAVITY  # [Kg]
# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================


# aircraft = vehicle['aircraft']
# weight_takeoff = aircraft['maximum_takeoff_weight']
# takeoff_field_length_weight = takeoff_field_length_check(vehicle, weight_takeoff)
# print('weight BFL requirement:', takeoff_field_length_weight/GRAVITY)

# # print(takeoff_field_length_weight/GRAVITY)
# second_segment_climb_weight =  second_segment_climb_check(vehicle, weight_takeoff)
# print('weight second segment requirement:', second_segment_climb_weight/GRAVITY)

# maximum_takeoff_weight = min(takeoff_field_length_weight, second_segment_climb_weight)
# print('========================================================================================')
# print('MTOW [kg]:', maximum_takeoff_weight/GRAVITY)
# print('========================================================================================')

# aircraft = vehicle['aircraft']
# weight_landing = aircraft['maximum_landing_weight']
# landing_field_length_weight = landing_field_length_check(vehicle, maximum_takeoff_weight, weight_landing)
# print('weight landing field requirement:', landing_field_length_weight/GRAVITY)

# landing_climb = landing_climb_check(vehicle, maximum_takeoff_weight, weight_landing)
# print('weight landing climb:', landing_climb/GRAVITY)


# missed_approach = missed_approach_climb_check(vehicle, maximum_takeoff_weight, weight_landing)
# print('weight missed approach:', missed_approach/GRAVITY)

# maximum_landing_weight = min(landing_field_length_weight, landing_climb, missed_approach)
# print('========================================================================================')
# print('MLW [kg]:', maximum_landing_weight/GRAVITY)
# print('========================================================================================')
