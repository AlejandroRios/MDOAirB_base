"""
MDOAirB

Description:
    - This module performs an iterative cycle to re-size the vertical and 
    horizontal stabilizer to decrease the whole airplane weight.

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

from framework.Sizing.Geometry.sizing_vertical_tail import sizing_vertical_tail
from framework.Sizing.Geometry.sizing_horizontal_tail import sizing_horizontal_tail
from framework.Weights.center_of_gravity_position import center_of_gravity
# from framework.CPACS_update.cpacsfunctions import *
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global deg_to_rad
deg_to_rad = np.pi/180


def sizing_tail(vehicle, mach, altitude):
    """
    Description:
        - This function performs an iterative cycle to re-size the vertical and 
        horizontal stabilizer to decrease the whole airplane weight.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - mach - mach number
        - altitude - [m]
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """

    # MODULE_DIR = 'c:/Users/aarc8/Documents/github\MDOAirB_base/framework/CPACS_update'
    # cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'Aircraft_In.xml')
    # cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'Aircraft_Out.xml')
    # tixi = open_tixi(cpacs_out_path)
    # tigl = open_tigl(tixi)

    # np.save('vehicle_test.npy', vehicle)
    wing = vehicle['wing']
    aircraft = vehicle['aircraft']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    relaxation = 0.7
    kink_distance = (wing['span']/2)*wing['semi_span_kink']

    horizontal_tail['aerodynamic_center_ref'] = 0.25
    vertical_tail['aerodynamic_center_ref'] = 0.25

    # Calc of cg here
    vehicle = center_of_gravity(vehicle)

    delta_horizontal_tail = 10000
    delta_vertical_tail = 10000
    margin = aircraft['static_margin']*wing['mean_aerodynamic_chord']

    while (delta_horizontal_tail > 0.025) or (delta_vertical_tail > 0.025):
        airfoil_aerodynamic_center_reference = wing['aerodynamic_center_ref']

        aircraft['neutral_point_xposition'] = wing['leading_edge_xposition'] + wing['mean_aerodynamic_chord_yposition'] * \
            np.tan(wing['sweep_leading_edge']*deg_to_rad) + \
            airfoil_aerodynamic_center_reference*wing['mean_aerodynamic_chord']

        distance_xnp_xcg = aircraft['neutral_point_xposition'] - \
            aircraft['after_center_of_gravity_xposition']

        delta_distance = distance_xnp_xcg - margin
        wing_leading_edge_xposition_new = wing['leading_edge_xposition'] - \
            delta_distance
        wing['leading_edge_xposition'] = wing_leading_edge_xposition_new

        # Iteration cycle for vertical tail
        vertical_tail['aerodynamic_center_xposition'] = 0.95*fuselage['length'] - vertical_tail['center_chord'] + vertical_tail['mean_aerodynamic_chord_yposition'] * \
            np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad) + \
            vertical_tail['aerodynamic_center_ref'] * \
            vertical_tail['mean_aerodynamic_chord']

        distance_vtxac_xcg = vertical_tail['aerodynamic_center_xposition'] - \
            aircraft['after_center_of_gravity_xposition']

        vertical_tail_area_new = (
            wing['area']*vertical_tail['volume']*wing['span'])/distance_vtxac_xcg

        delta_vertical_tail = np.abs(
            vertical_tail['area'] - vertical_tail_area_new)

        vertical_tail['area'] = relaxation*vertical_tail_area_new + \
            (1-relaxation)*vertical_tail['area']

        vehicle = sizing_vertical_tail(
            vehicle,
            mach+0.05,
            altitude)


        # Iteration cycle for horizontal tail
        if horizontal_tail['position'] == 1:
            horizontal_tail['aerodynamic_center_xposition'] = 0.95*fuselage['length'] - horizontal_tail['center_chord'] + horizontal_tail['mean_aerodynamic_chord_yposition'] * \
                np.tan(horizontal_tail['sweep_leading_edge']*deg_to_rad) + \
                horizontal_tail['aerodynamic_center_ref'] * \
                horizontal_tail['mean_aerodynamic_chord']
        else:
            horizontal_tail['aerodynamic_center_xposition'] = 0.95*fuselage['length'] - vertical_tail['center_chord'] + vertical_tail['span'] * \
                np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad) + \
                horizontal_tail['aerodynamic_center_ref'] * \
                horizontal_tail['mean_aerodynamic_chord'] + horizontal_tail['mean_aerodynamic_chord_yposition'] * \
                np.tan(horizontal_tail['sweep_leading_edge']*deg_to_rad)

        distance_htxac_xcg = horizontal_tail['aerodynamic_center_xposition'] - \
            aircraft['after_center_of_gravity_xposition']

        horizontal_tail_area_new = (
            horizontal_tail['volume']*wing['area']*wing['mean_aerodynamic_chord'])/distance_htxac_xcg

        delta_horizontal_tail = np.abs(
            horizontal_tail['area'] - horizontal_tail_area_new)

        horizontal_tail['area'] = relaxation*horizontal_tail_area_new + \
            (1-relaxation)*horizontal_tail['area']

        vehicle = sizing_horizontal_tail(vehicle, mach, altitude)

        vehicle = center_of_gravity(vehicle)

        horizontal_tail['leading_edge_xposition'] = horizontal_tail['aerodynamic_center'] - horizontal_tail['center_chord']*0.25

        # tixi_out = open_tixi(cpacs_out_path)

        # horizontal_thail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[2]/'
        

        # # Update leading edge position
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'transformation/translation/x', horizontal_tail['leading_edge_xposition'], '%g')
        # # Update center chord 
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', horizontal_tail['center_chord'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', horizontal_tail['center_chord'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', horizontal_tail['center_chord'], '%g')

        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', horizontal_tail['tip_chord'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', horizontal_tail['tip_chord'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', horizontal_tail['tip_chord'], '%g')
        # # Update root chord 
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'positionings/positioning[2]/length',horizontal_tail['span']/2, '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'positionings/positioning[2]/sweepAngle',horizontal_tail['sweep_leading_edge'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'positionings/positioning[1]/dihedralAngle',horizontal_tail['dihedral'], '%g')
        # tixi_out.updateDoubleElement(horizontal_thail_xpath+'positionings/positioning[2]/dihedralAngle',horizontal_tail['dihedral'], '%g')

        # tixi_out = close_tixi(tixi_out, cpacs_out_path)

        # tixi_out = open_tixi(cpacs_out_path)

        # vertical_tail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[3]/'
        # vertical_tail['leading_edge_xposition'] = vertical_tail['aerodynamic_center_xposition'] - vertical_tail['center_chord']*0.25
        

        # # Update leading edge position
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'transformation/translation/x', vertical_tail['leading_edge_xposition'], '%g')
        # # Update center chord 
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', vertical_tail['center_chord'], '%g')
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', vertical_tail['center_chord'], '%g')
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', vertical_tail['center_chord'], '%g')

        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', vertical_tail['tip_chord'], '%g')
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', vertical_tail['tip_chord'], '%g')
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', vertical_tail['tip_chord'], '%g')
        # # Update root chord 
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/length',vertical_tail['span'], '%g')
        # tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/sweepAngle',vertical_tail['sweep_leading_edge'], '%g')

        # tixi_out = close_tixi(tixi_out, cpacs_out_path)

        

    return vehicle

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
# vehicle = np.load('vehicle_test.npy',allow_pickle='TRUE').item()
# # print(vehicle['wing']) # displays "world"
# aircraft = vehicle['aircraft']
# horizontal_tail = vehicle['horizontal_tail']
# vertical_tail = vehicle['vertical_tail']
# engine = vehicle['engine']
# fuselage = vehicle['fuselage']
# # print(aircraft)
# # print('---------------------------------------------------------')
# mach = 0.8
# altitude = 41000
# vehicle = sizing_tail(vehicle, mach, altitude)
# aircraft = vehicle['aircraft']
# horizontal_tail = vehicle['horizontal_tail']
# vertical_tail = vehicle['vertical_tail']

# print(vertical_tail)
