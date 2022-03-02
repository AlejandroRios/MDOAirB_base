"""
MDOAirB

Description:
    - This module calculates the cabine dimensions.

Reference:
    - PreSTO-Cabin - https://www.fzt.haw-hamburg.de/pers/Scholz/PreSTo/PreSTo-Cabin_Documentation_10-11-15.pdf

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
import math
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
in_to_m = 0.0254

def fuselage_cross_section(vehicle):
    """
    Description:
        - This function calculates the cabine dimensions
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    fuselage = vehicle['fuselage']
    cabine = vehicle['cabine']
    # Seat dimensions economy class
    armrest_top = cabine['armrest_top']  # [inch]
    armrest_bottom = cabine['armrest_bottom']    # [inch]
    armrest_width = cabine['armrest_width']*in_to_m  # armrest width
    armrest_top_height = armrest_top*in_to_m
    armrest_bottom_height = armrest_bottom*in_to_m

    seat_cushion_thickness_YC = cabine['seat_cushion_thickness_YC'] # [m] YC - economy class
    seat_cushion_width_YC = cabine['seat_width']  
    double_container = 'no'
    backrest_height = cabine['backrest_height'] # [m]
    floor_thickness = cabine['floor_thickness']  # [m]
    aux = ((armrest_top_height-armrest_bottom_height) -
           seat_cushion_thickness_YC)/2
    seat_cushion_height = aux + armrest_bottom_height

    # Default values (95% american male)
    pax_distance_head_wall = cabine['pax_distance_head_wall']  # [m]
    pax_distance_shoulder_wall = cabine['pax_distance_shoulder_wall']  # [m]
    pax_shoulder_breadth = cabine['pax_shoulder_breadth']  # [m]
    pax_eye_height = cabine['pax_eye_height']  # [m]
    pax_midshoulder_height = cabine['pax_midshoulder_height']  # [m]

    delta_z_symmetry_inferior = cabine['delta_z_symmetry_inferior']
    delta_z_symmetry_superior = cabine['delta_z_symmetry_superior']
    seat_delta_width_floor = cabine['seat_delta_width_floor']
    points_number = 20

    iterations = 12
    if double_container == 'no':
        if fuselage['container_type'] == 'None':
            lowerdeck_width_top = 0
            lowerdeck_width_bottom = 0
            lowerdeck_height = 0
        elif fuselage['container_type'] == 'LD1':
            lowerdeck_width_bottom = 1.56
            lowerdeck_width_top = 2.44
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD11':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD2':
            lowerdeck_width_bottom = 1.19
            lowerdeck_width_top = 1.66
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD26':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.16
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD29':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.82
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD3':
            lowerdeck_width_bottom = 1.56
            lowerdeck_width_top = 2.11
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD3-45':
            lowerdeck_width_bottom = 1.56
            lowerdeck_width_top = 2.11
            lowerdeck_height = 1.19
        elif fuselage['container_type'] == 'LD3-45R':
            lowerdeck_width_bottom = 1.56
            lowerdeck_width_top = 1.66
            lowerdeck_height = 1.19
        elif fuselage['container_type'] == 'LD3-45W':
            lowerdeck_width_bottom = 1.43
            lowerdeck_width_top = 2.53
            lowerdeck_height = 1.14
        elif fuselage['container_type'] == 'LD39':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.82
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD4':
            lowerdeck_width_bottom = 2.44
            lowerdeck_width_top = 2.54
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD6':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.16
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD8':
            lowerdeck_width_bottom = 2.44
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD9':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68
    else:
        if fuselage['container_type'] == 'None':
            lowerdeck_width_top = 0
            lowerdeck_width_bottom = 0
            lowerdeck_height = 0
        elif fuselage['container_type'] == 'LD1':
            lowerdeck_width_bottom = 3.22
            lowerdeck_width_top = 4.77
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD11':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD2':
            lowerdeck_width_bottom = 2.49
            lowerdeck_width_top = 3.22
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD26':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.16
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD29':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.82
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD3':
            lowerdeck_width_bottom = 3.22
            lowerdeck_width_top = 4.11
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD3-45':
            lowerdeck_width_bottom = 3.22
            lowerdeck_width_top = 4.11
            lowerdeck_height = 1.19
        elif fuselage['container_type'] == 'LD3-45R':
            lowerdeck_width_bottom = 1.56
            lowerdeck_width_top = 1.66
            lowerdeck_height = 1.19
        elif fuselage['container_type'] == 'LD3-45W':
            lowerdeck_width_bottom = 1.43
            lowerdeck_width_top = 2.53
            lowerdeck_height = 1.14
        elif fuselage['container_type'] == 'LD39':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.82
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD4':
            lowerdeck_width_bottom = 2.44
            lowerdeck_width_top = 2.54
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD6':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 4.16
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD8':
            lowerdeck_width_bottom = 2.44
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68
        elif fuselage['container_type'] == 'LD9':
            lowerdeck_width_bottom = 3.18
            lowerdeck_width_top = 3.28
            lowerdeck_height = 1.68

    if fuselage['aisles_number'] == 1:
        seats_number = max(fuselage['seat_abreast_number'], 2)  # minor number of rows == 3
        seats_number = min(seats_number, 6)  # major number of rows == 9
    elif fuselage['aisles_number'] == 2:
        seats_number = max(fuselage['seat_abreast_number'], 6)  # minor number of rows == 3
        seats_number = min(seats_number, 9)  # major number of rows == 9

    # fuselage['seat_abreast_number'] = seats_number

    if fuselage['aisles_number'] == 1:
        left_fuselage_seats = math.ceil(fuselage['seat_abreast_number']/2)
        right_fuselage_seats = fuselage['seat_abreast_number'] - left_fuselage_seats
    else:
        if fuselage['seat_abreast_number'] == 6:
            left_fuselage_seats = 2
            right_fuselage_seats = 2
            center_fuselage_seats = 2
        elif fuselage['seat_abreast_number'] == 7:
            left_fuselage_seats = 2
            right_fuselage_seats = 2
            center_fuselage_seats = 3
        elif fuselage['seat_abreast_number'] == 8:
            left_fuselage_seats = 3
            right_fuselage_seats = 3
            center_fuselage_seats = 2
        elif fuselage['seat_abreast_number'] == 9:
            left_fuselage_seats = 3
            right_fuselage_seats = 3
            center_fuselage_seats = 3

    # Calculate the width coordinates for the various points
    w0 = 0.5*fuselage['cabine_height']*fuselage['aisles_number']
    w4 = fuselage['seat_abreast_number']*seat_cushion_width_YC + fuselage['aisles_number'] * \
        fuselage['aisle_width'] + (fuselage['seat_abreast_number'] -
                       fuselage['aisles_number'] + 1)*armrest_width
    y_last_seat = 0.5*(w4 - seat_cushion_width_YC - 2*armrest_width)
    w1 = 2*y_last_seat
    w2 = 2*(pax_distance_head_wall + 0.084 + y_last_seat)
    w3 = 2*(pax_distance_shoulder_wall + pax_shoulder_breadth/2 + y_last_seat)
    w5 = w4
    w6 = w4 - 2*seat_delta_width_floor - 2*armrest_width
    w7 = lowerdeck_width_top
    w8 = lowerdeck_width_top
    w9 = lowerdeck_width_bottom

    while iterations > 0:
        iterations = iterations - 1

        k = 0
        k_minimum = points_number
        result_z_symmetry_minimum = 1000
        k_minimum2 = points_number
        result_z_symmetry_minimum2 = 1000

        while k <= points_number:
            delta_z_symmetry = k * \
                (delta_z_symmetry_superior - delta_z_symmetry_inferior) / \
                points_number + delta_z_symmetry_inferior
            h0 = fuselage['cabine_height'] - delta_z_symmetry
            h1 = pax_eye_height + seat_cushion_height - \
                delta_z_symmetry + 0.126 + pax_distance_head_wall
            h2 = pax_eye_height + seat_cushion_height - delta_z_symmetry
            h3 = pax_midshoulder_height + seat_cushion_height - delta_z_symmetry
            h4 = armrest_top_height - delta_z_symmetry
            h5 = armrest_bottom_height - delta_z_symmetry
            h6 = -delta_z_symmetry
            h7 = -delta_z_symmetry - floor_thickness
            h8 = -delta_z_symmetry - floor_thickness - lowerdeck_height + \
                (lowerdeck_width_top - lowerdeck_width_bottom)/2
            h9 = -delta_z_symmetry - floor_thickness - lowerdeck_height

            # Calculate semi width of the ellipse describing the fuselage
            a0 = np.sqrt(((w0/2)**2 + (h0**2)/(fuselage['height_to_width_ratio'])**2))
            a1 = np.sqrt(((w1/2)**2 + (h1**2)/(fuselage['height_to_width_ratio'])**2))
            a2 = np.sqrt(((w2/2)**2 + (h2**2)/(fuselage['height_to_width_ratio'])**2))
            a3 = np.sqrt(((w3+0.04)/2)**2 +
                         ((h3**2)/(fuselage['height_to_width_ratio']**2)))
            a4 = np.sqrt(((w4+0.04)/2)**2 +
                         ((h4**2)/(fuselage['height_to_width_ratio']**2)))
            a5 = np.sqrt(((w5/2)**2 + (h5**2)/(fuselage['height_to_width_ratio'])**2))
            a6 = np.sqrt(((w6/2)**2 + (h6**2)/(fuselage['height_to_width_ratio'])**2))
            a7 = np.sqrt(((w7/2)**2 + (h7**2)/(fuselage['height_to_width_ratio'])**2))
            a8 = np.sqrt(((w8/2)**2 + (h8**2)/(fuselage['height_to_width_ratio'])**2))
            a9 = np.sqrt(((w9/2)**2 + (h9**2)/(fuselage['height_to_width_ratio'])**2))

            # Get the maximum value of these widths, so each point == inside
            # the fuselage
            array_widths = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
            maximum_width = max(array_widths)

            # If the current width == one of the 2 smallest, then it has to be stored
            if maximum_width < result_z_symmetry_minimum:
                k_minimum2 = k_minimum
                result_z_symmetry_minimum2 = result_z_symmetry_minimum
                k_minimum = k
                result_z_symmetry_minimum = maximum_width
            elif maximum_width < result_z_symmetry_minimum2:
                k_minimum2 = k
                result_z_symmetry_minimum2 = maximum_width

            k = k + 1

        # Update the interval where delta_z_symmetry has to be
        if k_minimum < k_minimum2:
            if k_minimum > 0:
                k = k_minimum - 1
            delta_z_symmetry_inferior_new = k * \
                (delta_z_symmetry_superior - delta_z_symmetry_inferior) / \
                points_number + delta_z_symmetry_inferior
            if k_minimum < points_number:
                k = k_minimum2 + 1
            delta_z_symmetry_superior_new = k * \
                (delta_z_symmetry_superior - delta_z_symmetry_inferior) / \
                points_number + delta_z_symmetry_inferior
        else:
            if k_minimum2 > 0:
                k = k_minimum2 - 1
            delta_z_symmetry_inferior_new = k * \
                (delta_z_symmetry_superior - delta_z_symmetry_inferior) / \
                points_number + delta_z_symmetry_inferior
            if k_minimum < points_number:
                k = k_minimum + 1
            delta_z_symmetry_superior_new = k * \
                (delta_z_symmetry_superior - delta_z_symmetry_inferior) / \
                points_number + delta_z_symmetry_inferior

        delta_z_symmetry_inferior = delta_z_symmetry_inferior_new
        delta_z_symmetry_superior = delta_z_symmetry_superior_new
        fuselage['minimum_width'] = result_z_symmetry_minimum

        # Update the fuselage equivalent diameter and fuselage and floor thickness
        fuselage_equivalent_diameter = 2 * \
            fuselage['minimum_width']*np.sqrt(fuselage['height_to_width_ratio'])
        fuselage_thickness = (0.084 + 0.045*fuselage_equivalent_diameter)/2
        floor_thickness = 0.035 * \
            (fuselage_equivalent_diameter + fuselage_thickness)

    besta = [delta_z_symmetry_inferior, delta_z_symmetry_superior]
    fuselage['Dz_floor'] = min(besta)
    fuselage_thickness_floor = floor_thickness
    fuselage_outer_equivalent_diameter = fuselage_equivalent_diameter + 2*fuselage_thickness
    fuselage_df = fuselage_outer_equivalent_diameter
    fuselage_wf = fuselage_thickness

    fuselage_rails_number_right = 2
    fuselage_rails_number_left = 2
    fuselage_dseat_seat_rail = 0.2

    axis_x = 2*fuselage['minimum_width']
    axis_y = axis_x*fuselage['height_to_width_ratio']

    axis_x_exterior = axis_x + 2*fuselage_thickness
    axis_y_exterior = axis_y + 2*fuselage_thickness

    fuselage['width'] = axis_x_exterior
    fuselage['height'] = axis_y_exterior

    return vehicle
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
