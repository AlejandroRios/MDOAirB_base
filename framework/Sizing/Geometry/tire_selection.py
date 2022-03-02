"""
MDOAirB

Description:
    - This module selects the tire as function of sizing parameters.
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
import pandas as pd
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
in_to_m = 0.0254


def tire_selection(load, V_qualified, pneu_pressure, selection_criteria):
    """
    Description:
        - This module selects the tire as function of sizing parameters.
    Inputs:
        - load - tire load [lbf]
        - V_qualified - qualified speed [mph]
        - pneu_pressure - [psi]
        - selection_criteria - weight or size
    Outputs:
        - tire_diameter - [m]
        - tire_width - [m]
    """

    tire_database = {1: {'ply': 12.0, 'velocity': 210.0, 'load': 11200.0, 'pressure': 120.0, 'max_force': 16800.0, 'weight': 64.1, 'diameter': 32.0, 'width': 11.5, 'radius': 13.5, 'AR': 0.742},
                     2: {'ply': 12.0, 'velocity': 210.0, 'load': 11200.0, 'pressure': 120.0, 'max_force': 16800.0, 'weight': 70.6, 'diameter': 32.0, 'width': 11.5, 'radius': 13.5, 'AR': 0.742},
                     3: {'ply': 12.0, 'velocity': 225.0, 'load': 11200.0, 'pressure': 120.0, 'max_force': 16800.0, 'weight': 69.7, 'diameter': 32.0, 'width': 11.5, 'radius': 13.5, 'AR': 0.742},
                     4: {'ply': 20.0, 'velocity': 210.0, 'load': 22200.0, 'pressure': 165.0, 'max_force': 33300.0, 'weight': 110.2, 'diameter': 37.0, 'width': 13.0, 'radius': 15.4, 'AR': 0.821},
                     5: {'ply': 24.0, 'velocity': 225.0, 'load': 25000.0, 'pressure': 160.0, 'max_force': 37500.0, 'weight': 110.8, 'diameter': 37.0, 'width': 14.0, 'radius': 15.1, 'AR': 0.825},
                     6: {'ply': 24.0, 'velocity': 225.0, 'load': 25000.0, 'pressure': 160.0, 'max_force': 37500.0, 'weight': 106.4, 'diameter': 37.0, 'width': 14.0, 'radius': 15.1, 'AR': 0.825},
                     7: {'ply': 26.0, 'velocity': 235.0, 'load': 36300.0, 'pressure': 180.0, 'max_force': 54450.0, 'weight': 138.8, 'diameter': 40.0, 'width': 15.5, 'radius': 16.1, 'AR': 0.778},
                     8: {'ply': 26.0, 'velocity': 235.0, 'load': 36300.0, 'pressure': 180.0, 'max_force': 54450.0, 'weight': 147.4, 'diameter': 40.0, 'width': 15.5, 'radius': 16.1, 'AR': 0.778},
                     9: {'ply': 30.0, 'velocity': 225.0, 'load': 42500.0, 'pressure': 195.0, 'max_force': 63750.0, 'weight': 196.3, 'diameter': 44.5, 'width': 16.5, 'radius': 18.4, 'AR': 0.807},
                     10: {'ply': 30.0, 'velocity': 225.0, 'load': 42500.0, 'pressure': 195.0, 'max_force': 63750.0, 'weight': 193.8, 'diameter': 44.5, 'width': 16.5, 'radius': 18.4, 'AR': 0.807},
                     11: {'ply': 32.0, 'velocity': 235.0, 'load': 51900.0, 'pressure': 195.0, 'max_force': 77800.0, 'weight': 269.1, 'diameter': 49.0, 'width': 19.0, 'radius': 20.3, 'AR': 0.767},
                     12: {'ply': 32.0, 'velocity': 235.0, 'load': 51900.0, 'pressure': 195.0, 'max_force': 77800.0, 'weight': 270.4, 'diameter': 49.0, 'width': 19.0, 'radius': 20.3, 'AR': 0.767},
                     13: {'ply': 26.0, 'velocity': 200.0, 'load': 41800.0, 'pressure': 150.0, 'max_force': 62700.0, 'weight': 249.7, 'diameter': 50.0, 'width': 20.0, 'radius': 20.6, 'AR': 0.754},
                     14: {'ply': 26.0, 'velocity': 210.0, 'load': 41800.0, 'pressure': 150.0, 'max_force': 62700.0, 'weight': 230.1, 'diameter': 50.0, 'width': 20.0, 'radius': 20.6, 'AR': 0.754},
                     15: {'ply': 32.0, 'velocity': 225.0, 'load': 53800.0, 'pressure': 190.0, 'max_force': 80700.0, 'weight': 276.8, 'diameter': 50.0, 'width': 20.0, 'radius': 20.6, 'AR': 0.754},
                     16: {'ply': 32.0, 'velocity': 225.0, 'load': 53800.0, 'pressure': 190.0, 'max_force': 80700.0, 'weight': 254.1, 'diameter': 50.0, 'width': 20.0, 'radius': 20.6, 'AR': 0.754},
                     17: {'ply': 34.0, 'velocity': 225.0, 'load': 57000.0, 'pressure': 205.0, 'max_force': 85500.0, 'weight': 293.8, 'diameter': 50.0, 'width': 20.0, 'radius': 20.6, 'AR': 0.754},
                     18: {'ply': 34.0, 'velocity': 225.0, 'load': 57800.0, 'pressure': 185.0, 'max_force': 86700.0, 'weight': 297.6, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.786},
                     19: {'ply': 26.0, 'velocity': 235.0, 'load': 55000.0, 'pressure': 165.0, 'max_force': 82500.0, 'weight': 261.7, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     20: {'ply': 26.0, 'velocity': 235.0, 'load': 55000.0, 'pressure': 165.0, 'max_force': 82500.0, 'weight': 285.4, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     21: {'ply': 28.0, 'velocity': 235.0, 'load': 59500.0, 'pressure': 180.0, 'max_force': 98250.0, 'weight': 300.2, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     22: {'ply': 28.0, 'velocity': 235.0, 'load': 59500.0, 'pressure': 180.0, 'max_force': 89250.0, 'weight': 294.3, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     23: {'ply': 28.0, 'velocity': 235.0, 'load': 59500.0, 'pressure': 180.0, 'max_force': 89250.0, 'weight': 286.8, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     24: {'ply': 30.0, 'velocity': 235.0, 'load': 63700.0, 'pressure': 195.0, 'max_force': 95550.0, 'weight': 306.4, 'diameter': 52.0, 'width': 20.5, 'radius': 21.3, 'AR': 0.711},
                     25: {'ply': 28.0, 'velocity': 210.0, 'load': 46700.0, 'pressure': 150.0, 'max_force': 70050.0, 'weight': 290.4, 'diameter': 50.0, 'width': 21.0, 'radius': 20.2, 'AR': 0.719},
                     26: {'ply': 30.0, 'velocity': 210.0, 'load': 49000.0, 'pressure': 160.0, 'max_force': 73500.0, 'weight': 307.3, 'diameter': 50.0, 'width': 21.0, 'radius': 20.2, 'AR': 0.719},
                     27: {'ply': 14.0, 'velocity': 210.0, 'load': 8200.0, 'pressure': 135.0, 'max_force': 12300.0, 'weight': 29.4, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     28: {'ply': 14.0, 'velocity': 225.0, 'load': 8200.0, 'pressure': 135.0, 'max_force': 12300.0, 'weight': 26.2, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     29: {'ply': 14.0, 'velocity': 225.0, 'load': 8200.0, 'pressure': 135.0, 'max_force': 12300.0, 'weight': 24.9, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     30: {'ply': 16.0, 'velocity': 200.0, 'load': 9725.0, 'pressure': 165.0, 'max_force': 14590.0, 'weight': 32.1, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     31: {'ply': 16.0, 'velocity': 210.0, 'load': 9725.0, 'pressure': 165.0, 'max_force': 14590.0, 'weight': 34.9, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     32: {'ply': 16.0, 'velocity': 210.0, 'load': 9725.0, 'pressure': 165.0, 'max_force': 14590.0, 'weight': 32.7, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     33: {'ply': 16.0, 'velocity': 210.0, 'load': 9700.0, 'pressure': 165.0, 'max_force': 14550.0, 'weight': 28.7, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     34: {'ply': 16.0, 'velocity': 225.0, 'load': 8450.0, 'pressure': 135.0, 'max_force': 12675.0, 'weight': 26.7, 'diameter': 24.2, 'width': 7.7, 'radius': 9.8, 'AR': 0.924},
                     35: {'ply': 14.0, 'velocity': 160.0, 'load': 14000.0, 'pressure': 150.0, 'max_force': 21000.0, 'weight': 49.9, 'diameter': 33.4, 'width': 10.2, 'radius': 14.2, 'AR': 0.856},
                     36: {'ply': 14.0, 'velocity': 160.0, 'load': 14000.0, 'pressure': 150.0, 'max_force': 21000.0, 'weight': 53.2, 'diameter': 33.4, 'width': 10.2, 'radius': 14.2, 'AR': 0.856},
                     37: {'ply': 20.0, 'velocity': 225.0, 'load': 21000.0, 'pressure': 185.0, 'max_force': 31500.0, 'weight': 86.9, 'diameter': 33.4, 'width': 11.5, 'radius': 14.7, 'AR': 0.831},
                     38: {'ply': 22.0, 'velocity': 225.0, 'load': 23300.0, 'pressure': 200.0, 'max_force': 34950.0, 'weight': 98.1, 'diameter': 35.1, 'width': 11.5, 'radius': 14.7, 'AR': 0.831},
                     39: {'ply': 22.0, 'velocity': 225.0, 'load': 23300.0, 'pressure': 200.0, 'max_force': 34950.0, 'weight': 100.5, 'diameter': 35.1, 'width': 11.5, 'radius': 14.7, 'AR': 0.831},
                     40: {'ply': 14.0, 'velocity': 200.0, 'load': 15300.0, 'pressure': 100.0, 'max_force': 22500.0, 'weight': 85.4, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     41: {'ply': 14.0, 'velocity': 200.0, 'load': 15000.0, 'pressure': 100.0, 'max_force': 22500.0, 'weight': 83.6, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     42: {'ply': 16.0, 'velocity': 160.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 90.1, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     43: {'ply': 16.0, 'velocity': 160.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 90.5, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     44: {'ply': 16.0, 'velocity': 200.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 79.5, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     45: {'ply': 16.0, 'velocity': 200.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 85.5, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     46: {'ply': 16.0, 'velocity': 210.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 86.5, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     47: {'ply': 16.0, 'velocity': 210.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 94.2, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     48: {'ply': 16.0, 'velocity': 210.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 86.8, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     49: {'ply': 16.0, 'velocity': 210.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 90.7, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     50: {'ply': 16.0, 'velocity': 225.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 84.3, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     51: {'ply': 16.0, 'velocity': 225.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 96.0, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     52: {'ply': 16.0, 'velocity': 225.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 97.4, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     53: {'ply': 16.0, 'velocity': 225.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 87.6, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862},
                     54: {'ply': 16.0, 'velocity': 225.0, 'load': 17200.0, 'pressure': 115.0, 'max_force': 25800.0, 'weight': 90.7, 'diameter': 38.3, 'width': 13.0, 'radius': 15.8, 'AR': 0.862}}

    load = float(load)
    V_qualified = float(V_qualified)
    speed = min(V_qualified, 235)
    df_tires = pd.DataFrame.from_dict(tire_database, orient='index')
    df_tires_meet_condition = df_tires[(df_tires.load >= load*1.1) & (
        df_tires.velocity >= speed) & (df_tires.pressure <= pneu_pressure)]

    if selection_criteria == 'weight':
        tire_selected = df_tires_meet_condition.loc[df_tires_meet_condition['weight']
                                                    == df_tires_meet_condition['weight'].min(), :]
    elif selection_criteria == 'size':
        df_tire_min_size = df_tires_meet_condition.loc[df_tires_meet_condition['diameter']
                                                       == df_tires_meet_condition['diameter'].min(), :]
        tire_selected = df_tires_meet_condition.loc[df_tires_meet_condition['weight']
                                                    == df_tires_meet_condition['weight'].min(), :]

    tire_diameter = float(tire_selected['diameter'])*in_to_m
    tire_width = float(tire_selected['width'])*in_to_m
    return tire_diameter, tire_width
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# load = np.asarray([6640.94852496])
# pneu_pressure = 200
# selection_criteria = 'weight'
# V_qualified= np.asarray([124.66532282])

# print(tire_selection(load, V_qualified, pneu_pressure, selection_criteria))
