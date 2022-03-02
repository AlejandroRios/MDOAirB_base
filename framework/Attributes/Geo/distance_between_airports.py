"""
MDOAirB

Description:
    - Haversine distance computation

Reference:

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
import haversine
from haversine import haversine, Unit
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
def haversine_distance(coordinates_departure,coordinates_arrival):
    """
    Description:
        - Perform haversine distance calculation in nautical miles
    Inputs:
        - coordinates_departure - latitude and longitude coordinates departure
        - coordinates_arrival - latitude and longitude coordinates arrival

    Outputs:
        - distance - distance between points [nm]
    """
    # Perform haversine distance calculation in nautical miles
    distance = float(haversine(coordinates_departure,coordinates_arrival,unit='nmi'))
    return distance
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

# # Frankfrt coordinates
# lon_departure = 8.5705
# lat_departure = 50.0333
# # Paris coordinates
# lon_arrival =2.5477
# lat_arrival = 49.0097

# # Conversion to tuple
# coordinates_departure = (lat_departure,lon_departure)
# coordinates_arrival = (lat_arrival,lon_arrival)

# # Perform distance calculation
# distance = haversine_distance(coordinates_departure,coordinates_arrival)