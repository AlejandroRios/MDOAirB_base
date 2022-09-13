"""
MDOAirB

Description:
    - This function calculate the crew salary based on  major US Airlines
    narrowbodies/single aisle, 95% significance
    https://www.airlinepilotcentral.com

Reference: 
    - Reference: 

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

# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def crew_salary(MTOW):
    """
    Description:
        - Calculate the tripulation salary as function of MTOW
    Inputs:
        - MTOW - maximum takeoff weight [kg]
    Outputs:
        - captain_salary - [US$]
        - first_officer_salary - [US$]
        - flight_attendant_salary - [US$]
    """
    A0_captain = -2.2342E+03
    A1_captain = 3.3898E+00
    A0_first_officer = -1.9221E+04
    A1_first_officer = 2.6687E+00
    A0_flight_attendant = -1.1533E+04
    A1_flight_attendant = 1.6012E+00

    captain_salary = A0_captain + A1_captain*MTOW
    first_officer_salary = A0_first_officer + A1_first_officer*MTOW
    flight_attendant_salary = A0_flight_attendant + A1_flight_attendant*MTOW
    return captain_salary, first_officer_salary, flight_attendant_salary
# =============================================================================
# MAIN
# =============================================================================
print(crew_salary(20000))