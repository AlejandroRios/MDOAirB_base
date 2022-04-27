"""
MDOAirB

Description:
    - This functions calculates the net present value. Methodology from Fregnani 2020

Reference: 
    - Reference: Roskam

TODO's:
    - Finish and include this function

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
import matplotlib.pyplot as plt
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def aircraft_price():
    '''
    Methodology from Airbus price list (Narrow Bodies, 1 aisle) 2018
    Inputs:
        - MTOW - maximum takeoff weight - [kg]
        - share - market share [%]
    Outputs:
        - price - [US$]
    TODO's:
        - ask for more especification of WTADJ_price - weight adjusted price
    '''
    x_ref = 0.6
    y_ref = 0.3

    y0 = 3.3383
    mu = 0.0013

    WTADJ_price = y0+mu*MTOW
    discount = y_ref/x_ref*share
    price = WTADJ_price*(1-discount)
    return price*1e6


def market_info():
    '''
    Methodology from
    Inputs:
        -
    Outputs:
        -
    TODO's:
        -
    '''
    share_reference = 0.6
    share_factor = share/share_reference
    price = aircraft_price()
    return share_reference, share_factor, price


def delivery_forecast():
    '''
    Methodology from
    Inputs:
        -
    Outputs:
        -
    '''
    _, share_factor, _ = market_info()
    year = list(range(1, 16))
    deliveries_list = [0, 0, 0, 0, 6, 20, 44, 92, 94, 102, 94, 90, 84, 30, 10]

    deliveries = []
    for i in range(len(deliveries_list)):
        if i > 3:
            deliveries_aux = share_factor*deliveries_list[i]
        else:
            deliveries_aux = deliveries_list[i]

        deliveries.append(deliveries_aux)

    return year, deliveries

# year, deliveries = delivery_forecast()
# plt.plot(year, deliveries)
# plt.show()


def program_share():
    '''
    Methodology from
    Inputs:
        -
    Outputs:
        -
    '''
    manufacturer = 0.7
    partner = 1-manufacturer
    man_power_cost_manufacturer = 41.72
    man_power_cost_partner = 83.44
    man_power_cost_average = manufacturer * \
        man_power_cost_manufacturer + partner*man_power_cost_partner
    return manufacturer, man_power_cost_average


def product_development_cost():
    '''
    Product Development Cost
    Methodology from
    Inputs:
        -
    Outputs:
        -
    '''
    # Man Power Distribution
    man_power_distribution = [0.25, 0.35, 0.30, 0.1]
    # Infrastructure Distribution
    infrastructure_distribution = [0.55, 0.25, 0.15, 0.05]
    # Generic Cost Distribution
    generic_cost_distribution = [0.25, 0.35, 0.40, 0.1]
    return man_power_distribution, infrastructure_distribution, generic_cost_distribution


def complexity_factor():
    '''
    Production Cost - Complexity factor
    Methodology from
    Inputs:
        -
    Outputs:
        -
    '''
    complexity_factors = {}
    complexity_factors['landing_gear'] = 1.2
    complexity_factors['flight_controls'] = 1.4
    complexity_factors['engines'] = 1.1
    complexity_factors['nacelles'] = 1.0
    complexity_factors['interior'] = 1.0
    complexity_factors['electrical'] = 1.3
    complexity_factors['avionics'] = 0.9
    complexity_factors['structures'] = 1.5
    complexity_factors['fuel'] = 1.0
    return complexity_factors

# =============================================================================
# NON-RECURRING COSTS"""
# =============================================================================


def manegment_and_administration_hours(MTOW_factor,thrust_factor):
    '''
    Man Power - Management and Administration hours
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    '''
    preliminar_design_baseline = 27983
    interior_baseline = 229950
    structures_baseline = 279833
    electrics_baseline = 365000
    mechanics_baseline = 121667
    hidrahulics_baseline = 60833
    propulsion_baseline = 51100
    man_power_managment_and_administration = (preliminar_design_baseline
                                              + interior_baseline * MTOW_factor
                                              + structures_baseline * MTOW_factor
                                              + electrics_baseline * MTOW_factor
                                              + mechanics_baseline * MTOW_factor
                                              + hidrahulics_baseline * MTOW_factor
                                              + propulsion_baseline * thrust_factor)
    return man_power_managment_and_administration


def engineering_hours(MTOW_factor, thrust_factor):
    '''
    Man Power - Engineering hours
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    '''

    aerodinamics_baseline = 135050
    aeroelasticity_baseline = 237250
    performance_baseline = 76650
    structures_baseline = 365000
    interiors_baseline = 92467
    electrics_baseline = 365000
    mechanics_baseline = 399067
    hydraulics_baseline = 97333
    propulsion_baseline = 121667
    cetification_baseline = 121667
    configuration_baseline = 62050
    documents_baseline = 115583

    engineering_hours = (aerodinamics_baseline * MTOW_factor
                         + aeroelasticity_baseline * MTOW_factor
                         + performance_baseline * MTOW_factor
                         + structures_baseline * MTOW_factor
                         + interiors_baseline * MTOW_factor
                         + electrics_baseline * MTOW_factor
                         + mechanics_baseline * MTOW_factor
                         + hydraulics_baseline * MTOW_factor
                         + propulsion_baseline * thrust_factor
                         + cetification_baseline * MTOW_factor
                         + configuration_baseline
                         + documents_baseline)

    return engineering_hours


def test_and_validation(MTOW_factor,thrust_factor):
    '''
    Man Power - Test and Validation
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    '''
    ground_testing_baseline = 208050
    systems_testing_baseline = 111933
    flight_test_baseline = 160600
    instrumentation_baseline = 122883
    pilots_baseline = 48667
    prototype_baseline = 2828750
    rigs_baseline = 60833
    body_baseline = 97333
    prototypes_maintenance_baseline = 585217
    manufacturing_tooling_baseline = 526817
    methods_process_baseline = 784750
    quality_control_baseline = 138700

    test_hours = (ground_testing_baseline * MTOW_factor
                  + systems_testing_baseline * MTOW_factor
                  + flight_test_baseline * MTOW_factor
                  + instrumentation_baseline * MTOW_factor
                  + pilots_baseline * MTOW_factor
                  + prototype_baseline * MTOW_factor
                  + rigs_baseline * MTOW_factor
                  + body_baseline * MTOW_factor
                  + prototypes_maintenance_baseline * MTOW_factor
                  + manufacturing_tooling_baseline * MTOW_factor
                  + methods_process_baseline
                  + quality_control_baseline)

    return test_hours


def support_hours(MTOW_factor,thrust_factor):
    '''
    Man Power - Support hours
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - service_bul_baseline - Trabalho de documentação manuais etc.
    '''
    service_bul_baseline = 19467
    spare_parts_baseline = 103417
    operations_manuals_baseline = 203183
    custumer_services_baseline = 6083
    documentation_baseline = 128967
    technical_support_baseline = 77867
    maintenance_plan_baseline = 96117

    support_hours = (service_bul_baseline
                     + spare_parts_baseline * MTOW_factor
                     + operations_manuals_baseline
                     + custumer_services_baseline
                     + documentation_baseline
                     + technical_support_baseline
                     + maintenance_plan_baseline)
    return support_hours

# Infrastructure Cost


def infrastructure(MTOW_factor,thrust_factor):
    '''
    Infrastructure Cost
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - FSW - factory shop and warehouse
    '''
    machine_equipement_baseline = 17880000
    FSW_baseline = 9536000
    prototypes_baseline = 35760000
    assembly_line_baseline = 1192000
    computer_resources_baseline = 6913600
    instruments_baseline = 596000
    office_supplies_baseline = 596000
    software_baseline = 1788000
    others_baseline = 596000

    infrastructure_cost = (machine_equipement_baseline * MTOW_factor
                           + FSW_baseline * MTOW_factor
                           + prototypes_baseline * MTOW_factor
                           + assembly_line_baseline * MTOW_factor
                           + computer_resources_baseline
                           + instruments_baseline
                           + office_supplies_baseline
                           + software_baseline
                           + others_baseline)

    return infrastructure_cost

# General Cost


def general(MTOW_factor,thrust_factor):
    '''
    General costs
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - instrFT_baseline - intrumentation and flight test
        - TPMP_baseline - third part men-power
        - TPMPP_baseline - third part men-power partner
    '''

    communications_baseline = 834400
    instrFT_baseline = 4529600
    tooling_baseline = 2980000
    traveling_baseline = 5006400
    TPMP_baseline = 23840000
    TPMPP_baseline = 27416000
    fuel_baseline = 4872900

    general_cost = (communications_baseline
                    + instrFT_baseline
                    + tooling_baseline
                    + traveling_baseline
                    + TPMP_baseline
                    + TPMPP_baseline
                    + fuel_baseline * MTOW_factor)
    return general_cost


def total_non_recurrig_year_cost(MTOW_factor,thrust_factor):
    '''
    TOTAL NON RECURRING COST PER YEAR
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        -
    '''
    manufacturer, man_power_cost_average = program_share()
    total_hours = manegment_and_administration_hours(MTOW_factor,thrust_factor) + engineering_hours(MTOW_factor,thrust_factor) + \
        test_and_validation(MTOW_factor,thrust_factor) + support_hours(MTOW_factor,thrust_factor)

    hour_cost = total_hours*man_power_cost_average

    man_power_distribution, infrastructure_distribution, generic_cost_distribution = product_development_cost()
    man_power_cost_vector = [i * hour_cost for i in man_power_distribution]

    infrastructure_cost = infrastructure(MTOW_factor,thrust_factor)
    infrastructure_cost_vector = [
        i * infrastructure_cost for i in infrastructure_distribution]

    general_cost = general(MTOW_factor,thrust_factor)
    general_cost_vector = [i * general_cost for i in generic_cost_distribution]

    total_non_recurrig_year_cost = []
    for i in range(len(general_cost_vector)):
        total_non_recurring_cost_aux = manufacturer * \
            (man_power_cost_vector[i] +
             infrastructure_cost_vector[i] + general_cost_vector[i])
        total_non_recurrig_year_cost.append(total_non_recurring_cost_aux)

    return total_non_recurrig_year_cost

# =============================================================================
# RECURRING COSTS
# =============================================================================


def material():
    '''
    MATERIAL COSTS
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        -
    '''
    complexity_factors = complexity_factor()
    materials = {}
    materials['landing_gear'] = 8
    materials['flight_controls'] = 834
    materials['engines'] = 2000000
    materials['naceles'] = 11920
    materials['interior'] = 19072
    materials['electrical_system'] = 8940
    materials['avionics'] = 953600
    materials['structures'] = 143
    materials['fuel'] = 119

    aircraft = {}

    aircraft['number_of_engines'] = 2

    materials_cost = (materials['landing_gear'] * complexity_factors['landing_gear'] * MTOW
                      + materials['flight_controls'] *
                      complexity_factors['flight_controls'] * wing_surface_ft
                      + materials['engines'] * complexity_factors['engines'] *
                      aircraft['number_of_engines'] * thrust_factor
                      + materials['naceles'] *
                      complexity_factors['nacelles'] * engine_diameter_in
                      + materials['interior'] *
                      complexity_factors['interior'] * pax_number
                      + materials['electrical_system'] *
                      complexity_factors['electrical'] * KVA
                      + materials['avionics'] * complexity_factors['avionics']
                      + materials['structures'] *
                      complexity_factors['structures'] * MTOW
                      + materials['fuel'] * complexity_factors['fuel'] * wing_surface_ft)

    return materials_cost


def man_power():
    '''
    MANPOWER COST
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - LCPH - labour cost per hour
        - Baseline's ???
        - why p = 14 - ciclo e 145
    '''
    _, delivers = delivery_forecast()
    LCPH = 25
    baseline = [0, 0, 0, 0, 36714, 26452, 23363, 21451,
                20218, 19416, 18899, 18581, 18389, 18281, 18173]

    recurring_man_power_cost = []
    for i in range(p):
        recurring_man_power_cost_aux = baseline[i] * \
            delivers[i] * MTOW_factor * LCPH
        recurring_man_power_cost.append(recurring_man_power_cost_aux)

    return recurring_man_power_cost


def cost_matrix(MTOW_factor,thrust_factor):
    '''
    Cost Matrix
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        -
    '''
    total_non_recurring_cost = total_non_recurrig_year_cost(MTOW_factor,thrust_factor)
    _, deliveries = delivery_forecast()
    material_costs = material()
    recurring_man_power_cost = man_power()

    costs = []
    for i in range(p):
        if i <= 3:
            costs_aux = total_non_recurring_cost[i]
        else:
            C1 = deliveries[i]*material_costs
            C2 = recurring_man_power_cost[i]
            C3 = (C1 + C2)*0.17
            C4 = (C1 + C2)*0.08
            C5 = C1*0.05
            costs_aux = C1 + C2 + C3 + C4 + C5
        costs.append(costs_aux)
    return costs


def cash_flow_matrix(MTOW_factor,thrust_factor):
    '''
    Cash Flow Matrix
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - This shouldnt perform a summatory? - ans yes!
    '''
    price = aircraft_price()
    _, deliveries = delivery_forecast()
    costs = cost_matrix(MTOW_factor,thrust_factor)

    total_net_present_value = np.zeros(1)
    cash_flow = []
    present_value = []
    for i in range(p):
        sales = deliveries[i]*price
        cash_flow_aux = sales - costs[i]
        present_value_aux = cash_flow_aux/((1+IR)**(i))
        total_net_present_value = total_net_present_value+present_value_aux

        cash_flow.append(cash_flow_aux)
        present_value.append(present_value_aux)

    return total_net_present_value, cash_flow, present_value


def IRR():
    '''
    IRR function
    Methodology from Roskam???
    Inputs:
        -
    Outputs:
        -
    TODO's:
        - Test with other values
        - Reference?
    '''
    _, cash_flow, _ = cash_flow_matrix(MTOW_factor,thrust_factor)
    n = len(cash_flow)
    r = 0
    f = 0

    while f == 0:
        sum = np.zeros(1)
        for i in range(n):
            C = cash_flow[i]/(1+r)**i
            sum = sum+C
        if sum <= 0:
            f = 1
        else:
            r = r+0.001
    return r


def break_even():
    '''
    Break even calculation
    Methodology from Roskam???

    Inputs:
        -
    Outputs:
        -
    TODO's:
        - Test other cases
        - This perform a interpolation
    '''
    _, _, present_value = cash_flow_matrix(MTOW_factor,thrust_factor)
    break_even = np.zeros(1)
    for i in range(1, p):
        x1 = i
        y1 = present_value[i-1]
        x2 = i+1
        y2 = present_value[i]
        x = x1-y1 * (x2-x1)/(y2-y1)
        if y1*y2 < 0:
            break_even = x
            break
    return break_even


# """
# TEST

# TODO's:
#     - This part should be inputed in a different way when integrated
# """
global MTOW, thrust_maximum, wing_surface, engine_diameter, engines_number, pax_number, KVA, p, share, IR

MTOW_baseline = 22010
# wing_surface_baseline = 52
# engines_number_baseline = 2
# engine_diameter_baseline = 1.52
# pax_number_baseline = 50
thrust_maximum_baseline = 8895
# KVA_baseline = 75
p = 14
share = 0.6
IR = 0.05

MTOW = 22000
thrust_maximum = (MTOW/22000)*8895
wing_surface = 50
wing_surface_ft = wing_surface * 10.764
engine_diameter = 1.11
engine_diameter_in = engine_diameter * 39.3701
engines_number = 2
pax_number = 44
KVA = 75


MTOW_factor = MTOW/MTOW_baseline
thrust_factor = thrust_maximum/thrust_maximum_baseline
wing_surface_ft2 = wing_surface*3.28**2
engine_diameter_in = engine_diameter*39.37

total  = total_non_recurrig_year_cost(MTOW_factor,thrust_factor)
print('Total non recurring year cost: ', total)

material_costs = material()
print('Material cost: ', material_costs)

print('Man power cost:', man_power())

print(delivery_forecast())
print('Matrix cost:', cost_matrix(MTOW_factor,thrust_factor))

total_net_present_value, cash_flow, present_value = cash_flow_matrix(MTOW_factor,thrust_factor)


print('Cash flow: ', cash_flow)

print('Present value: ', present_value)

print('Total net present value: ', total_net_present_value)

print('IRR: ', IRR())
print('Break even: ', break_even())
