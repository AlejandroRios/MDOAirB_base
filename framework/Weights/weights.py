"""
MDOAirB

Description:
    - This code describe the module to obtain the aircraft weight for the following groups:

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
from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
import numpy as np
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
from framework.Performance.Engine.engine_performance import turbofan
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
global deg_to_rad, m2_to_ft2, kg_to_lb, lb_to_kg, m_to_ft, N_to_lbf
deg_to_rad = np.pi/180
m2_to_ft2 = 10.76
kg_to_lb = 2.205
lb_to_kg = 0.4536
m_to_ft = 3.281
kn_to_m_s = 0.5144
N_to_lbf = 0.2248


def wing_mass(vehicle, maximum_takeoff_weight, mach, altitude):
    """
    Description: 
        - Calculates the wing mass in kg
    Reference: 
        - Methodology from Isikveren 2002, pag. 56, eq. 84
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - maximum_takeoff_weight
        - mach - mach number
        - altitude
    Outputs:
        - wing mass [kg]
    """
    # Assiciative constants definition
    safety_factor = 1.5
    YEIS = 2016  # Year of entry into service
    alpha_w = 0.0328
    phi_w = 0.656
    Beta_w = 1.5
    delta_w = 1.5
    epsilon_w = 1.5
    Chi_w = 1.1
    rho_sls_g = 0.125

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    landing_gear_position = wing['position']
    spoilers = aircraft['spoiler_presence']

    _, _, _, _, _, _, _, a = atmosphere_ISA_deviation(
        altitude, 0)

    V_MO = mach*a*kn_to_m_s  # Maximum operating speed [m/s]

    # limit load factor - Roskam, pag. 37, eq 4.23
    n_limit = 2.1 + 24000/((kg_to_lb*maximum_takeoff_weight) + 10000)

    if n_limit <= 2.5:
        n_limit = 2.5

    # wing installation philosophy
    if wing['position'] == 1:
        k_co = 1.17
    else:
        k_co = 1.25

    # spoilers existence correction
    if spoilers == 1:
        k_sp = 1.02
    else:
        k_sp = 1

    # landing gear installation philosophy
    if wing['position'] == 1:
        k_lg = 1.03
    else:
        k_lg = 1.015

    n_ultimate = safety_factor*n_limit
    Pi_ATM = np.exp(2.965 - 0.001525*YEIS)
    Pi_Cw = k_co*k_sp*k_lg
    Pi_tc = 16.5*np.sin(2*np.pi*wing['mean_thickness'])

    tau_s = 1 + 1.31*(((0.5*rho_sls_g*V_MO**2)/(1000))**2) * \
        (1/n_ultimate**3)  # structural stiffness

    aux1 = Pi_ATM*alpha_w*Pi_Cw
    aux2 = maximum_takeoff_weight*n_ultimate*wing['area']*(wing['aspect_ratio']**Beta_w) * \
        (Chi_w + wing['taper_ratio']/2)*(tau_s**delta_w)
    aux2 = float(aux2)
    aux3 = Pi_tc*np.cos(wing['sweep_c_4']*deg_to_rad)**epsilon_w

    return aux1*((aux2/aux3)**phi_w)


def horizontal_tail_mass(V_dive, vehicle):
    """
    Description: 
        - Calculates the horizontal tail mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-Torenbeek, pag. 74, eq. 5.19
    Inputs:
        - V_dive
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - horizontal tail mass [kg]
    """
    horizontal_tail = vehicle['horizontal_tail']

    fin = 'trimmable'  # This should be an input in the future

    fin = 'trimmable'
    if fin == 'trimmable':
        k_h = 1.1
    elif fin == 'fixed':
        k_h = 1

    aux_1 = (3.18*(horizontal_tail['area']*m2_to_ft2)**0.2)*V_dive
    aux_2 = 1000*np.sqrt(np.cos(horizontal_tail['sweep_c_4']*deg_to_rad))
    return (k_h*(horizontal_tail['area']*m2_to_ft2)*((aux_1/aux_2) - 0.287))*lb_to_kg


def vertical_tail_mass(V_dive, vehicle):
    """
    Description: 
        - Calculates the vertical tail mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-Torenbeek, pag. 74, eq. 5.20
    Inputs:
        - V_dive
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - vertical tail mass [kg]
    """

    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']

    horizontal_tail['position'] = 1

    if horizontal_tail['position'] == 1:
        k_v = 1
    if horizontal_tail['position'] == 0:
        z_h = 0.95
        k_v = 1 + 0.15*(((horizontal_tail['area']*m2_to_ft2)*z_h) /
                        ((vertical_tail['area']*m2_to_ft2)*(vertical_tail['span']*m_to_ft)))

    aux_1 = (3.81*((vertical_tail['area']*m2_to_ft2)**0.2))*V_dive
    aux_2 = 1000*np.sqrt(np.cos(vertical_tail['sweep_c_2'] * deg_to_rad))
    return (k_v*(vertical_tail['area']*m2_to_ft2)*((aux_1/aux_2) - 0.287))*lb_to_kg


def fuselage_mass(V_dive, vehicle):
    """
    Description: 
        - Calculates the fuselage mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-Torenbeek, pag. 77, eq. 5.27
    Inputs:
        - V_dive
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - fuselage mass [kg]
    """
    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['horizontal_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']

    V_dive = V_dive/kn_to_m_s

    fuselage_configuration = "pressurized"

    if fuselage_configuration == "pressurized":
        k_f = 1.08
    elif fuselage_configuration == "maing_gear_attached":
        k_f = 1.07
    elif fuselage_configuration == "cargo":
        k_f = 1.10

    if engine['position'] == 2 or engine['position'] == 3:
        k_e = 1.04
    else:
        k_e = 1.0

    if horizontal_tail['position'] == 1:
        horizontal_tail['aerodynamic_center_xposition'] = 0.98*fuselage['length'] - horizontal_tail['center_chord'] + \
            horizontal_tail['mean_aerodynamic_chord_yposition'] * \
            np.tan(horizontal_tail['sweep_leading_edge']*deg_to_rad) + \
            0.25*horizontal_tail['mean_aerodynamic_chord']
    else:
        horizontal_tail['aerodynamic_center_xposition'] = 0.98*fuselage['length'] - vertical_tail['center_chord'] + vertical_tail['span'] * \
            np.tan(vertical_tail['sweep_leading_edge']*deg_to_rad) + 0.25*horizontal_tail['mean_aerodynamic_chord'] + \
            horizontal_tail['mean_aerodynamic_chord_yposition'] * \
            np.tan(horizontal_tail['sweep_leading_edge']*deg_to_rad)

    wing['aerodynamic_center_xposition'] = wing['leading_edge_xposition'] + wing['mean_aerodynamic_chord_yposition'] * \
        np.tan(wing['sweep_leading_edge']*deg_to_rad) + \
        0.25*wing['mean_aerodynamic_chord']

    # arm between wing aerodynamic center an hotizontal tail aerodynamic center
    aircraft['aerodynamic_centers_arm_wing_horizontal'] = horizontal_tail['aerodynamic_center_xposition'] - \
        wing['aerodynamic_center_xposition']

    return (0.021*k_f*k_e * np.sqrt((V_dive*(aircraft['aerodynamic_centers_arm_wing_horizontal']*m_to_ft))/((fuselage['height']*m_to_ft)+(fuselage['width']*m_to_ft))) * (fuselage['wetted_area']*m2_to_ft2)**1.2)*lb_to_kg


def nacelle_mass(vehicle):
    """
    Description: 
        - Calculates the nacelle mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-GD, pag. 79, eq. 5.35
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - nacelle mass [kg]
    """

    engine = vehicle['engine']
    engine_inlet_area = (np.pi*engine['fan_diameter']**2)/4
    aircraft = vehicle['aircraft']
    return (7.435*aircraft['number_of_engines'] * (((engine_inlet_area*m2_to_ft2)**0.5)*(engine['length']*m_to_ft)*engine['compressor_pressure_ratio'])**0.731)*lb_to_kg


def main_landig_gear_mass(maximum_takeoff_weight, vehicle):
    """
    Description: 
        - Calculates the main landing gear mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-Torenbeek, pag. 82, eq. 5.42
    Inputs:
        - maximum_takeoff_weight
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - main landing gear mass [kg]
    """
    wing = vehicle['wing']

    if wing['position'] == 1:
        k_lg = 1
    else:
        k_lg = 1.08

    A_lg = 40
    B_lg = 0.16
    C_lg = 0.019
    D_lg = 1.5E-5

    return (k_lg*(A_lg + B_lg*(maximum_takeoff_weight*kg_to_lb)**(3/4) + C_lg*(maximum_takeoff_weight*kg_to_lb) + D_lg*(maximum_takeoff_weight*kg_to_lb)**(3/2)))*lb_to_kg


def nose_landig_gear_mass(maximum_takeoff_weight, vehicle):
    """
    Description: 
        - Calculates the nose landing gear mass in lb, but the result is converted to kg
    Reference: 
        - Methodology from Roskam-Torenbeek, pag. 82, eq. 5.42
    Inputs:
        - maximum_takeoff_weight
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - nose landing gear mass [kg]
    """
    wing = vehicle['wing']

    if wing['position'] == 1:
        k_lg = 1
    else:
        k_lg = 1.08

    A_lg = 20
    B_lg = 0.1
    C_lg = 0.0
    D_lg = 2E-6

    return (k_lg*(A_lg + B_lg*(maximum_takeoff_weight*kg_to_lb)**(3/4) + C_lg*(maximum_takeoff_weight*kg_to_lb) + D_lg*(maximum_takeoff_weight*kg_to_lb)**(3/2)))*lb_to_kg


def engine_mass(engine_static_thrust, vehicle):
    """
    Description: 
        - Calculates the engine mass in kg
    Reference: 

    Inputs:
        - engine_static_thrust
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - engine mass [kg]
    """
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    return (0.084 * ((engine_static_thrust*N_to_lbf)**1.1)*np.exp(-0.045*engine['bypass']))*lb_to_kg


def engine_mass_bento(engine_static_thrust, vehicle):
    """
    Description: 
        - Calculates the engine mass in kg
    Reference: 

    Inputs:
        - engine_static_thrust
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - engine mass [kg]
    """
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']

    fan_pressure_ratio = engine['fan_pressure_ratio']
    compressor_pressure_ratio = engine['compressor_pressure_ratio']
    bypass_ratio = engine['bypass']
    fan_diameter = engine['fan_diameter']
    turbine_inlet_temperature = engine['turbine_inlet_temperature']
    _, _, vehicle = turbofan(
        0, 0.01, 1, vehicle)

    engine = vehicle['engine']
    mdots = engine['fuel_flows']
    mdot = mdots[0]
    engine_performance_parameters = engine['performance_parameters']
    fgros = engine_performance_parameters[-1]
    BPR = engine['bypass']
    DIA = engine['fan_diameter']
    OPR = engine['compressor_pressure_ratio']
    THR = fgros*mdot/1000
    LEN = engine['length']
    FLO = mdot*(1+BPR)

    BPR0 = 4.6024
    D0 = 1.7316
    OPR0 = 24.8588
    T0 = 129.5858
    L0 = 3.1988
    FL0 = 426.2471

    A = 2637.9004
    B = 47.8549
    C = 127.9729
    a = -0.181
    b = -0.0746
    c = 1.0270
    d = 0.1905
    e = -0.2920
    f = -0.1250

    weight_kg = C+B*((BPR/BPR0))*((OPR/OPR0))*((THR/T0))*((DIA/D0))*((LEN/L0))*((FLO/FL0)) + \
        A*((BPR/BPR0)**a)*((OPR/OPR0)**b)*((THR/T0)**c) * \
        ((DIA/D0)**d)*((LEN/L0)**e)*((FLO/FL0)**f)

    return weight_kg


def fuel_system_mass(vehicle):
    """
    Description: 
        - Calculates the fuel system mass in lb, but the result is converted to kg
    Reference:
        - Methodology from Roskam-Torenbeek, pag. 92, eq. 6.24

    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - fuel system mass [kg]
    """

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    performance = vehicle['performance']

    fuel_type = 3

    if fuel_type == 1:
        k_fsp = 5.870
    elif fuel_type == 2:
        k_fsp = 6.550
    elif fuel_type == 3:
        k_fsp = 6.710

    tanks_number = 3

    if performance['range'] > 2000 or aircraft['number_of_engines'] > 2:
        tanks_number = 6

    return (80*(aircraft['number_of_engines'] + tanks_number - 1) + 15*(tanks_number**0.5) * ((wing['fuel_capacity']*kg_to_lb)/k_fsp)**0.333)*lb_to_kg


def propulsion_system_mass(vehicle, engine_weight):
    """
    Description: 
        - Calculates the nose landing gear mass in lb, but the result is converted to kg
    Reference:
        - Methodology from Roskam-Torenbeek, pag. 93, eq. 6.28 - 6.41

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - engine_weight
    Outputs:
        - propulsion system mass [kg]
    """
    aircraft = vehicle['aircraft']
    engine = vehicle['engine']
    wing = vehicle['wing']
    fuselage = vehicle['fuselage']

    engine_afterburning_presence = 0
    if engine_afterburning_presence == 1:
        k_ec = 1.080
    elif engine_afterburning_presence == 0:
        k_ec = 0.686

    if engine['position'] == 2 or engine['position'] == 3:
        engine_controls = k_ec * \
            ((fuselage['length']*m_to_ft)*aircraft['number_of_engines'])**0.792
    else:
        engine_controls = 88.46 * \
            (((fuselage['length'] + wing['span'])*m_to_ft)
             * (aircraft['number_of_engines'] / 100))**0.294

    engine_starting_system = 38.93*((engine['weight']*kg_to_lb)/1000)**0.918

    engine_reverser_presence = 0

    if engine_reverser_presence == 0:
        engine_reverser = 0
    else:
        engine_reverser = 0.18*engine['weight']*kg_to_lb

    return (engine_controls + engine_starting_system + engine_reverser)*lb_to_kg


def flight_control_system_mass(maximum_takeoff_weight):
    """
    Description: 
        - Calculates the flight control system mass
    Reference:
        - Methodology from Roskam-Torenbeek, pag. 82, eq. 5.42

    Inputs:
        - maximum_takeoff_weight
    Outputs:
        - flight control system mass [kg]
    """
    powered_flight_controls = 1

    if powered_flight_controls == 0:
        k_fc = 0.44
    else:
        k_fc = 0.64

    k_le = 1.2

    return (k_fc*k_le*(maximum_takeoff_weight*kg_to_lb)**(2/3))*lb_to_kg


def fixed_equipment_mass(vehicle, maximum_takeoff_weight, fuel_weight):
    """
    Description: 
        - Calculates the fixed equipment mass in lb, but the result is converted to kg
    Reference:
        - Methodology from Raymer, pag. 459, eq. 15.25 - 15.45 and Roskam-Torenbeek, pag. 105, eq. 7.31 - 7.45

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - maximum_takeoff_weight
        - fuel_weight
    Outputs:
        - fixed equiped mass [kg]
    """
    aircraft = vehicle['aircraft']
    fuselage = vehicle['fuselage']
    wing = vehicle['wing']
    engine = vehicle['engine']
    performance = vehicle['performance']
    systems = vehicle['systems']

    # Hydraulic system
    number_of_functions_performed_by_controls = 7
    hydraulic_system_mass = (0.2673*number_of_functions_performed_by_controls *
                             ((fuselage['length'] + wing['span'])*m_to_ft)**0.937)

    systems['hydraulic_weight'] = hydraulic_system_mass * lb_to_kg

    # Electrical system
    system_electrical_rating = 60  # [kV]

    if engine['position'] == 1 or engine['position'] == 4:
        electrical_routing_distance = fuselage['length']*m_to_ft
    else:
        electrical_routing_distance = (
            fuselage['length'] - aircraft['aerodynamic_centers_arm_wing_horizontal'])*m_to_ft

    electrical_system_mass = 7.291*system_electrical_rating**0.782 * \
        electrical_routing_distance**0.346 * \
        aircraft['number_of_engines'] ** 0.1

    systems['electrical_weight'] = electrical_system_mass*lb_to_kg

    # Avionics
    uninstalled_avionics_mass = 1200
    avionics_mass = 1.73*(uninstalled_avionics_mass**0.983)

    systems['avionics_weight'] = avionics_mass*lb_to_kg

    # Air ice pressure
    air_ice_pressure_system_mass = 6.75 * \
        ((fuselage['cabine_length']*m_to_ft)**1.28)

    systems['air_weight'] = air_ice_pressure_system_mass * lb_to_kg

    # Oxygen system
    oxygen_system_mass = 30 + 1.2*aircraft['passenger_capacity']

    systems['oxygen_weight'] = oxygen_system_mass*lb_to_kg

    # APU
    APU_mass = 0.0085*maximum_takeoff_weight*kg_to_lb
    systems['APU_weight'] = APU_mass*lb_to_kg

    # Furnishing
    furnishing_mass = 0.211 * \
        ((maximum_takeoff_weight - fuel_weight)*kg_to_lb)**0.91

    systems['furnishing_weight'] = furnishing_mass*lb_to_kg

    # Paint
    paint_mass = 0.0045*maximum_takeoff_weight*kg_to_lb

    systems['paint_weight'] = paint_mass*lb_to_kg

    # Safety equipment - pag 141 Jenkinson
    if performance['range'] > 2000:
        value = 3.4
    else:
        value = 0.9

    systems['safety'] = 0.0003*maximum_takeoff_weight*kg_to_lb

    # Handling gear
    systems['handling_gear'] = 3.0E-4 * maximum_takeoff_weight*kg_to_lb

    # Slats
    if aircraft['slat_presence'] == 1:
        if engine['position'] == 2:
            slats_span = 0.8
        else:
            slats_span = 0.7

        slats_area = slats_span*0.15*(wing['area']*m2_to_ft2)
        slats_mass = 3.53*(slats_area**0.82)
    else:
        slats_mass = 0

    return (hydraulic_system_mass + electrical_system_mass + avionics_mass + air_ice_pressure_system_mass + oxygen_system_mass + APU_mass + furnishing_mass + paint_mass + systems['safety'] + systems['handling_gear'] + slats_mass)*lb_to_kg


def aircraft_empty_weight(vehicle, maximum_takeoff_weight, fuel_mass, engine_static_thrust, mach, altitude):
    """
    Description: 
        - Calculates the aircraft empty weight
    Reference:
        -

    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - maximum_takeoff_weight
        - fuel_mass
        - engine_static_thrust
        - mach - mach number
        - altitude
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """
    aircraft = vehicle['aircraft']

    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']
    nose_landing_gear = vehicle['nose_landing_gear']
    main_landing_gear = vehicle['main_landing_gear']
    systems = vehicle['systems']

    # Dive speed
    _, _, sigma, _, _, _, _, a = atmosphere_ISA_deviation(
        altitude, 0)

    V_cruise = mach*a
    V_cruise_keas = V_cruise*sigma**0.5
    V_dive = 1.25*V_cruise_keas

    # Structural weight
    wing['weight'] = wing_mass(vehicle, maximum_takeoff_weight, mach, altitude)

    horizontal_tail['weight'] = horizontal_tail_mass(
        V_dive, vehicle)

    vertical_tail['weight'] = vertical_tail_mass(
        V_dive, vehicle)

    fuselage['weight'] = fuselage_mass(V_dive,
                                       vehicle)

    nacelle['weight'] = nacelle_mass(vehicle)

    main_landing_gear['weight'] = main_landig_gear_mass(
        maximum_takeoff_weight, vehicle)

    nose_landing_gear['weight'] = nose_landig_gear_mass(
        maximum_takeoff_weight, vehicle)

    aircraft['structural_weight'] = wing['weight'] + horizontal_tail['weight'] + vertical_tail['weight'] + \
        fuselage['weight'] + nacelle['weight'] + \
        main_landing_gear['weight'] + nose_landing_gear['weight']

    # Power plant weight
    engine['weight'] = engine_mass(engine_static_thrust, vehicle)

    systems['fuel_weight'] = fuel_system_mass(vehicle)

    systems['propulsion_weight'] = propulsion_system_mass(
        vehicle, engine['weight'])

    aircraft['power_plant_weight'] = aircraft['number_of_engines'] * \
        engine['weight'] + systems['fuel_weight'] + \
        systems['propulsion_weight']

    # Fixed equionet weight
    systems['flight_control_weight'] = flight_control_system_mass(
        maximum_takeoff_weight)

    systems['fixed_equipment_weight'] = fixed_equipment_mass(
        vehicle, maximum_takeoff_weight, fuel_mass)

    aircraft['fixed_equipment_weight'] = systems['flight_control_weight'] + \
        systems['fixed_equipment_weight']

    aircraft['operational_empty_weight'] = aircraft['structural_weight'] + \
        aircraft['power_plant_weight'] + aircraft['fixed_equipment_weight']

    return vehicle

# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# TEST
# =============================================================================
vehicle = initialize_aircraft_parameters()
# maximum_takeoff_weight = 43718
# mach = 0.8
# altitude = 41000

# wing = vehicle['wing']
# wing['mean_thickness'] = 0.1164
# wing['aspect_ratio'] = 8.3
# wing['area'] = 85
# wing['sweep_c_4'] = 21.9
# wing['taper_ratio'] = 0.32
# print(wing_mass(vehicle, maximum_takeoff_weight, mach, altitude))

# V_dive = 312

# print(horizontal_tail_mass(V_dive, vehicle))

# print(vertical_tail_mass(V_dive, vehicle))
engine_static_thrust = 191798
engine = vehicle['engine']
engine['length'] = 2.995


engine['fan_pressure_ratio'] = 1.5
engine['compressor_pressure_ratio'] = 25.8
engine['bypass'] = 4.3
engine['fan_diameter'] = 1.882
engine['turbine_inlet_temperature'] = 1115


print(engine_mass_bento(engine_static_thrust, vehicle))

print(engine_mass(engine_static_thrust, vehicle))

a,b,c =turbofan(0, 0.01, 1, vehicle)
print(a,b)

engine_static_thrust = 104533
engine = vehicle['engine']
engine['length'] = 2.362

engine['fan_pressure_ratio'] = 1.1
engine['compressor_pressure_ratio'] = 30.6
engine['bypass'] = 4.8
engine['fan_diameter'] = 1.524
engine['turbine_inlet_temperature'] = 1706

print(engine_mass_bento(engine_static_thrust, vehicle))

print(engine_mass(engine_static_thrust, vehicle))


a,b,c =turbofan(0, 0.01, 1, vehicle)
print(a,b)