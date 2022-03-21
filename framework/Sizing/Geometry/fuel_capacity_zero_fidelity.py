from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================

kg_l_to_kg_m3 = 1000
def zero_fidelity_fuel_capacity(vehicle):
    aircraft = vehicle['aircraft']
    fuselage = vehicle['fuselage']
    wing = vehicle['wing']
    engine= vehicle['engine']
    operations = vehicle['operations']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']

    
    FusWidth = (0.38 * fuselage['seat_abreast_number']  + 1.05 * fuselage['aisles_number'] + 0.55)

    FusHeight = (0.38 * fuselage['seat_abreast_number']  + 1.05 * fuselage['aisles_number'])

    FusLength = (7.9 * FusWidth + 0.0063 * (aircraft['passenger_capacity']/fuselage['seat_abreast_number'])**2.2)

    FusFuel = (0.27*FusLength*FusWidth*max(0,FusHeight-2.2))

    WingMac = (1.2 * np.sqrt(wing['area'] / wing['aspect_ratio']))

    WingSpan = np.sqrt(wing['area'] * wing['aspect_ratio'])

    K = 3*WingSpan*(1.2*np.sqrt(wing['area']/wing['aspect_ratio']))/(4*wing['area'])

    WingTR = 0.8*((2*K-1)-np.sqrt(4*K-3))/(2*(1-K))+0.15 


    WingToCr = -0.030 * operations['mach_cruise']+ 0.180 * np.sqrt(np.cos(wing['sweep_c_4']*np.pi/180))

    WingToCk = -0.028 * operations['mach_cruise']+ 0.140 * np.sqrt(np.cos(wing['sweep_c_4']*np.pi/180))

    WingToCt = -0.016 * operations['mach_cruise']+ 0.120 * np.sqrt(np.cos(wing['sweep_c_4']*np.pi/180)) + 0.625 

    WingFuel = (0.2*wing['area']*WingMac*(5*WingToCr + 3*WingToCk + 2*WingToCt)/10) 

    WingCTkFuel = (1.3 * FusWidth * WingToCr * WingMac**2) 


    HtpVolume = -0.087 * ( aircraft['number_of_engines']*engine['maximum_thrust'] * 1e-5) + 1.05

    HtpLarm = ((0.0002 * FusLength + 0.45) * FusLength)

    HtpArea = 1.012 * (HtpVolume * wing['area'] * WingMac / HtpLarm) + 0.16

    HtpFuel = (0.08 * HtpArea) 

    MFW = (FusFuel + WingCTkFuel + HtpFuel) * operations['fuel_density']*kg_l_to_kg_m3

    return MFW



# vehicle = initialize_aircraft_parameters()
# print(zero_fidelity_fuel_capacity(vehicle))