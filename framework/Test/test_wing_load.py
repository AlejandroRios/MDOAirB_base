
from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import atmosphere_ISA_deviation
import matplotlib.pyplot as plt
import numpy as np

g = 9.81
gamma = 1.4
delta_ISA = 0
load_factor = 1.3
weight = 43112

maximum_altitude = 41000
theta, delta, sigma, T_ISA, P_ISA, rho_ISA, a = atmosphere_ISA_deviation(
    maximum_altitude, delta_ISA)
mach_design = 0.78
CL_constraint = ((2)/(gamma*P_ISA*mach_design*mach_design))*6000

print(CL_constraint)
aircraft_data = baseline_aircraft()
wing_surface = wing['area']
# altitude = 20000
# theta, delta, sigma, T_ISA, P_ISA, rho_ISA, a = atmosphere_ISA_deviation(altitude, delta_ISA)


# CL = (2*load_factor*weight*9.81)/(1.4*P_ISA*M*M*S)
delta_altitude = 100
initial_altitude = 100

altitude = initial_altitude
CL = 0

while CL < CL_constraint:

    theta, delta, sigma, T_ISA, P_ISA, rho_ISA, a = atmosphere_ISA_deviation(
        altitude, delta_ISA)
    CL = ((2*load_factor)/(gamma*P_ISA*mach_design*mach_design)) * \
        (weight*9.81/wing_surface)

    altitude = altitude+delta_altitude

    print(altitude, CL)


# # mach = np.linspace(0.5, 1, 100)
# altitude = 10000
# CL_tot = []
# for Mach in (altitude):
#     theta, delta, sigma, T_ISA, P_ISA, rho_ISA, a = atmosphere_ISA_deviation(altitude, delta_ISA)
#     CL = ((2*load_factor)/(gamma*P_ISA*Mach*Mach))*5000

#     CL_tot.append(CL)

# plt.plot(mach, CL_tot)
# plt.show()
