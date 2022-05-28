import pickle
import numpy as np
import matplotlib.pyplot as plt
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN

def calculate_coefficients_mach(vehicle,switch_neural_network,alpha_deg,CL,h,mach_i,mach_f,n):
    wing = vehicle['wing']
    aircraft = vehicle['aircraft']

    print(wing['maximum_camber'] )
    print(wing['camber_at_maximum_thickness_chordwise_position'])
    
    # wing['area'] =125
    # wing['aspect_ratio']=10
    # wing['taper_ratio']=0.4
    # wing['mean_aerodynamic_chord']=3.84
    # wing['root_incidence']=2
    # wing['kink_incidence']=0
    # wing['tip_incidence']=-2
    # wing['semi_span_kink']=0.4
    # wing['leading_edge_radius'] = [0.1,0.05,0.05]
    # wing['thickness_ratio'] = [0.12,0.10,0.08]
    # wing['thickness_line_angle_trailing_edge'] = [-0.2,-0.2,-0.2]
    # wing['maximum_thickness_chordwise_position'] = [0.3,0.3,0.3]
    # wing['camber_line_angle_leading_edge'] = [0,0,0]
    # wing['camber_line_angle_trailing_edge'] = [-0.005,-0.005,-0.005]
    # wing['maximum_camber'] = [0.03,0.03,0.03]
    # wing['camber_at_maximum_thickness_chordwise_position'] = [0.025,0.025,0.25]

    # # wing['maximum_camber'] = [0.0,0.0,0.0]
    # # wing['camber_at_maximum_thickness_chordwise_position'] = [0.0,0.0,0.0]

    # # wing['maximum_camber'][0] = 0.01
    # # wing['camber_at_maximum_thickness_chordwise_position'][0] = 0.01
    # wing['maximum_camber_chordwise_position'] = [0.7,0.7,0.7]


    print(wing['sweep_leading_edge'])

    ft_to_m = 0.3048

    mach_vec = np.linspace(mach_i,mach_f, n)

    CD_vec = []
    CL_vec = []

    CDfp_vec = []
    CDwave_vec = []
    CDind_vec = []
    CDtot_vec = []

    for i in mach_vec:

        CD_wing, CL_wing, CDfp, CDwave, CDind = aerodynamic_coefficients_ANN(
            vehicle, h*ft_to_m, i, CL, alpha_deg, switch_neural_network)
        CD_vec.append(CD_wing)
        CDfp_vec.append(CDfp)
        CDwave_vec.append(CDwave)
        CDind_vec.append(CDind)
        CL_vec.append(CL_wing)
        friction_coefficient = wing['friction_coefficient']
        CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

        CD_tot = CD_wing + CD_ubrige

        CDtot_vec.append(CD_tot)
    
    return  CDfp_vec, CDwave_vec, CDind_vec, CD_vec, CDtot_vec,CL_vec

# with open('Database/Family/40_to_100/all_dictionaries/'+str(15)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family/101_to_160/all_dictionaries/'+str(20)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

with open('Database/Family/161_to_220/all_dictionaries/'+str(51)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)
vehicle = all_info_acft1['vehicle']
wing = vehicle['wing']
aircraft = vehicle['aircraft']

switch_neural_network = 0
alpha_deg = 5
CL = 0.2
h = 35000
mach_i = 0.1
mach_f = 0.8
n = 30

mach_vec = np.linspace(mach_i,mach_f, n)
wing['sweep_leading_edge'] = 10

CDfp_vec1, CDwave_vec1, CDind_vec1, CD_vec1, CDtot_vec1,CL_vec1 = calculate_coefficients_mach(vehicle,switch_neural_network,alpha_deg,CL,h,mach_i,mach_f,n)

switch_neural_network = 0
alpha_deg = 5
CL = 0.2
h = 0
mach_i = 0.1
mach_f = 0.8

wing['sweep_leading_edge']= 25

CDfp_vec2, CDwave_vec2, CDind_vec2, CD_vec2, CDtot_vec2,CL_vec2= calculate_coefficients_mach(vehicle,switch_neural_network,alpha_deg,CL,h,mach_i,mach_f,n)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)  # using a size in points
plt.rc('legend', fontsize='medium')  # using a named size
plt.rc('axes', labelsize=12, titlesize=12)  # using a size in points

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('CD')
ax.set_ylabel('CL')
# ax.set_title('Activation function: ReLU')

ax.plot(mach_vec, CD_vec1,linewidth=2, label='Sweep c/4 = 10$^\circ$')
ax.plot(mach_vec, CD_vec2,linewidth=2, label='Sweep c/4 = 25$^\circ$')
# ax.plot(CDfp_vec, CL_vec, label='CD parasite')
# ax.plot(CDwave_vec, CL_vec, label='CD wave')
# ax.plot(CDind_vec, CL_vec, label='CD ind')
# ax.plot(CDtot_vec, CL_vec, label='CD tot')
ax.legend()
# ax.set_xlim([0.2,0.8])
# ax.set_ylim([0.01,None])
ax.grid('True')
plt.show()
