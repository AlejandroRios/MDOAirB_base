import pickle
import numpy as np
import matplotlib.pyplot as plt
from framework.Aerodynamics.aerodynamic_coefficients_ANN import aerodynamic_coefficients_ANN

def generate_polar(vehicle,switch_neural_network,alpha_deg,mach,h,CLi,CLf,n):
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

    # wing['maximum_camber'] = [0.0,0.0,0.0]
    # wing['camber_at_maximum_thickness_chordwise_position'] = [0.0,0.0,0.0]

    # wing['maximum_camber'][0] = 0.01
    # wing['camber_at_maximum_thickness_chordwise_position'][0] = 0.01
    # wing['maximum_camber_chordwise_position'] = [0.7,0.7,0.7]


    print(wing['sweep_leading_edge'])

    ft_to_m = 0.3048

    CL_vec = np.linspace(CLi,CLf, n)

    CD_vec = []
    # CL_vec = []

    CDfp_vec = []
    CDwave_vec = []
    CDind_vec = []
    CDtot_vec = []

    for i in CL_vec:

        CD_wing, CL_wing, CDfp, CDwave, CDind = aerodynamic_coefficients_ANN(
            vehicle, h*ft_to_m, mach, i, alpha_deg, switch_neural_network)
        CD_vec.append(CD_wing)
        CDfp_vec.append(CDfp)
        CDwave_vec.append(CDwave)
        CDind_vec.append(CDind)
        # CL_vec.append(i)
        friction_coefficient = wing['friction_coefficient']
        CD_ubrige = friction_coefficient * \
        (aircraft['wetted_area'] - wing['wetted_area']) / \
        wing['area']

        CD_tot = CD_wing + CD_ubrige

        CDtot_vec.append(CD_tot)
    
    return  CDfp_vec, CDwave_vec, CDind_vec, CD_vec, CDtot_vec,CL_vec


with open('Database/Family/161_to_220/all_dictionaries/'+str(51)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)
vehicle = all_info_acft1['vehicle']
wing = vehicle['wing']
aircraft = vehicle['aircraft']

switch_neural_network = 0
alpha_deg = 1
CL = 0.5
h = 35000
CLi = 0
CLf= 0.8
mach = 0.2
n = 30

CL_vec = np.linspace(CLi,CLf, n)
wing['sweep_leading_edge'] = 10

CDfp_vec1, CDwave_vec1, CDind_vec1, CD_vec1, CDtot_vec1,CL_vec1= generate_polar(vehicle,switch_neural_network,alpha_deg,mach,h,CLi,CLf,n)
switch_neural_network = 0
alpha_deg = 1
CL = 0.5
h = 35000
CLi = 0.0
CLf= 0.8
mach = 0.8
n = 30



CDfp_vec2, CDwave_vec2, CDind_vec2, CD_vec2, CDtot_vec2,CL_vec2= generate_polar(vehicle,switch_neural_network,alpha_deg,mach,h,CLi,CLf,n)

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

ax.plot(CD_vec1, CL_vec1,linewidth=2, label='CD wing - Mach = 0.2')
# ax.plot(CDfp_vec, CL_vec, label='CD parasite')
# ax.plot(CDwave_vec, CL_vec, label='CD wave')
# ax.plot(CDind_vec, CL_vec, label='CD ind')
# ax.plot(CDtot_vec, CL_vec, label='CD tot')
ax.legend()

# fig = plt.figure(figsize=(10, 9))
# ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Drag coefficient')
ax.set_ylabel('Lift coefficient')
# ax.set_title('Activation function: ReLU')

ax.plot(CD_vec2, CL_vec2,'-' ,linewidth=2,label='CD wing - Mach = 0.8')
# ax.plot(CDfp_vec, CL_vec, '+--' ,label='CD parasite - Mach = 0.8')
# ax.plot(CDwave_vec, CL_vec, '+--' ,label='CD wave - Mach = 0.8')
# ax.plot(CDind_vec, CL_vec, '+--' ,label='CD ind - Mach = 0.8')
# ax.plot(CDtot_vec, CL_vec,'--'  ,label='CD tot - Mach = 0.8')
ax.grid(True)
ax.legend()
ax.set_xlim([0,None])
ax.set_ylim([0,None])


plt.show()
