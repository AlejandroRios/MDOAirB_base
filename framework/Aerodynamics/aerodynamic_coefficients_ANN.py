"""
MDOAirB

Description:
    - This module computes the wing aerodynamic coefficients using a neural network.

TODO's:
    - Rename variables
    - Check issue with dtype object

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
import array
import scipy.io as spio
from sklearn.preprocessing import normalize
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def loadmat(filename):
    '''
    Description:
        this function should be called instead of direct snp.pio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    Description:
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    Description:
        A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def logical(varin):
    if varin == 0:
        varout = 0
    else:
        varout = 1
    return varout


def aerodynamic_coefficients_ANN(vehicle, altitude, mach, CL, alpha_deg,switch_neural_network):
    '''
    Description:
        Update neural network parammeters to be used in ANN_aerodynamics_main
    Inputs:
        - vehicle - dictionary containing aircraft parameters - directory with all relevant information from aircraft
        - altitude - [ft]
        - mach - mach number - mach - mach number number
        - CL - lift coefficient
        - alpha - angle of attack [deg]
        - switch_neural_network - switch to define analysis: 0 for CL | 1 for alpha input
    Outputs:
        - CD - drag coefficient
        - CL - lift coefficient
    '''
    CL_input = CL

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']

    inputs_neural_network = {
        'mach': mach,
        'altitude': altitude,
        'angle_of_attack': alpha_deg*np.pi/180,
        'aspect_ratio': wing['aspect_ratio'],
        'taper_ratio': wing['taper_ratio'],
        'leading_edge_sweep': wing['sweep_leading_edge']*np.pi/180,
        'inboard_wing_dihedral': 2*np.pi/180,
        'outboard_wing_dihedral': 6*np.pi/180,
        'break_position': wing['semi_span_kink'],
        'wing_area': wing['area'],
        'wing_root_airfoil_incidence': wing['root_incidence']*np.pi/180,
        'wing_break_airfoil_incidence': wing['kink_incidence']*np.pi/180,
        'wing_tip_airfoil_incidence': wing['tip_incidence']*np.pi/180,
        'root_airfoil_leading_edge_radius': wing['leading_edge_radius'][0],
        'root_airfoil_thickness_ratio': wing['thickness_ratio'][0],
        'root_airfoil_thickness_line_angle_trailing_edge': wing['thickness_line_angle_trailing_edge'][0],
        'root_airfoil_maximum_thickness_chordwise_position': wing['maximum_thickness_chordwise_position'][0],
        'root_airfoil_camber_line_angle_leading_edge': wing['camber_line_angle_leading_edge'][0],
        'root_airfoil_camber_line_angle_trailing_edge': wing['camber_line_angle_trailing_edge'][0],
        'root_airfoil_maximum_camber': wing['maximum_camber'][0],
        'root_airfoil_camber_at_maximum_thickness_chordwise_position': wing['camber_at_maximum_thickness_chordwise_position'][0],
        'root_airfoil_maximum_camber_chordwise_position ': wing['maximum_camber_chordwise_position'][0],
        'break_airfoil_leading_edge_radius': wing['leading_edge_radius'][1],
        'break_airfoil_thickness_ratio': wing['thickness_ratio'][1],
        'break_airfoil_thickness_line_angle_trailing_edge': wing['thickness_line_angle_trailing_edge'][1],
        'break_airfoil_maximum_thickness_chordwise_position': wing['maximum_thickness_chordwise_position'][1],
        'break_airfoil_camber_line_angle_leading_edge': wing['camber_line_angle_leading_edge'][1],
        'break_airfoil_camber_line_angle_trailing_edge': wing['camber_line_angle_trailing_edge'][1],
        'break_airfoil_maximum_camber': wing['maximum_camber'][1],
        'break_airfoil_camber_at_maximum_thickness_chordwise_position': wing['camber_at_maximum_thickness_chordwise_position'][1],
        'break_airfoil_maximum_camber_chordwise_position ': wing['maximum_camber_chordwise_position'][1],
        'tip_airfoil_leading_edge_radius': wing['leading_edge_radius'][2],
        'tip_airfoil_thickness_ratio': wing['thickness_ratio'][2],
        'tip_airfoil_thickness_line_angle_trailing_edge': wing['thickness_line_angle_trailing_edge'][2],
        'tip_airfoil_maximum_thickness_chordwise_position': wing['maximum_thickness_chordwise_position'][2],
        'tip_airfoil_camber_line_angle_leading_edge': wing['camber_line_angle_leading_edge'][2],
        'tip_airfoil_camber_line_angle_trailing_edge': wing['camber_line_angle_trailing_edge'][2],
        'tip_airfoil_maximum_camber': wing['maximum_camber'][2],
        'tip_airfoil_camber_at_maximum_thickness_chordwise_position': wing['camber_at_maximum_thickness_chordwise_position'][2],
        'tip_airfoil_maximum_camber_chordwise_position ': wing['maximum_camber_chordwise_position'][2]
    }

    '''
    If a new network is to be used from matlab database uncomment lines realted to .mat extensions
    '''
    # NN_induced = loadmat('Aerodynamics/NN_CDind.mat')
    # np.save('NN_induced.npy', NN_induced)
    # NN_wave = loadmat('Aerodynamics/NN_CDwave.mat')
    # np.save('NN_wave.npy', NN_wave)
    # NN_cd0 = loadmat('Aerodynamics/NN_CDfp.mat')
    # np.save('NN_cd0.npy', NN_cd0)
    # NN_CL = loadmat('Aerodynamics/NN_CL.mat')
    # np.save('NN_CL.npy', NN_CL)

    # Load the neural network weights for each of the NN
    NN_induced = np.load('Database/Neural_Network/NN_induced.npy',
                         allow_pickle=True).item()
    NN_wave = np.load('Database/Neural_Network/NN_wave.npy', allow_pickle=True).item()
    NN_cd0 = np.load('Database/Neural_Network/NN_cd0.npy', allow_pickle=True).item()
    NN_CL = np.load('Database/Neural_Network/NN_CL.npy', allow_pickle=True).item()

    CLout, Alpha, CDfp, CDwave, CDind, grad_CL, grad_CDfp, grad_CDwave, grad_CDind = ANN_aerodynamics_main(
        CL_input,
        inputs_neural_network,
        switch_neural_network,
        NN_induced,
        NN_wave,
        NN_cd0,
        NN_CL
    )
    
    # Total wing drag sum
    CDfp = 1.04*CDfp
    CDwing = CDfp + CDwave + CDind

    return CDwing, CLout


def ANN_aerodynamics_main(
    CL_input,
    inputs_neural_network,
    switch_neural_network,
    NN_ind,
    NN_wave,
    NN_cd0,
    NN_CL,
    ):
    '''
    Description:
        This function calculate the aerodynamic coefficients related to the NN's
    Inputs:
        - CL_input - lift coefficient
        - inputs_neural_network - dictionary defining inputs for analysis
        - switch_neural_network - switch to define analysis: 0 for CL | 1 for alpha input
        - NN_ind - induced drag neural network weights
        - NN_wave - wave drag neural network weights
        - NN_cd0 - parasite drag neural network weights
        - NN_CL - lift coefficient neural network weights
    Outputs:
        - CL - lift coefficient
        - alpha - angle of attack [deg]
        - CD_fp - parasite drag coefficient
        - CD_wave - wave drag coefficient
        - CD_ind - indiced drag coefficient
        - grad_CL - gradient of lift coefficient
        - grad_CD_fp - gradient of parasite drag coefficient
        - grad_CD_wave - gradientt of wave drag coefficient
        - grad_CD_ind - gradient of induced drag coefficient

    Translated to python from Matlab.    
    | Soure: Ney Rafael Seccô and Bento Mattos
    | Aeronautical Institute of Technology
    '''


    sizes = len(inputs_neural_network)
    # if sizes != 40 :
    # print('\n The number of input variables should be 40.')
    # print('\n Check the size of input_neural_network columns.')

    m = 1
    # DEFINE VARIABLE BOUNDS
    # Flight conditions
    mach = np.array([0.2, 0.85])  # 1 - Flight Mach number
    altitude = np.array([0, 13000])  # 2 - Flight altitude [m]
    alpha = np.array([-5, 10])*np.pi/180  # 3 - Angle of attack [rad]

    # Wing planform
    aspect_ratio = np.array([7, 12])  # 4 - Aspect ratio
    taper_ratio = np.array([0.2, 0.6])  # 5 - Taper ratio
    # 6 - Leading edge sweep angle np.array([rad])
    leading_edge_sweep = np.array([10, 35])*np.pi/180
    # 7 - Inner panel dihedral np.array([rad])
    dihedral_inner_panel = np.array([0, 5])*np.pi/180
    # 8 - Outer panel dihedral np.array([rad])
    dihedral_outer_panel = np.array([5, 10])*np.pi/180
    # 9 - Span-wise kink position np.array([span fraction])
    span_wise_kink_position = np.array([0.3, 0.6])
    wing_area = np.array([50, 200])  # 10 - Wing area np.array([m?])

    # Airfoil incidences (realtive to fuselage centerline)
    # 11 - Root airfoil incidence np.array([rad])
    root_incidence = np.array([0, 2])*np.pi/180
    # 12 - Kink airfoil incidence np.array([rad])
    kink_incidence = np.array([-1, 1])*np.pi/180
    # 13 - Tip airfoil incidence np.array([rad])
    tip_incidence = np.array([-3, 0])*np.pi/180

    # Root airfoil
    rBA_root = np.array([0.02, 0.20])  # 14 - Leading edge radius
    # EspBF_root = np.array([0.0025 0.0025])### - Trailing edge thickness (constant)
    t_c_root = np.array([0.10, 0.18])  # 15 - Thickness ratio
    # 16 - Thickness line angle at trailing edge
    phi_root = np.array([-0.12, 0.05])*2
    # 17 - Maximum thickness chord-wise position
    X_tcmax_root = np.array([0.20, 0.46])
    # 18 - Camber line angle at leading edge
    theta_root = np.array([-0.20, 0.10])
    # 19 - Camber line angle at trailing edge
    epsilon_root = np.array([-0.300, -0.005])
    Ycmax_root = np.array([-0.05, 0.03])  # 20 - Maximum camber
    # 21 - Maximum camber at maximum thickness chord-wise position
    YCtcmax_root = np.array([-0.05, 0.025])
    # Hte_root = np.array([0 0])### - Trailing edge height with respect to chord-line (constant)
    # 22 - Maximum camber chord-wise position
    X_Ycmax_root = np.array([0.50, 0.80])

    # Kink airfoil
    rBA_kink = np.array([0.03, 0.20])  # 23 - Leading edge radius
    # EspBF_kink = np.array([0.0025 0.0025])### - Trailing edge thickness (constant)
    t_c_kink = np.array([0.08, 0.13])  # 24 - Thickness ratio
    # 25 - Thickness line angle at trailing edge
    phi_kink = np.array([-0.12, 0.05])*2
    # 26 - Maximum thickness chord-wise position
    X_tcmax_kink = np.array([0.20, 0.46])
    # 27 - Camber line angle at leading edge
    theta_kink = np.array([-0.20, 0.10])
    # 28 - Camber line angle at trailing edge
    epsilon_kink = np.array([-0.300, -0.005])
    Ycmax_kink = np.array([0.00, 0.03])  # 29 - Maximum camber
    # 30 - Maximum camber at maximum thickness chord-wise position
    YCtcmax_kink = np.array([0.000, 0.025])
    # Hte_kink = np.array([0 0]) - Trailing edge height with respect to chord-line (constant)
    # 31 - Maximum camber chord-wise position
    X_Ycmax_kink = np.array([0.50, 0.80])

    rBA_tip = np.array([0.03, 0.15])  # 32 - Leading edge radius
    # EspBF_tip = np.array([0.0025 0.0025])### - Trailing edge thickness (constant)
    t_c_tip = np.array([0.08, 0.12])  # 33 - Thickness ratio
    # 34 - Thickness line angle at trailing edge
    phi_tip = np.array([-0.12, 0.05])*2
    # 35 - Maximum thickness chord-wise position
    X_tcmax_tip = np.array([0.20, 0.46])
    # 36 - Camber line angle at leading edge
    theta_tip = np.array([-0.20, 0.10])
    # 37 - Camber line angle at trailing edge
    epsilon_tip = np.array([-0.300, -0.005])
    Ycmax_tip = np.array([0.00, 0.03])  # 38 - Maximum camber
    # 39 - Maximum camber at maximum thickness chord-wise position
    YCtcmax_tip = np.array([0.000, 0.025])
    # Hte_tip = np.array([0 0]) - Trailing edge height with respect to chord-line (constant)
    # 40 - Maximum camber chord-wise position
    X_Ycmax_tip = np.array([0.50, 0.80])

    # intervals = np.concatenate((mach,
    #                            altitude), axis=0)
    intervals = np.vstack((mach,
                           altitude,
                           alpha,
                           aspect_ratio,
                           taper_ratio,
                           leading_edge_sweep,
                           dihedral_inner_panel,
                           dihedral_outer_panel,
                           span_wise_kink_position,
                           wing_area,
                           root_incidence,
                           kink_incidence,
                           tip_incidence,
                           rBA_root,
                           t_c_root,
                           phi_root,
                           X_tcmax_root,
                           theta_root,
                           epsilon_root,
                           Ycmax_root,
                           YCtcmax_root,
                           X_Ycmax_root,
                           rBA_kink,
                           t_c_kink,
                           phi_kink,
                           X_tcmax_kink,
                           theta_kink,
                           epsilon_kink,
                           Ycmax_kink,
                           YCtcmax_kink,
                           X_Ycmax_kink,
                           rBA_tip,
                           t_c_tip,
                           phi_tip,
                           X_tcmax_tip,
                           theta_tip,
                           epsilon_tip,
                           Ycmax_tip,
                           YCtcmax_tip,
                           X_Ycmax_tip))

    input_nn = list(inputs_neural_network.values())

    # for var_index in range(0, 32):
    # if input_nn[var_index] < intervals[var_index, 0] or  input_nn[var_index] > intervals[var_index, 1]:
    # print('\n ==> Warning: variable %d out of boundary limits  \n', var_index)

    if switch_neural_network == 1:
        output_nn, grad_nn = ANN_internal_use(input_nn, NN_CL)
        CL = output_nn[0]
        grad_CL = grad_nn[0]
        Alpha = input_nn[2]
    else:
        Alpha = input_nn[2]
        # print(input_nn)
        output_nn, _ = ANN_internal_use(input_nn, NN_CL)
        CL1 = output_nn[0]
        input_nn[2] = Alpha + np.pi/180
        output_nn, _ = ANN_internal_use(input_nn, NN_CL)
        CL2 = output_nn[0]
        CL_alpha = (CL2-CL1)/(np.pi/180)
        CL0 = CL1-CL_alpha*Alpha
        Alphades = (CL_input-CL0)/CL_alpha
        input_nn[2] = Alphades
        output_nn, grad_nn = ANN_internal_use(input_nn, NN_CL)
        CL = output_nn[0]
        grad_CL = grad_nn[0]
        # print(output_nn)

    del(output_nn, grad_nn)

    output_nn, grad_nn = ANN_internal_use(input_nn, NN_wave)
    CD_wave = output_nn[0]
    CD_wave = max(CD_wave, np.zeros(np.shape(CD_wave)))
    grad_CD_wave = grad_nn
    grad_CD_wave = grad_CD_wave * (np.ones(len(grad_CD_wave))*logical(CD_wave))

    del(output_nn, grad_nn)

    output_nn, grad_nn = ANN_internal_use(input_nn, NN_cd0)
    CD_fp = output_nn[0]
    grad_CD_fp = grad_nn[0]

    del(output_nn, grad_nn)

    output_nn, grad_nn = ANN_internal_use(input_nn, NN_ind)
    CD_ind = output_nn[0]
    grad_CD_ind = grad_nn[0]

    del(output_nn, grad_nn)

    return CL, Alpha, CD_fp, CD_wave, CD_ind, grad_CL, grad_CD_fp, grad_CD_wave, grad_CD_ind


def calculation_alfa(alfa, input_nn, NN, CLinput):
    # Gathering outputs
    input_nn = alfa
    output, doutput_dinput = ANN_internal_use(input_nn, NN)
    CL = output
    delta_alfa = CL-CLinput
    return delta_alfa


def ANN_internal_use(input_nn, NN):
    output = {}
    doutput_dinput = {}

    for nn_index in range(0, 1):
        output, doutput_dinput = feedfoward_gradients(
            input_nn, NN['NN']['theta1'], NN['NN']['theta2'], NN['NN']['theta3'], NN['NN']['Norm_struct'])

    return output, doutput_dinput


def feedfoward_gradients(inputs, theta1, theta2, theta3, norm_struct):

    input_norm = normalize_internal(
        inputs, norm_struct['Mean_input'],
        norm_struct['Range_input']
    )
    # print(norm_struct['Range_input'])
    # mean_iiinputt = np.asarray(norm_struct['Mean_input'])
    # np.savetxt("a0.csv", mean_iiinputt, delimiter=", ")
    m = 1
    if not theta3.size == 0:
        a0 = np.append(np.ones(m), input_norm)
        # print(a0)
        # np.savetxt("a0.csv", a0, delimiter=", ")
        # DANGERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
        z1 = np.dot(theta1, a0)
        z1 = z1.astype(np.float)
        a1 = np.array(
            np.append(np.ones(m), 2/(1+np.exp(-2*z1))-1), dtype=float)
        z2 = np.dot(theta2, a1)
        a2 = np.array(
            np.append(np.ones(m), 2/(1+np.exp(-2*z2))-1), dtype=float)
        output_norm = np.dot(theta3, a2)
    else:
        a0 = [np.ones(m), input_norm]
        z1 = theta1*a0
        a1 = [np.ones(m), 2./(1+np.exp(-2*z1))-1]
        output_norm = theta2@a1

    doutput_norm_da2 = theta3[1:]*np.ones(m)
    doutput_norm_da1 = np.transpose(
        theta2[:, 1:])@(doutput_norm_da2*(1-a2[1:]*a2[1:]))
    doutput_norm_da0 = np.transpose(
        theta1[:, 1:])@(doutput_norm_da1*(1-a1[1:]*a1[1:]))

    output = denormalize_internal(
        output_norm,
        norm_struct['Mean_output'],
        norm_struct['Range_output'],
    )

    aux1 = np.tile(norm_struct['Range_output']*np.ones(m), 40)
    aux2 = norm_struct['Range_input']*np.ones(m)
    doutput_dinput = doutput_norm_da0*aux1/aux2
    return output, doutput_dinput


def normalize_internal(X, mean, Range):
    # Number of training sets
    m = 1
    X = np.asarray(X)
    # Normalization
    X_norm = 2*(X - mean*np.ones(m))/(Range*np.ones(m))
    # print(X_norm)
    return X_norm


def denormalize_internal(X_norm, mean, Range):
    # Number of training sets
    m = 1
    # Denormalization
    # X = np.array([[1, 2, 3], [5, 0, 0]], dtype=object)
    X = X_norm*(Range*np.ones(m))/2 + mean*np.ones(m)
    return X

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
