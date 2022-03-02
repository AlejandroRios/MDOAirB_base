import matplotlib.pyplot as plt


import numpy as np
def plot3d(aircraft):

    xr_w = aircraft['geo_param']['wing']['xr']
    zr_w = aircraft['geo_param']['wing']['zr']
    cr_w = aircraft['geo_param']['wing']['cr']
    xt_w = aircraft['geo_param']['wing']['xt']
    yt_w = aircraft['geo_param']['wing']['yt']
    zt_w = aircraft['geo_param']['wing']['zt']
    ct_w = aircraft['geo_param']['wing']['ct']
    xr_h = aircraft['geo_param']['EH']['xr']
    zr_h = aircraft['geo_param']['EH']['zr']
    cr_h = aircraft['geo_param']['EH']['cr']
    xt_h = aircraft['geo_param']['EH']['xt']
    yt_h = aircraft['geo_param']['EH']['yt']
    zt_h = aircraft['geo_param']['EH']['zt']
    ct_h = aircraft['geo_param']['EH']['ct']
    xr_v = aircraft['geo_param']['EV']['xr']
    zr_v = aircraft['geo_param']['EV']['zr']
    cr_v = aircraft['geo_param']['EV']['cr']
    xt_v = aircraft['geo_param']['EV']['xt']
    zt_v = aircraft['geo_param']['EV']['zt']
    ct_v = aircraft['geo_param']['EV']['ct']
    L_f = aircraft['dimensions']['fus']['Lf']
    D_f = aircraft['dimensions']['fus']['Df']
    x_n = aircraft['dimensions']['nacelle']['xn']
    y_n = aircraft['dimensions']['nacelle']['yn']
    z_n = aircraft['dimensions']['nacelle']['zn']
    L_n = aircraft['dimensions']['nacelle']['Ln']
    D_n = aircraft['dimensions']['nacelle']['Dn']
    xcg_0 = aircraft['dimensions']['fus']['xcg']
    xnp = aircraft['dimensions']['fus']['xnp']
           
    ### PLOT

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    ax.plot([xr_w, xt_w, xt_w+ct_w, xr_w+cr_w, xt_w+ct_w, xt_w, xr_w],
            [0.0, yt_w, yt_w, 0.0, -yt_w, -yt_w, 0.0],
            [zr_w, zt_w, zt_w, zr_w, zt_w, zt_w, zr_w])
    ax.plot([xr_h, xt_h, xt_h+ct_h, xr_h+cr_h, xt_h+ct_h, xt_h, xr_h],
            [0.0, yt_h, yt_h, 0.0, -yt_h, -yt_h, 0.0],
            [zr_h, zt_h, zt_h, zr_h, zt_h, zt_h, zr_h])
    ax.plot([xr_v, xt_v, xt_v+ct_v, xr_v+cr_v, xr_v],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [zr_v, zt_v, zt_v, zr_v, zr_v])

    ax.plot([0.0, L_f],
            [0.0, 0.0],
            [0.0, 0.0])
    ax.plot([x_n, x_n+L_n],
            [y_n, y_n],
            [z_n, z_n])
    ax.plot([x_n, x_n+L_n],
            [-y_n, -y_n],
            [z_n, z_n])

    ax.plot([xcg_0, xcg_0],
            [0.0, 0.0],
            [0.0, 0.0],'o')

    ax.plot([xnp, xnp],
            [0.0, 0.0],
            [0.0, 0.0],'o')

    # Create cubic bounding box to simulate equal aspect ratio
    X = np.array([xr_w, xt_h+ct_h, xt_v+ct_v])
    Y = np.array([-yt_w, yt_w])
    Z = np.array([zt_w, zt_h, zt_v])
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()



def default_aircraft():
    # Defining general geometric parameters
    wing = {'S':93.5,
    		'AR':8.43,
    		'taper':0.235,
    		'sweep':17.45*np.pi/180,
    		'dihedral':5*np.pi/180,
    		'xr':13.5,
    		'zr':0.0,
    		'tcr': 0.123,
    		'tct': 0.096,
    		'c_tank_c_w': 0.4,
    		'x_tank_c_w': 0.2,
            'cr':2,
            'ct':1,
            'xt':10,
            'yt':10,
            'zt':10}
    
    EH  =  {'Cht':0.94,
    		'AR':4.64,
    		'taper':0.39,
    		'sweep':26*np.pi/180,
    		'dihedral':2*np.pi/180,
            'xr':13.5,
    		'Lc':4.83,
    		'zr':0.0,
    		'tcr': 0.1,
    		'tct': 0.1,
    		'eta': 1.0,
            'cr':2,
            'ct':1,
            'xt':10,
            'yt':10,
            'zt':10}
    
    EV  =  {'Cvt':0.088,
    		'AR':1.27,
    		'taper':0.74,
    		'sweep':41*np.pi/180,
    		'Lb':0.55,
    		'zr':0.0,
    		'tcr': 0.1,
    		'tct': 0.1,
            'xr':13.5,
            'cr':2,
            'ct':1,
            'xt':10,
            'yt':10,
            'zt':10}
    
    geo_param = {'wing':wing,
     			 'EH':EH,
     			 'EV':EV}
    
    aircraft = {'geo_param':geo_param}
    
    flap = {'max_def': 0.6981317007977318,
    		'type': 'double slotted',
    		'c_flap_c_wing': 1.2,
    		'b_flap_b_wing': 0.6,}
    
    slat = {'max_def': 0.0,
    		'type': 'slat',
    		'c_slat_c_wing': 1.05,
    		'b_slat_b_wing': 0.75}
    
    engines = {'n': 2,
     		   'n_uw': 0,
               'BPR': 3.04}
    
    misc = {'kexc': 0.03,
    		'rho_f': 804.0,
    		'x_tailstrike': 23.68,
    		'z_tailstrike': -0.84,
            'CLmax_airfoil': 2.3}
    
    weights = {'W_payload': 95519.97000000000116,
               'xcg_payload': 14.4,
               'W_crew': 4463.55000000000018,
               'xcg_crew': 2.5,
               'per_xcg_allelse': 0.45}
    
    aircraft['weights'] = weights
    
    data = {'engines': engines,
    		'flap': flap,
    		'slat': slat,
    		'misc':misc}
    
    aircraft['data'] = data
    
    fus = {'Lf': 32.8,
           'Df': 3.3,
           'xcg':2,
           'xnp':4}
    
    aircraft['dimensions'] = {}
    aircraft['dimensions']['fus'] = fus
    
    nacelle = {'Ln': 4.3,
     		   'Dn': 1.5,
               'xn': 23.2,
               'Ln': 4.3,
               'yn':1.0,
               'zn':1.0}

    
    aircraft['dimensions']['nacelle'] = nacelle
    
    ldg = {'xnlg': 3.6,
           'xmlg': 17.8,
           'ymlg': 2.47,
           'z': -2.0}
    
    aircraft['dimensions']['ldg'] = ldg
        
    return(aircraft)

aircraft = default_aircraft()

# print(aircraft['geo_param']['wing']['xr'])

xr_w = aircraft['geo_param']['wing']['xr']
zr_w = aircraft['geo_param']['wing']['zr']
cr_w = aircraft['geo_param']['wing']['cr']
xt_w = aircraft['geo_param']['wing']['xt']
yt_w = aircraft['geo_param']['wing']['yt']
zt_w = aircraft['geo_param']['wing']['zt']
ct_w = aircraft['geo_param']['wing']['ct']
xr_h = aircraft['geo_param']['EH']['xr']
zr_h = aircraft['geo_param']['EH']['zr']
cr_h = aircraft['geo_param']['EH']['cr']
xt_h = aircraft['geo_param']['EH']['xt']
yt_h = aircraft['geo_param']['EH']['yt']
zt_h = aircraft['geo_param']['EH']['zt']
ct_h = aircraft['geo_param']['EH']['ct']
xr_v = aircraft['geo_param']['EV']['xr']
zr_v = aircraft['geo_param']['EV']['zr']
cr_v = aircraft['geo_param']['EV']['cr']
xt_v = aircraft['geo_param']['EV']['xt']
zt_v = aircraft['geo_param']['EV']['zt']
ct_v = aircraft['geo_param']['EV']['ct']
L_f = aircraft['dimensions']['fus']['Lf']
D_f = aircraft['dimensions']['fus']['Df']
x_n = aircraft['dimensions']['nacelle']['xn']
y_n = aircraft['dimensions']['nacelle']['yn']
z_n = aircraft['dimensions']['nacelle']['zn']
L_n = aircraft['dimensions']['nacelle']['Ln']
D_n = aircraft['dimensions']['nacelle']['Dn']
xcg_0 = aircraft['dimensions']['fus']['xcg']
xnp = aircraft['dimensions']['fus']['xnp']


plot3d(aircraft)