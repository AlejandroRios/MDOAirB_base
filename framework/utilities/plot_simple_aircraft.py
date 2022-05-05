import matplotlib.pyplot as plt
import numpy as np

def plot3d(vehicle):

    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']
    aircraft = vehicle['aircraft']

    delta_x = (wing['center_chord'] - wing['tip_chord'])/4 + wing['semi_span']*np.tan((wing['sweep_c_4']*np.pi)/180)
    zt_w = 0 + wing['semi_span']*(np.tan((wing['dihedral']*np.pi)/180))
    
    cr_h = horizontal_tail['center_chord']
    ct_h = horizontal_tail['taper_ratio']*horizontal_tail['center_chord']
    yt_h = (horizontal_tail['span']/2)

    cr_v = vertical_tail['center_chord']
    ct_v = vertical_tail['tip_chord']
    
    xr_w = wing['leading_edge_xposition']
    zr_w = 0
    cr_w = wing['center_chord']
    xt_w = xr_w + delta_x
    yt_w = wing['semi_span']
    zt_w = zt_w
    ct_w = wing['tip_chord']
    xr_h = horizontal_tail['leading_edge_xposition']
    zr_h = 0
    cr_h = cr_h
    xt_h = xr_h + (cr_h - ct_h)/4 + yt_h*np.tan((horizontal_tail['sweep_c_4']*np.pi)/180)
    yt_h = yt_h
    zt_h = 0
    ct_h = ct_h
    xr_v = vertical_tail['leading_edge_xposition']
    zr_v = 0
    cr_v = cr_v
    ct_v = ct_v
    xt_v = xr_v + cr_v/4 + vertical_tail['span']*np.tan((vertical_tail['sweep_c_4']*np.pi)/180) - ct_v/4
    zt_v = vertical_tail['span']
    
    L_f = fuselage['length']
    D_f = fuselage['diameter']
    x_n = engine['center_of_gravity_xposition']
    y_n = engine['yposition']
    z_n = 0
    L_n = engine['length']
    D_n = engine['diameter'] 
    xcg_0 = aircraft['after_center_of_gravity_xposition']
    xnp = aircraft['neutral_point_xposition']

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

    ax.view_init(azim=0, elev=90)

    plt.show()


# import pickle

# with open('Database/Family/40_to_100/all_dictionaries/'+str(15)+'.pkl', 'rb') as f:
# with open('Database/Family/101_to_160/all_dictionaries/'+str(21)+'.pkl', 'rb') as f:
# with open('Database/Family/161_to_220/all_dictionaries/'+str(60)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)
#     all_info_acft1 = pickle.load(f)


# vehicle = all_info_acft1['vehicle']
# plot3d(vehicle)