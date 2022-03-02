import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822

#==============================================

def atmosphere(z, Tba):

    '''
    Funçao que retorna a Temperatura, Pressao e Densidade para uma determinada
    altitude z [m]. Essa funçao usa o modelo padrao de atmosfera para a
    temperatura no solo de Tba.
    '''

    # Zbase (so para referencia)
    # 0 11019.1 20063.1 32161.9 47350.1 50396.4

    # DEFINING CONSTANTS
    # Earth radius
    r = 6356766
    # gravity
    g0 = 9.80665
    # air gas constant
    R = 287.05287
    # layer boundaries
    Ht = [0, 11000, 20000, 32000, 47000, 50000]
    # temperature slope in each layer
    A = [-6.5e-3, 0, 1e-3, 2.8e-3, 0]
    # pressure at the base of each layer
    pb = [101325, 22632, 5474.87, 868.014, 110.906]
    # temperature at the base of each layer
    Tstdb = [288.15, 216.65, 216.65, 228.65, 270.65];
    # temperature correction
    Tb = Tba-Tstdb[0]
    # air viscosity
    mi0 = 18.27e-6 # [Pa s]
    T0 = 291.15 # [K]
    C = 120 # [K]

    # geopotential altitude
    H = r*z/(r+z)

    # selecting layer
    if H < Ht[0]:
        raise ValueError('Under sealevel')
    elif H <= Ht[1]:
        i = 0
    elif H <= Ht[2]:
        i = 1
    elif H <= Ht[3]:
        i = 2
    elif H <= Ht[4]:
        i = 3
    elif H <= Ht[5]:
        i = 4
    else:
        raise ValueError('Altitude beyond model boundaries')

    # Calculating temperature
    T = Tstdb[i]+A[i]*(H-Ht[i])+Tb

    # Calculating pressure
    if A[i] == 0:
        p = pb[i]*np.exp(-g0*(H-Ht[i])/R/(Tstdb[i]+Tb))
    else:
        p = pb[i]*(T/(Tstdb[i]+Tb))**(-g0/A[i]/R)

    # Calculating density
    rho = p/R/T

    # Calculating viscosity with Sutherland's Formula
    mi=mi0*(T0+C)/(T+C)*(T/T0)**(1.5)

    return T,p,rho,mi

#==============================================

def plot3d(aircraft):

    xr_w = aircraft['geo_param']['wing']['xr']
    zr_w = aircraft['geo_param']['wing']['zr']
    cr_w = aircraft['dimensions']['wing']['cr']
    xt_w = aircraft['dimensions']['wing']['xt']
    yt_w = aircraft['dimensions']['wing']['yt']
    zt_w = aircraft['dimensions']['wing']['zt']
    ct_w = aircraft['dimensions']['wing']['ct']
    xr_h = aircraft['dimensions']['EH']['xr']
    zr_h = aircraft['geo_param']['EH']['zr']
    cr_h = aircraft['dimensions']['EH']['cr']
    xt_h = aircraft['dimensions']['EH']['xt']
    yt_h = aircraft['dimensions']['EH']['yt']
    zt_h = aircraft['dimensions']['EH']['zt']
    ct_h = aircraft['dimensions']['EH']['ct']
    xr_v = aircraft['dimensions']['EV']['xr']
    zr_v = aircraft['geo_param']['EV']['zr']
    cr_v = aircraft['dimensions']['EV']['cr']
    xt_v = aircraft['dimensions']['EV']['xt']
    zt_v = aircraft['dimensions']['EV']['zt']
    ct_v = aircraft['dimensions']['EV']['ct']
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

#==============================================

def geo_change_sweep(x,y,sweep_x,panel_length,chord_root,chord_tip):

    '''
    This function converts sweep computed at chord fraction x into
    sweep measured at chord fraction y
    (x and y should be between 0 (leading edge) and 1 (trailing edge).
    '''

    sweep_y=sweep_x+np.arctan((x-y)*(chord_root-chord_tip)/panel_length)

    return sweep_y



def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
                

def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '*'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=70)