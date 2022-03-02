import numpy as np
from aux_tools import geo_change_sweep, atmosphere


ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
gravity = 9.81


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
    		'x_tank_c_w': 0.2}
    
    EH  =  {'Cht':0.94,
    		'AR':4.64,
    		'taper':0.39,
    		'sweep':26*np.pi/180,
    		'dihedral':2*np.pi/180,
    		'Lc':4.83,
    		'zr':0.0,
    		'tcr': 0.1,
    		'tct': 0.1,
    		'eta': 1.0}
    
    EV  =  {'Cvt':0.088,
    		'AR':1.27,
    		'taper':0.74,
    		'sweep':41*np.pi/180,
    		'Lb':0.55,
    		'zr':0.0,
    		'tcr': 0.1,
    		'tct': 0.1}
    
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
           'Df': 3.3}
    
    aircraft['dimensions'] = {}
    aircraft['dimensions']['fus'] = fus
    
    nacelle = {'Ln': 4.3,
     		   'Dn': 1.5,
               'xn': 23.2,
               'Ln': 4.3}
    
    aircraft['dimensions']['nacelle'] = nacelle
    
    ldg = {'xnlg': 3.6,
           'xmlg': 17.8,
           'ymlg': 2.47,
           'z': -2.0}
    
    aircraft['dimensions']['ldg'] = ldg
        
    return(aircraft)

def geometry(aircraft):
    
    # Separates wing geometry parameters
    S_w = aircraft["geo_param"]['wing']['S'] # Wing area
    AR_w = aircraft["geo_param"]['wing']['AR'] # Wing aspect ratio
    taper_w = aircraft["geo_param"]['wing']['taper'] # Wing taper ratio
    sweep_w = aircraft["geo_param"]['wing']['sweep'] # Wing sweep angle
    dihedral_w = aircraft["geo_param"]['wing']['dihedral'] # Wing dihedral angle
    xr_w = aircraft["geo_param"]['wing']['xr'] # Wing Root x-axis position
    zr_w = aircraft["geo_param"]['wing']['zr'] # Wing Root z-axis position

    # Separates horizontal tail geometry parameters
    Cht = aircraft["geo_param"]['EH']['Cht'] # Horizontal tail volume ratio
    AR_h = aircraft["geo_param"]['EH']['AR'] # Horizontal tail aspect ratio
    taper_h = aircraft["geo_param"]['EH']['taper'] # Horizontal tail taper ratio
    sweep_h = aircraft["geo_param"]['EH']['sweep'] # Horizontal tail sweep angle
    dihedral_h = aircraft["geo_param"]['EH']['dihedral'] # Horizontal tail dihedral
    Lc_h = aircraft["geo_param"]['EH']['Lc'] # Horizontal tail dimensionless lever
    zr_h = aircraft["geo_param"]['EH']['zr'] # Horizontal tail Root z-axis position
    
    # Separates vertical tail geometry parameters
    Cvt = aircraft["geo_param"]['EV']['Cvt'] # Vertical tail volume ratio
    AR_v = aircraft["geo_param"]['EV']['AR'] # Vertical tail aspect ratio
    taper_v = aircraft["geo_param"]['EV']['taper'] # Vertical tail taper ratio
    sweep_v = aircraft["geo_param"]['EV']['sweep'] # Vertical tail sweep angle
    Lb_v = aircraft["geo_param"]['EV']['Lb'] # Vertical tail dimensionless lever
    zr_v = aircraft["geo_param"]['EV']['zr'] # Vertical tail Root z-axis position
 
    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']
    # S_wi = aircraft["geo_param"]['wing']['Si']
    # S_wo = aircraft["geo_param"]['wing']['So']
    # AR_wi = aircraft["geo_param"]['wing']['ARi']
    # AR_wo = aircraft["geo_param"]['wing']['ARo']
    # taper_wi = aircraft["geo_param"]['wing']['taperi']
    # taper_wo = aircraft["geo_param"]['wing']['tapero']
    y_k = aircraft["geo_param"]['wing']['y_k']

    if count <= 0:
        # Calculates wing dimensions
        b_w = np.sqrt(S_w*AR_w) # Wingspan
        cr_w = 2*S_w/(b_w*(1+taper_w)) # Wing root chord
        ct_w = taper_w*cr_w # Wing tip chord
        yt_w = b_w/2 # Wing tip y-axis position
        xt_w = xr_w+yt_w*np.tan(sweep_w)+(cr_w-ct_w)/4 # Wing tip x-axis position
        zt_w = zr_w+yt_w*np.tan(dihedral_w) # Wing tip z-axis position
        cm_w = (2*cr_w/3)*(1+taper_w+taper_w**2)/(1+taper_w) # Wing mean chord
        ym_w = (b_w/6)*(1+2*taper_w)/(1+taper_w) # Wing mean chord y-axis position
        xm_w = xr_w+ym_w*np.tan(sweep_w)+(cr_w-cm_w)/4 # Wing mean chord x-axis position
        zm_w = zr_w+ym_w*np.tan(dihedral_w) # Wing mean chord z-axis position
    else:
        b_w = np.sqrt(S_w*AR_w)
        b_wi = 2*y_k #np.sqrt(S_wi*AR_wi)
        b_wo = b_w-b_wi #np.sqrt(S_wo*AR_wo)
        cr_w_old = 2*S_w/(b_w*(1+taper_w))       
        ct_w = taper_w*cr_w_old
        yt_w = b_w/2
        xt_w = xr_w+yt_w*np.tan(sweep_w)+(cr_w_old-ct_w)/4
        cr_w = cr_w_old + y_k * ((ct_w + xt_w - xr_w - cr_w_old)/yt_w)#2*S_wi/(b_wi*(1+taper_wi))
        cr_wi = cr_w
        cr_wo = cr_w - y_k * ((xt_w - xr_w)/yt_w)#2*S_wo/(b_wo*(1+taper_wo))
        #ct_w = taper_wo*cr_wo
        #yt_w = b_w/2
        #xt_w = xr_w+yt_w*np.tan(sweep_w)+(cr_w-ct_w)/4
        zt_w = zr_w+yt_w*np.tan(dihedral_w)
        taper_wi = cr_wo/cr_wi
        taper_wo = ct_w/cr_wo
        cm_wi = (2*cr_wi/3)*(1+taper_wi+taper_wi**2)/(1+taper_wi)
        cm_wo = (2*cr_wo/3)*(1+taper_wo+taper_wo**2)/(1+taper_wo)
        S_wi = 2 * (cr_wo + cr_wi) * y_k / 2
        S_wo = 2 * (cr_wo + ct_w) * (yt_w - y_k) / 2
        cm_w = (cm_wi*S_wi + cm_wo*S_wo) / (S_wi + S_wo)
        ym_wi = (b_wi/6)*(1+2*taper_wi)/(1+taper_wi)
        ym_wo = (b_wo/6)*(1+2*taper_wo)/(1+taper_wo)
        ym_w = (ym_wi*S_wi + (y_k + ym_wo)*S_wo) / (S_wi + S_wo)
        xm_w = xr_w+ym_w*np.tan(sweep_w)+(cr_w-cm_w)/4
        zm_w = zr_w+ym_w*np.tan(dihedral_w)
        cr_w_eq = 2*(S_wi + S_wo)/b_w-ct_w
        taper_w_eq = ct_w/cr_w_eq
        AR_w_eq = b_w**2/(S_wi + S_wo)
        aircraft['geo_param']['wing']['S_eq'] = S_wi+S_wo
        aircraft['geo_param']['wing']['cr_eq'] = cr_w_eq
        aircraft['geo_param']['wing']['taper_eq'] = taper_w_eq
        aircraft['geo_param']['wing']['AR_eq'] = AR_w_eq
        aircraft['data']['misc']['cr_w_fuel'] = 2*S_w/(b_w*(1+taper_w))
        S_w = S_wi+S_wo

    # Calculates horizontal tail dimensions
    L_h = Lc_h*cm_w # Horizontal tail lever arm
    S_h = S_w*cm_w*Cht/L_h # Horizontal tail area
    b_h = np.sqrt(AR_h*S_h) # Horizontal tail span
    cr_h = 2*S_h/(b_h*(1+taper_h)) # Horizontal tail root chord
    ct_h = taper_h*cr_h # Horizontal tail tip chord
    cm_h = (2*cr_h/3)*(1+taper_h+taper_h**2)/(1+taper_h) # Horizontal tail mean chord
    xm_h = xm_w+L_h+(cm_w-cm_h)/4 # Horizontal tail mean chord x-axis position
    ym_h = b_h/6*(1+2*taper_h)/(1+taper_h) # Horizontal tail mean chord y-axis position
    zm_h = zr_h+ym_h*np.tan(dihedral_h) # Horizontal tail mean chord z-axis position
    xr_h = xm_h-ym_h*np.tan(sweep_h)+(cm_h-cr_h)/4 # Horizontal tail root chord x-axis position
    yt_h = b_h/2 # Horizontal tail root chord y-axis position
    xt_h = xr_h+yt_h*np.tan(sweep_h)+(cr_h-ct_h)/4 # Horizontal tail tip chord x-axis position
    zt_h = zr_h+yt_h*np.tan(dihedral_h) # Horizontal tail tip chord y-axis position
    
    # Calculates vertical tail dimensions
    L_v = Lb_v*b_w # Vertical tail lever arm
    S_v = S_w*b_w*Cvt/L_v # Vertical tail area
    b_v = np.sqrt(AR_v*S_v) # Vertical tail span
    cr_v = 2*S_v/(b_v*(1+taper_v)) # Vertical tail root chord
    ct_v = taper_v*cr_v # Vertical tail tip chord
    cm_v = 2*cr_v/3*(1+taper_v+taper_v**2)/(1+taper_v) # Vertical tail mean chord
    xm_v = xm_w+L_v+(cm_w-cm_v)/4 # Vertical tail mean chord x-axis position
    zm_v = zr_v+b_v/3*(1+2*taper_v)/(1+taper_v) # Vertical tail mean chord z-axis position
    xr_v = xm_v-(zm_v-zr_v)*np.tan(sweep_v)+(cm_v-cr_v)/4 # Vertical tail root chord x-axis position
    zt_v = zr_v+b_v # Vertical tail tip chord z-axis position
    xt_v = xr_v+(zt_v-zr_v)*np.tan(sweep_v)+(cr_v-ct_v)/4 # Vertical tail tip chord x-axis position
        
    # Writes dictionary for wing data
    wing = {'b': b_w,
            'cr': cr_w,
            'ct': ct_w,
            'xt': xt_w,
            'yt': yt_w,
            'zt': zt_w,
            'cm': cm_w,
            'xm': xm_w,
            'ym': ym_w,
            'zm': zm_w}
    
    # Writes dictionary for horizontal tail data
    EH = {'S': S_h,
          'L': L_h,
          'b': b_h,
          'cr': cr_h,
          'ct': ct_h,
          'xr': xr_h,
          'xt': xt_h,
          'yt': yt_h,
          'zt': zt_h,
          'cm': cm_h,
          'xm': xm_h,
          'ym': ym_h,
          'zm': zm_h}
    
    # Writes dictionary for vertical tail data
    EV = {'S': S_v,
          'L': L_v,
          'b': b_v,
          'cr': cr_v,
          'ct': ct_v,
          'xr': xr_v,
          'xt': xt_v,
          'zt': zt_v,
          'cm': cm_v,
          'xm': xm_v,
          'zm': zm_v}
    
    # Groups all dictionaries into "dimensions" list
    dimensions = {'wing': wing,
                  'EH': EH,
                  'EV': EV}
    
    # Ends function returning the "dimensions" list
    return dimensions

def aerodynamics(aircraft, Mach, altitude, n_engines_failed, flap_def, 
                 slat_def, lg_down, h_ground, W0_guess):
    
    geo_param = aircraft['geo_param']
    dimensions = aircraft['dimensions']
    
    S_w = geo_param['wing']['S']
    AR_w = geo_param['wing']['AR']
    cr_w = dimensions['wing']['cr']
    taper_w = geo_param['wing']['taper']
    sweep_w = geo_param['wing']['sweep']
    tcr_w = geo_param['wing']['tcr']
    tct_w = geo_param['wing']['tct']
    b_w = dimensions['wing']['b']
    S_h = dimensions['EH']['S']
    taper_h = geo_param['EH']['taper']
    tcr_h = geo_param['EH']['tcr']
    tct_h = geo_param['EH']['tct']
    S_v = dimensions['EV']['S']
    taper_v = geo_param['EV']['taper']
    tcr_v = geo_param['EV']['tcr']
    tct_v = geo_param['EV']['tct']
    L_f = dimensions['fus']['Lf']
    D_f = dimensions['fus']['Df']
    L_n = dimensions['nacelle']['Ln']
    D_n = dimensions['nacelle']['Dn']
    n_engines = aircraft['data']['engines']['n']
    n_engines_under_wing = aircraft['data']['engines']['n_uw']
    max_flap_def = aircraft['data']['flap']['max_def']
    flap_type = aircraft['data']['flap']['type']
    c_flap_c_wing = aircraft['data']['flap']['c_flap_c_wing']
    b_flap_b_wing = aircraft['data']['flap']['b_flap_b_wing']
    max_slat_def = aircraft['data']['slat']['max_def']
    slat_type = aircraft['data']['slat']['type']
    c_slat_c_wing = aircraft['data']['slat']['c_slat_c_wing']
    b_slat_b_wing = aircraft['data']['slat']['b_slat_b_wing']
    k_exc_drag = aircraft['data']['misc']['kexc']
    CLmax_airfoil = aircraft['data']['misc']['CLmax_airfoil']
    
    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']
    
    # c_flap_c_wing: extended total chord/ retracted total chord

    ### WING
    
    if count <= 0:
        #Exposed Area
        Sexp = S_w - cr_w*D_f
        
    else:
        taper_w = aircraft['geo_param']['wing']['taper_eq']
        AR_w = aircraft['geo_param']['wing']['AR_eq']
        S_w = aircraft['geo_param']['wing']['S_eq']
        #Exposed Area
        Sexp = S_w - cr_w*D_f + (D_f/2)*(D_f/2)*np.tan(sweep_w)/2
        
    #Wetted Area
    tau = tcr_w/tct_w
    Swet_w = 2*Sexp*(1 + 0.25*tcr_w*(1 + tau*taper_w)/(1 + taper_w))

    ### HORIZONTAL TAIL

    #Exposed Area
    Sexp = S_h

    #Wetted Area
    tau = tcr_h/tct_h
    Swet_h = 2*Sexp*(1 + 0.25*tcr_h*(1 + tau*taper_h)/(1 + taper_h))

    ### VERTICAL TAIL

    #Exposed Area
    Sexp = S_v

    #Wetted Area
    tau = tcr_v/tct_v
    Swet_v = 2*Sexp*(1 + 0.25*tcr_v*(1 + tau*taper_v)/(1 + taper_v))

    ### FUSELAGE

    lambda_fus = L_f/D_f

    Swet_f = np.pi*D_f*L_f*(1 - 2/lambda_fus)**(2.0/3.0)*(1 + 1/lambda_fus**2)

    ### NACELLE

    Swet_n = n_engines*np.pi*D_n*L_n

    ### VISCOUS DRAG

    # Total wetted area
    Swet = Swet_w + Swet_h + Swet_v + Swet_f + Swet_n
    # Wetted area ratio
    Sr = Swet/S_w

    # Average t/c
    tc_avg = 0.5*(tcr_w + tct_w)
    # t/c correction
    tau = (Sr-2)/Sr + 1.9/Sr*(1 + 0.526*(4*tc_avg)**3)
    # Other parameters for jet aircraft
    Af = 0.93
    clam = 0.05
    Tf = 1.1

    # Friction coefficient (Howe Eq 6.13)
    Cfe = 0.005*(1-2*clam/Sr)*tau*(1 - 0.2*Mach + 0.12*(Mach*np.sqrt(np.cos(sweep_w))/(Af - tc_avg))**20)*Tf*S_w**(-0.1)
    # Viscous drag
    CD0 = Cfe*Swet/S_w

    ### INDUCED

    # Oswald Factor (Howe Eq 6.14)
    f_taper = 0.005*(1 + 1.5*(taper_w - 0.6)**2)
    e = 1/(1 + 0.12*Mach**6)/(1 + (0.142 + AR_w*(10*tc_avg)**0.33*f_taper)/np.cos(sweep_w)**2 + 0.1*(3*n_engines_under_wing + 1)/(4 + AR_w)**0.8)

    # Induced drag term
    K = 1/np.pi/AR_w/e

    ### GROUND EFFECT
    if h_ground > 0:
        aux = 33*(h_ground/b_w)**1.5
        Kge = aux/(1+aux) # Raymer Eq. 12.61
        K = K*Kge

    ### CLmax
    CLmax_clean = 0.9*CLmax_airfoil*np.cos(sweep_w)

    ### Flaps deflection
    ct_w = cr_w*taper_w
    if max_flap_def > 0.0:
        CD0_flap = 0.0023*b_flap_b_wing*flap_def*180/np.pi # Raymer Eq 12.37
        sweep_flap=geo_change_sweep(0.25, 2-c_flap_c_wing, sweep_w, b_w/2, cr_w, ct_w)
        if flap_type == 'plain':
            dclmax = 0.9
        elif flap_type == 'slotted':
            dclmax = 1.3
        elif flap_type == 'fowler':
            dclmax = 1.3*c_flap_c_wing
        elif flap_type == 'double slotted':
            dclmax = 1.6*c_flap_c_wing
        elif flap_type == 'triple slotted':
            dclmax = 1.9*c_flap_c_wing
        deltaCLmax_flap = dclmax*b_flap_b_wing*np.cos(sweep_flap)*flap_def/max_flap_def # Raymer Eq 12.21
    else:
        CD0_flap = 0.0
        deltaCLmax_flap = 0.0

    ### Slats deflection
    if max_slat_def > 0.0:
        CD0_slat = 0.0023*b_slat_b_wing*slat_def*180/np.pi # Raymer Eq 12.37
        sweep_slat=geo_change_sweep(0.25, c_slat_c_wing-1, sweep_w, b_w/2, cr_w, ct_w)
        if slat_type == 'fixed':
            dclmax = 0.2
        elif slat_type == 'flap':
            dclmax = 0.3
        elif slat_type == 'kruger':
            dclmax = 0.3
        elif slat_type == 'slat':
            dclmax = 0.4*c_slat_c_wing
        deltaCLmax_slat = dclmax*b_slat_b_wing*np.cos(sweep_slat)*slat_def/max_slat_def # Raymer Eq 12.21
    else:
        CD0_slat = 0.0
        deltaCLmax_slat = 0.0

    # Maximum lift
    CLmax = CLmax_clean + deltaCLmax_flap + deltaCLmax_slat

    ### Landing gear (ESDU)
    lg_factor = (0.57 - 0.26*flap_def/max_flap_def)*1e-3
    CD0_lg = lg_down*lg_factor*(W0_guess/9.81)**0.785/S_w

    ### Windmill engine
    #Vn_V = 0.42
    #CDwdm = (0.0785*D_n**2 + 1/(1 + 0.16*Mach**2)*np.pi/2*D_n**2*Vn_V*(1-Vn_V))/S_w
    #CD0_wdm = n_engines_failed*CDwdm
    CD0_wdm = n_engines_failed*0.3*np.pi/4*D_n**2/S_w # Raymer Eq 12.41

    # Add all drag values found so far
    CD0 = CD0 + CD0_flap + CD0_slat + CD0_lg + CD0_wdm

    ### Excrescence
    CD0_exc = CD0*k_exc_drag/(1-k_exc_drag)
    CD0 = CD0 + CD0_exc

    ### WAVE DRAG (Korn Equation)
    if Mach > 0.5:
        # Estimate flight CL
        T,p,rho,mi = atmosphere(altitude, 288.15)
        a = np.sqrt(1.4*287*T)
        V = a*Mach
        CL = 2*W0_guess/rho/V**2/S_w
    
        Mach_dd = 0.95/np.cos(sweep_w) - tc_avg/np.cos(sweep_w)**2 - CL/10/np.cos(sweep_w)**3
        Mach_crit = Mach_dd - (0.1/80)**(1/3)
        if Mach > Mach_crit:
            CDw = 20*(Mach - Mach_crit)**4
        else:
            CDw = 0.0
    else:
        CDw = 0.0

    CD0 = CD0 + CDw
    
    aero = {'CD0': CD0, 'K': K, 'Swet_f': Swet_f}

    aero['Swet_w'] = Swet_w
    aero['Swet_h'] = Swet_h
    aero['Swet_n'] = Swet_n
    aero['Swet_v'] = Swet_v
    return aero, CLmax

#------------------------------------------------------------------------------

def engineTSFC(BPR, Mach, altitude):

    # Atmospheric conditions at cruise altitude
    T,p,rho,mi = atmosphere(altitude, 288.15)

    # Density ratio
    sigma = rho/1.225

    # Base TSFC
    if BPR < 4.0:
        Cbase = 0.85/3600
    else:
        Cbase = 0.70/3600

    # Howe Eq 3.12a
    C = Cbase * (1 - 0.15 * BPR ** 0.65) * (1 + 0.28 * (1 + 0.063 * BPR ** 2) * Mach) * sigma ** 0.08

    return C

#------------------------------------------------------------------------------

def empty_weight(aircraft, W0_guess, T0_guess):
    
    geo_param = aircraft['geo_param']
    dimensions = aircraft['dimensions']
    
    S_w = geo_param['wing']['S']
    AR_w = geo_param['wing']['AR']
    taper_w = geo_param['wing']['taper']
    sweep_w = geo_param['wing']['sweep']

    xm_w = dimensions['wing']['xm']
    cm_w = dimensions['wing']['cm']
    tcr_w = geo_param['wing']['tcr']
    xr_w = geo_param['wing']['xr']
    
    S_h = dimensions['EH']['S']
    xm_h = dimensions['EH']['xm']
    cm_h = dimensions['EH']['cm']
    
    S_v = dimensions['EV']['S']
    xm_v = dimensions['EV']['xm']
    cm_v = dimensions['EV']['cm']
    
    L_f = dimensions['fus']['Lf']
    Swet_f = dimensions['fus']['Swet']
    
    n_engines = aircraft['data']['engines']['n']
    BPR = aircraft['data']['engines']['BPR']
    
    x_n_rel = dimensions['nacelle']['xn_xr']
    L_n = dimensions['nacelle']['Ln']
    
    x_nlg = dimensions['ldg']['xnlg']
    dx_mlg = dimensions['ldg']['dx_mlg_wing']
    
    per_xcg_allelse = aircraft['weights']['per_xcg_allelse']

    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']
    
    ### WING
    if count >= 1:
        taper_w = aircraft['geo_param']['wing']['taper_eq']
        AR_w = aircraft['geo_param']['wing']['AR_eq']
        S_w = aircraft['geo_param']['wing']['S_eq']
     
    # Raymer Eq 15.25
    # I increased the AR_w exponent from 0.5 to 0.55 to make it more sensitive.
    # Otherwise, the optimum would be around AR = 12, which may be too optimistic.
    Nz = 1.5*2.5 # Ultimate load factor
    Scsw = 0.15*S_w # Area of control surfaces
    W_w = 0.75*0.0051*(W0_guess*Nz/lb2N)**0.557*(S_w/ft2m**2)**0.649*AR_w**0.55*tcr_w**(-0.4)*(1+taper_w)**0.1/np.cos(sweep_w)*(Scsw/ft2m**2)**0.1*lb2N
    xcg_w = xm_w + 0.4*cm_w

    # Raymer Tab 15.2

    W_h = 0.75*S_h*gravity*27
    xcg_h = xm_h + 0.4*cm_h

    W_v = 0.75*S_v*gravity*27
    xcg_v = xm_v + 0.4*cm_v

    W_f = 0.85*Swet_f*gravity*24
    xcg_f = 0.45*L_f

    W_nlg = 0.15*W0_guess*0.043
    xcg_nlg = x_nlg
    
    x_mlg = dx_mlg + xr_w
    W_mlg = 0.85*W0_guess*0.043
    xcg_mlg = x_mlg

    # Engine weight
    T_eng = T0_guess/n_engines
    W_eng = gravity * 2857.6 #gravity*14.7*(T_eng/1000.0)**1.1*np.exp(-0.045*BPR)
    x_n = x_n_rel + xr_w
    W_eng_installed = n_engines*W_eng*1.3
    xcg_eng = x_n + 0.5*L_n

    # All else weight
    W_allelse = 0.17*W0_guess
    xcg_allelse = per_xcg_allelse*L_f

    # Empty weight
    We = W_w + W_h + W_v + W_f + W_nlg + W_mlg + W_eng_installed + W_allelse

    # Empty weight CG
    xcg_e = (W_w*xcg_w + W_h*xcg_h + W_v*xcg_v + W_f*xcg_f +
             W_nlg*xcg_nlg + W_mlg*xcg_mlg + W_eng_installed*xcg_eng +
             W_allelse*xcg_allelse)/We
    
    weightsvec = [W_w,  W_h, W_v, W_f, W_nlg, W_mlg, W_eng_installed, W_allelse]
     
    return We, xcg_e, weightsvec

#------------------------------------------------------------------------------

def fuel_weight(aircraft, W0_guess,
                CD0_cruise, K_cruise, altitude_cruise, Mach_cruise, range_cruise, C_cruise,
                loiter_time,
                CD0_altcruise, K_altcruise, altitude_altcruise, Mach_altcruise, range_altcruise, C_altcruise):
    
    geo_param = aircraft['geo_param']
    
    S_w = geo_param['wing']['S']
    
    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']
    if count >= 1:
        S_w = aircraft['geo_param']['wing']['S_eq']
    
    # Initialize product of all phases
    Mf = 1.0

    ### Start and warm-up
    Mf = Mf*0.99

    ### Taxi
    Mf = Mf*0.99

    ### Take-off
    Mf = Mf*0.995

    ### Climb
    Mf = Mf*0.98

    ### Cruise

    # Store weight fraction at beginning of the cruise
    Mf_cruise = Mf

    # Atmospheric conditions at cruise altitude
    T,p,rho,mi = atmosphere(altitude_cruise, 288.15)

    # Cruise speed
    a_cruise = np.sqrt(1.4*287*T)
    v_cruise = Mach_cruise*a_cruise

    # Cruise CL
    CL = 2.0*W0_guess*Mf/rho/S_w/v_cruise**2

    # Cruise C
    CD = CD0_cruise + K_cruise*CL**2
    
    Mf = Mf*np.exp(-range_cruise*C_cruise/v_cruise*CD/CL)

    ### Loiter

    # Loiter at max L/D
    LDmax = 0.5/np.sqrt(CD0_cruise*K_cruise)

    # Factor to fuel comsumption
    C_loiter = C_cruise*0.4/0.5

    Mf = Mf*np.exp(-loiter_time*C_loiter/LDmax)

    ### Descent
    Mf = Mf*0.99

    ### Cruise 2

    # Atmospheric conditions at cruise altitude
    T,p,rho,mi = atmosphere(altitude_altcruise, 288.15)

    # Cruise speed
    a_altcruise = np.sqrt(1.4*287*T)
    v_altcruise = Mach_altcruise*a_altcruise
    #v_altcruise = 128
    # Cruise CL
    CL = 2.0*W0_guess*Mf/rho/S_w/v_altcruise**2

    # Cruise CD
    CD = CD0_altcruise + K_altcruise*CL**2
    
    Mf = Mf*np.exp(-range_altcruise*C_altcruise/v_altcruise*CD/CL)
    
    ### Landing and Taxi
    Mf = Mf*0.992
    
    ### Fuel weight (Raymer Eq 3.13)
    Wf = 1.06*(1-Mf)*W0_guess

    return Wf, Mf_cruise

#------------------------------------------------------------------------------

def weight(aircraft, W0_guess, T0_guess,
           altitude_cruise, Mach_cruise, range_cruise,
           loiter_time,
           altitude_altcruise, Mach_altcruise, range_altcruise):

    
    W_payload = aircraft['weights']['W_payload']
    W_crew = aircraft['weights']['W_crew']

    BPR = aircraft['data']['engines']['BPR']
    
       
    # Set iterator
    delta = 1000

    while abs(delta) > 100:
        # Get cruise aerodynamic data
        aero, _ = aerodynamics(aircraft, Mach_cruise, altitude_cruise, 0., 0., 
                               0., 0., 0., W0_guess)

        CD0 = aero['CD0']
        K = aero['K']
        aircraft['dimensions']['fus']['Swet'] = aero['Swet_f']
            
        aeroalt, _ = aerodynamics(aircraft, Mach_altcruise, altitude_altcruise, 0., 0., 
                                  0., 0., 0., W0_guess)
        CD0_altcruise = aeroalt['CD0']
        K_altcruise = aeroalt['K']
            
            
        # Get engine TSFC
        C_cruise = engineTSFC(BPR, Mach_cruise, altitude_cruise)
        C_altcruise = engineTSFC(BPR, Mach_altcruise, altitude_altcruise)

        We, xcg_e, weightsvec = empty_weight(aircraft, W0_guess, T0_guess)

        

        Wf, Mf_cruise = fuel_weight(aircraft, W0_guess,
                                    CD0, K, altitude_cruise, Mach_cruise, range_cruise, C_cruise,
                                    loiter_time,
                                    CD0_altcruise, K_altcruise, altitude_altcruise, Mach_altcruise, range_altcruise, C_altcruise)
        

        W0 = We + Wf + W_payload + W_crew
        

        delta = W0 - W0_guess
        
        W0_guess = W0
        aircraft['weights']['W0_guess'] = W0

    return W0, We, Wf, Mf_cruise, xcg_e, weightsvec

#------------------------------------------------------------------------------

def performance(aircraft, 
                TO_flap_def, LD_flap_def, 
                TO_slat_def, LD_slat_def,
                h_ground, 
                altitude_takeoff, distance_takeoff, 
                altitude_landing, distance_landing, MLW_frac, 
                altitude_cruise, Mach_cruise):

    geo_param = aircraft['geo_param']
    
    W0 = aircraft['weights']['W0']
    S_w = geo_param['wing']['S']
    n_engines = aircraft['data']['engines']['n']
    BPR = aircraft['data']['engines']['BPR']
    Mf_cruise = aircraft['data']['misc']['Mf_cruise']
    
    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']
    if count >= 1:
        S_w = aircraft['geo_param']['wing']['S_eq']
    
    '''
    This function computes the required thrust and wing areas
    required to meet takeoff, landing, climb, and cruise requirements.

    OUTPUTS:
    T0: real -> Total thrust required to meet all mission phases
    S_wlan: real -> Wing area required for landing. The wing area (S_w) should
                   be greater than this value.
    '''

    ### TAKEOFF

    # Compute air density at takeoff altitude
    T,p,rho,mi = atmosphere(altitude_takeoff, 288.15)

    # density ratio
    sigma = rho/1.225

    # Takeoff aerodynamics
    W0_guess = W0
    aeroTO, CLmaxTO = aerodynamics(aircraft, 0.2, altitude_takeoff, 0, TO_flap_def, TO_slat_def, 0, h_ground, W0_guess)

    T0W0 = 0.2387/sigma/CLmaxTO/distance_takeoff*W0/S_w

    T0_to = T0W0*W0

    ### LANDING

    # Compute air density at landing altitude
    T,p,rho,mi = atmosphere(altitude_landing, 288.15)

    # Landing aerodynamics
    W0_guess = W0*MLW_frac
    aeroLD, CLmaxLD = aerodynamics(aircraft, 0.2, altitude_landing, 0, LD_flap_def, LD_slat_def, 0, h_ground, W0_guess)


    # Landing Field Length (Roskam)
    # sfl = distance_landing/0.6

    # Approach speed (Roskam adapted to SI)
    Va = 1.701*np.sqrt(distance_landing)

    # Required stall speed
    Vs = Va/1.3

    # Required wing area
    S_wlan = 2*W0*MLW_frac/rho/Vs**2/CLmaxLD

    ### CRUISE

    # Compute air density at cruise altitude
    T,p,rho,mi = atmosphere(altitude_cruise, 288.15)

    # Cruise aerodynamics
    W0_guess = W0*Mf_cruise
    aeroCR, _ = aerodynamics(aircraft, Mach_cruise, altitude_cruise, 0, 0.0, 0.0, 0.0, 0.0, W0_guess)
    CD0 = aeroCR['CD0']
    K = aeroCR['K']

    # Cruise speed
    a_cruise = np.sqrt(1.4*287*T)
    v_cruise = Mach_cruise*a_cruise

    # Cruise CL
    CL = 2.0*W0*Mf_cruise/rho/S_w/v_cruise**2

    # Cruise C
    CD = CD0 + K*CL**2

    # Cruise traction
    T = 0.5*rho*v_cruise**2*S_w*CD

    # Cruise traction correction for takeoff conditions
    kT = (0.0013*BPR-0.0397)*altitude_cruise/1000.0 - 0.0248*BPR + 0.7125

    # Corrected thrust
    T0_cruise = T/kT

    ### CLIMB

    # Define standard function for climb analysis
    def climb_analysis(grad, Ks, altitude, CLmax_guess,
                       lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                       kT):

        '''
        We need a guess for CLmax just to get an approximate drag polar for
        speed computation. We will get the correct CLmax from the aerodynamics module

        kT: Thrust decay factor (e.g. use 0.94 for maximum continuous thrust)
        '''

        # Compute air density
        T,p,rho,mi = atmosphere(altitude, 288.15)

        # Compute stall speed
        Vs = np.sqrt(2*W0*Mf/rho/S_w/CLmax_guess)

        # Compute climb speed
        Vclimb = Vs*Ks

        # Compute sound speed and Mach number
        a = np.sqrt(1.4*287*T)
        Mach = Vclimb/a

        # Get aerodynamic data aerodynamics
        W0_guess = W0*Mf
        aeroCL, CLmax = aerodynamics(aircraft, Mach, altitude, n_engines_failed, 
                                     flap_def, slat_def, lg_down, h_ground_climb, W0_guess)

        CD0 = aeroCL['CD0']
        K = aeroCL['K']

        # Get climb CL
        CL = CLmax/Ks**2

        # Get corresponding CD
        CD = CD0 + K*CL**2

        # Compute T/W
        TW = n_engines/(n_engines-n_engines_failed)*(grad + CD/CL)

        # Compute required traction
        T0 = TW*W0*Mf/kT

        return T0

    # FAR 25.111
    grad = 0.012
    Ks = 1.2
    altitude = altitude_takeoff
    CLmax_guess = CLmaxTO
    lg_down = 0
    h_ground_climb = h_ground
    flap_def = TO_flap_def
    slat_def = TO_slat_def
    n_engines_failed = 1
    Mf = 1.0
    kT = 1.0
    T0_1 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # FAR 25.121a
    grad = 0.0
    Ks = 1.1
    altitude = altitude_takeoff
    CLmax_guess = CLmaxTO
    lg_down = 1
    h_ground_climb = h_ground
    flap_def = TO_flap_def
    slat_def = TO_slat_def
    n_engines_failed = 1
    Mf = 1.0
    kT = 1.0
    T0_2 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # FAR 25.121b
    grad = 0.024
    Ks = 1.2
    altitude = altitude_takeoff
    CLmax_guess = CLmaxTO
    lg_down = 0
    h_ground_climb = 0
    flap_def = TO_flap_def
    slat_def = TO_slat_def
    n_engines_failed = 1
    Mf = 1.0
    kT = 1.0
    T0_3 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # FAR 25.121c
    grad = 0.012
    Ks = 1.25
    altitude = altitude_takeoff
    CLmax_guess = CLmaxTO
    lg_down = 0
    h_ground_climb = 0
    flap_def = 0.0
    slat_def = 0.0
    n_engines_failed = 1
    Mf = 1.0
    kT = 0.94
    T0_4 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # FAR 25.119
    grad = 0.032
    Ks = 1.30
    altitude = altitude_landing
    CLmax_guess = CLmaxLD
    lg_down = 1
    h_ground_climb = 0
    flap_def = LD_flap_def
    slat_def = LD_slat_def
    n_engines_failed = 0
    Mf = MLW_frac
    kT = 1.0
    T0_5 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # FAR 25.121d
    grad = 0.021
    Ks = 1.40
    altitude = altitude_landing
    CLmax_guess = CLmaxLD
    lg_down = 1
    h_ground_climb = 0
    flap_def = LD_flap_def*0.8
    slat_def = LD_slat_def*0.8
    n_engines_failed = 1
    Mf = MLW_frac
    kT = 1.0
    T0_6 = climb_analysis(grad, Ks, altitude, CLmax_guess,
                          lg_down, h_ground_climb, flap_def, slat_def, n_engines_failed, Mf,
                          kT)

    # Get the maximum required thrust
    T0vec = [T0_to, T0_cruise, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6]
    T0 = 1.05 * max(T0vec)

    return T0, T0vec, S_wlan

#------------------------------------------------------------------------------

def thrust_matching(aircraft, W0_guess, T0_guess,
                    TO_flap_def, LD_flap_def,
                    TO_slat_def, LD_slat_def,
                    h_ground,
                    altitude_cruise, Mach_cruise, range_cruise,
                    loiter_time,
                    altitude_altcruise, Mach_altcruise, range_altcruise,
                    altitude_takeoff, distance_takeoff,
                    altitude_landing, distance_landing, MLW_frac):


    # Set iterator
    delta = 1000

    # Loop to adjust T0
    while abs(delta) > 100:

        W0, We, Wf, Mf_cruise, xcg_e, _ = weight(aircraft, W0_guess, T0_guess,
                                                 altitude_cruise, Mach_cruise, range_cruise,
                                                 loiter_time,
                                                 altitude_altcruise, Mach_altcruise, range_altcruise)
        aircraft['weights']['W0'] = W0
        aircraft['weights']['We'] = We
        aircraft['weights']['Wf'] = Wf
        aircraft['data']['misc']['Mf_cruise'] = Mf_cruise
        aircraft['weights']['xcg_e'] = xcg_e
        T0, T0vec, S_wlan = performance(aircraft, 
                                        TO_flap_def, LD_flap_def, 
                                        TO_slat_def, LD_slat_def,
                                        h_ground, 
                                        altitude_takeoff, distance_takeoff, 
                                        altitude_landing, distance_landing, MLW_frac, 
                                        altitude_cruise, Mach_cruise)

        # Compute change with respect to previous iteration
        delta = T0 - T0_guess

        # Update guesses for the next iteration
        T0_guess = T0
        W0_guess = W0

    # Return converged values
    return W0, We, Wf, xcg_e, T0, T0vec, S_wlan

#------------------------------------------------------------------------------

def balance(aircraft, Mach_cruise):

    geo_param = aircraft['geo_param']
    dimensions = aircraft['dimensions']
    W0 = aircraft['weights']['W0']
    W_payload = aircraft['weights']['W_payload']
    W_payload_upper_deck = aircraft['weights']['W_payload_upper_deck']
    xcg_payload_upper_deck = aircraft['weights']['xcg_payload_upper_deck']
    W_crew = aircraft['weights']['W_crew']
    xcg_crew = aircraft['weights']['xcg_crew']
    We = aircraft['weights']['We']
    xcg_e = aircraft['weights']['xcg_e']
    Wf = aircraft['weights']['Wf']
    S_w = geo_param['wing']['S']
    AR_w = geo_param['wing']['AR']
    sweep_w = geo_param['wing']['sweep']
    b_w = dimensions['wing']['b']
    cr_w = dimensions['wing']['cr']
    ct_w = dimensions['wing']['ct']
    xm_w = dimensions['wing']['xm']
    cm_w = dimensions['wing']['cm']
    xr_w = geo_param['wing']['xr']
    tcr_w = geo_param['wing']['tcr']
    tct_w = geo_param['wing']['tct']
    c_tank_c_w = geo_param['wing']['c_tank_c_w']
    x_tank_c_w = geo_param['wing']['x_tank_c_w']
    S_h = dimensions['EH']['S']
    AR_h = geo_param['EH']['AR']
    sweep_h = geo_param['EH']['sweep']
    b_h = dimensions['EH']['b']
    cr_h = dimensions['EH']['cr']
    ct_h = dimensions['EH']['ct']
    xm_h = dimensions['EH']['xm']
    cm_h = dimensions['EH']['cm']
    eta_h = geo_param['EH']['eta']
    L_f = dimensions['fus']['Lf']
    D_f = dimensions['fus']['Df']
    rho_f = aircraft['data']['misc']['rho_f']
    Cargo_CS = aircraft['data']['misc']['Cargo_CS_area']
    Vt = aircraft['data']['misc']['tank_max_vol']
    x_begin_cargo = aircraft['data']['misc']['x_begin_cargo']
    x_end_cargo = aircraft['data']['misc']['x_end_cargo']
    ### TANK CG
    '''
    We will compute the centroid of a trapezoidal tank.
    The expressions below where derived with the obelisk volume
    and centroid, assuming that both root and tip sections
    have the same c_tank_c_w and t/c
    '''
    
    # Check if it is double tappered (with kink corner)
    count = aircraft["geo_param"]['wing']['double_tappered']

    ### WING
    if count <= 0:
        cr_w_fuel = cr_w
        cr_w_eq = cr_w
    else:
        cr_w_fuel = aircraft['data']['misc']['cr_w_fuel']
        cr_w_eq = geo_param['wing']['cr_eq']
        S_w = aircraft['geo_param']['wing']['S_eq']
    
    # Required fuel volume
    Vf = Wf/rho_f/gravity
    
    # Average wing thickness
    tc_w = 0.5*(tcr_w + tct_w)
    
    y_tk_kink = aircraft['data']['misc']['y_tk_kink']
    x_tk_kink = aircraft['data']['misc']['x_tk_kink']
    
    tr_aw = tcr_w * cr_w_fuel
    tt_aw = (tcr_w - 2*y_tk_kink/b_w * (tcr_w - tct_w)) * (cr_w_fuel - 2*y_tk_kink/b_w * (cr_w_fuel - ct_w))
    t_aw = (tr_aw + tt_aw)/2
    cr_aw = x_tk_kink - cr_w_fuel*c_tank_c_w
    Vf_aw_t = cr_aw*y_tk_kink * t_aw
    xcg_f_aw = xr_w + cr_w_fuel*(c_tank_c_w + x_tank_c_w) + 2*y_tk_kink*np.arctan(y_tk_kink/(x_tk_kink - cr_w_fuel*c_tank_c_w))/3
    Vadd = Vf - Vf_aw_t
    if Vadd <= 0:
        Vf_w = 0
        Vf_aw = Vf
        Vf_fus = 0
        xcg_f_fus = 0
        xcg_f_w = 0
        l_extra = 0
        ycg_w = 0
    else:
        Vf_aw = Vf_aw_t
        b_tank_b_w = 3.0*Vadd/c_tank_c_w/tc_w/(cr_w_fuel**2 + ct_w**2 + cr_w_fuel*ct_w)/b_w
        extra = b_tank_b_w - 0.9
        if extra <= 0:
            ycg_w = b_tank_b_w*b_w/8*(cr_w_fuel**2 + 2*cr_w_fuel*ct_w + 3*ct_w**2)/(cr_w_fuel**2 + cr_w_fuel*ct_w + ct_w**2)
            Vf_w = Vf - Vf_aw_t
            Vf_w_t = 0.9*c_tank_c_w*tc_w*(cr_w_fuel**2 + ct_w**2 + cr_w_fuel*ct_w)*b_w/3.0
        else:
            ycg_w = 0.9*b_w/8*(cr_w_fuel**2 + 2*cr_w_fuel*ct_w + 3*ct_w**2)/(cr_w_fuel**2 + cr_w_fuel*ct_w + ct_w**2)
            Vadd2 = extra*c_tank_c_w*tc_w*(cr_w_fuel**2 + ct_w**2 + cr_w_fuel*ct_w)*b_w/3.0
            Vf_w = Vf - Vf_aw_t - Vadd2
            b_tank_b_w = 0.9
            Vf_w_t = b_tank_b_w*c_tank_c_w*tc_w*(cr_w_fuel**2 + ct_w**2 + cr_w_fuel*ct_w)*b_w/3.0
    if Vf_aw_t + Vf_w_t >= Vt:
        Vf_fus = 0
        l_extra = 0
        xcg_f_fus = 0
    else:
        l_extra = (Vt-(Vf_aw_t + Vf_w_t))/(Cargo_CS)
        Vf_fus = Vf-(Vf_aw + Vf_w)
        xcg_f_fus = xr_w - l_extra/2

    # Sweep at the tank center line
    sweep_tank_w = geo_change_sweep(0.25, x_tank_c_w + 0.5*c_tank_c_w,
                                  sweep_w, b_w/2, cr_w_fuel, ct_w)
    
    # Longitudinal position of the tank CG
    xcg_f_w = xr_w + cr_w_fuel*(x_tank_c_w + 0.5*c_tank_c_w) + ycg_w*np.tan(sweep_tank_w)
    xcg_f = rho_f*(xcg_f_w*Vf_w + xcg_f_aw*Vf_aw + xcg_f_fus*Vf_fus)/(Wf/gravity)

    # Longitudinal position of the payload
    l_fwd_cargo = (xr_w + x_tank_c_w*cr_w_fuel - l_extra) - x_begin_cargo
    l_aft_cargo = x_end_cargo - (xr_w + cr_w)
    W_payload_cargo = W_payload - W_payload_upper_deck
    xcg_payload_cargo = (l_fwd_cargo * (x_begin_cargo + l_fwd_cargo/2) + l_aft_cargo * (x_end_cargo - l_aft_cargo/2))/(l_fwd_cargo + l_aft_cargo)
    xcg_payload = (W_payload_upper_deck * xcg_payload_upper_deck + W_payload_cargo * xcg_payload_cargo) / (W_payload)
    
    ### CG RANGE
    
    # Empty airplane
    xcg_1 = xcg_e

    # Crew
    xcg_2 = (We*xcg_e + W_crew*xcg_crew)/(We + W_crew)

    # Payload and crew
    xcg_3 = (We*xcg_e + W_payload*xcg_payload + W_crew*xcg_crew)/(We + W_payload + W_crew)

    # Fuel and crew
    xcg_4 = (We*xcg_e + Wf*xcg_f + W_crew*xcg_crew)/(We + Wf + W_crew)

    # Payload, crew, and fuel (full airplane)
    xcg_5 = (We*xcg_e + Wf*xcg_f + W_payload*xcg_payload + W_crew*xcg_crew)/W0

    # Find CG range
    xcg_list = [xcg_1, xcg_2, xcg_3, xcg_4, xcg_5]
    xcg_fwd = min(xcg_list)
    xcg_aft = max(xcg_list)
    print(xcg_5)
    # We do not need to consider the static margin for the empty case
    # So we compute the flight CG range
    xcg_list = [xcg_2, xcg_3, xcg_4, xcg_5]
    xcg_fwd_flight = min(xcg_list)
    xcg_aft_flight = max(xcg_list)

    ### NEUTRAL POINT

    # Wing lift slope (Raymer Eq 12.6)
    sweep_maxt_w = geo_change_sweep(0.25, 0.40,
                                    sweep_w, b_w/2, cr_w_eq, ct_w) # Sweep at max. thickness
    beta2 = 1-Mach_cruise**2
    CLa_w = 2*np.pi*AR_w/(2 + np.sqrt(4 + AR_w**2*beta2/0.95**2*(1+np.tan(sweep_maxt_w)**2/beta2)))*0.98

    # Wing aerodynamic center at 25% mac
    xac_w = xm_w + 0.25*cm_w

    # HT lift slope (Raymer Eq 12.6)
    sweep_maxt_h = geo_change_sweep(0.25, 0.40,
                                    sweep_h, b_h/2, cr_h, ct_h) # Sweep at max. thickness
    CLa_h = 2*np.pi*AR_h/(2 + np.sqrt(4 + AR_h**2*beta2/0.95**2*(1+np.tan(sweep_maxt_h)**2/beta2)))*0.98

    # HT aerodynamic center at 25% mac
    xac_h = xm_h + 0.25*cm_h

    # Downwash (Nelson Eq 2.23)
    deda = 2*CLa_w/np.pi/AR_w

    # Fuselage moment slope (Raymer Eq 16.25)
    CMa_f = 0.03*180/np.pi*D_f**2*L_f/cm_w/S_w

    # Neutral point position (Raymer Eq 16.9 and Eq 16.23)
    xnp = (CLa_w*xac_w - CMa_f*cm_w + eta_h*S_h/S_w*CLa_h*(1-deda)*xac_h)/(CLa_w + eta_h*S_h/S_w*CLa_h*(1-deda))

    # Static margin
    SM_fwd = (xnp - xcg_fwd_flight)/cm_w
    SM_aft = (xnp - xcg_aft_flight)/cm_w

    return xcg_fwd, xcg_aft, xnp, SM_fwd, SM_aft, b_tank_b_w

#------------------------------------------------------------------------------

def landing_gear(aircraft):
    
    dimensions = aircraft['dimensions']
    geo_param = aircraft['geo_param']
    x_nlg = dimensions['ldg']['xnlg']
    dx_mlg = dimensions['ldg']['dx_mlg_wing']
    xr_w = geo_param['wing']['xr']
    y_mlg = dimensions['ldg']['ymlg']
    z_lg = dimensions['ldg']['z']
    xcg_fwd = aircraft['weights']['xcg_fwd']
    xcg_aft = aircraft['weights']['xcg_aft']
    x_tailstrike = aircraft['data']['misc']['x_tailstrike']
    z_tailstrike = aircraft['data']['misc']['z_tailstrike']
    
    # Weight fractions on NLG for both load cases
    x_mlg = dx_mlg + xr_w
    frac_nlg_fwd = (x_mlg-xcg_fwd)/(x_mlg-x_nlg)
    frac_nlg_aft = (x_mlg-xcg_aft)/(x_mlg-x_nlg)

    # Tipback angle (for now assume that CG is along fuselage axis)
    alpha_tipback = np.arctan((x_mlg - xcg_aft)/(-z_lg))

    # Tailstrike angle
    alpha_tailstrike = np.arctan((z_tailstrike - z_lg)/(x_tailstrike - x_mlg))

    # Overturn angle
    sgl = (xcg_fwd - x_nlg)*y_mlg/np.sqrt((x_mlg - x_nlg)**2 + y_mlg**2)
    phi_overturn = np.arctan(-z_lg/sgl)

    return frac_nlg_fwd, frac_nlg_aft, alpha_tipback, alpha_tailstrike, phi_overturn

#------------------------------------------------------------------------------
    
def analyze(aircraft, W0_guess, T0_guess,
            Mach_cruise, altitude_cruise, range_cruise, 
            Mach_altcruise, range_altcruise, altitude_altcruise,
            loiter_time,
            altitude_takeoff, distance_takeoff, TO_flap_def, TO_slat_def, 
            altitude_landing, distance_landing, LD_flap_def, LD_slat_def,
            MLW_frac):

    '''
    This is the main function that should be used for aircraft analysis.
    The standard parameters refer to the Fokker 100, but they could be redefined for
    any new aircraft.

    OUTPUTS:
    W0: real -> Aircraft MTOW
    T0: real -> Total takeoff thrust (accounting all engines)
    deltaS_wlan: real -> Wing area excess with respect to landing requirement.
                         The aircraft should have deltaS_wlan >= 0 to satisfy landing.
    SM: real -> Static margin
    frac_nlg: real -> Weight fraction applied at the nose landing gear
    '''

    # Generate geometry
    dimensions = aircraft['dimensions']
    new_dimensions = geometry(aircraft)

    aircraft['dimensions'] = {**dimensions, **new_dimensions} 
    # Converge MTOW and Takeoff Thrust
    h_ground = 35.0*ft2m
    W0, We, Wf, xcg_e, T0, T0vec, S_wlan = thrust_matching(aircraft, W0_guess, T0_guess,
                                                           TO_flap_def, LD_flap_def,
                                                           TO_slat_def, LD_slat_def,
                                                           h_ground,
                                                           altitude_cruise, Mach_cruise, range_cruise,
                                                           loiter_time,
                                                           altitude_altcruise, Mach_altcruise, range_altcruise,
                                                           altitude_takeoff, distance_takeoff,
                                                           altitude_landing, distance_landing, MLW_frac)

    # Compute wing area excess with respect to the landing requirement.
    # The aircraft should have deltaS_wlan >= 0 to satisfy landing.
    deltaS_wlan = aircraft['geo_param']['wing']['S'] - S_wlan

    # Balance analysis
    xcg_fwd, xcg_aft, xnp, SM_fwd, SM_aft, b_tank_b_w = balance(aircraft, Mach_cruise)
    
    aircraft['weights']['xcg_fwd'] = xcg_fwd
    aircraft['weights']['xcg_aft'] = xcg_aft

    # Landing gear design
    frac_nlg_fwd, frac_nlg_aft, alpha_tipback, alpha_tailstrike, phi_overturn = landing_gear(aircraft)



    return W0, Wf, T0, deltaS_wlan, SM_fwd, SM_aft, b_tank_b_w, frac_nlg_fwd, frac_nlg_aft, alpha_tipback, alpha_tailstrike, phi_overturn