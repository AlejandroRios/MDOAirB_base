import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
kt_to_ms = 1.944

def approach(S_LFL=None, V_APP=None):

    k_APP = 1.7 # (m/s^2)

    if S_LFL != None:
        V_APP = k_APP*np.sqrt(S_LFL)
        V_APP_kt = V_APP/kt_to_ms
    
    else:
        S_LFL =  V_APP/k_APP
        
    return V_APP, S_LFL

def landing(S_LFL,delta_ISA,CL_max_L,mML_mTO):

    k_APP = 1.7 # (m/s^2)

    gamma = 288.15/(288.15+delta_ISA)
    k_L = 0.03694455*k_APP**2

    # Wing loading at max landing mass
    mML_Sw = k_L*gamma*CL_max_L*S_LFL
    # Wing loading at max takeoff mass
    mMTOW_Sw = mML_Sw/mML_mTO

    return mMTOW_Sw

def takeoff(S_TOFL,delta_ISA,CL_max_L, mMTOW_Sw):
    

    gamma = 288.15/(288.15+delta_ISA)

    k_TO = 2.34

    CL_max_TO = 0.8*CL_max_L

    slope = k_TO/(S_TOFL*gamma*CL_max_TO)

    T_to_W_TO = slope*mMTOW_Sw

    return T_to_W_TO,slope


def second_segment(AR,CD0,CL_max_L,n_eng,oswald_eff):

    CL_max_TO = 0.8*CL_max_L
    CL_TO = CL_max_TO/(1.2**2)

    delta_CD_flap  = 0.05*(CL_TO-1.3)+0.01

    if delta_CD_flap < 0:
        delta_CD_flap = 0
    else:
        delta_CD_flap

    delta_CD_slats = 0.000

    CD_p = CD0+delta_CD_flap+delta_CD_slats

    E_TO = CL_TO/(CD_p+CL_TO**2/np.pi/AR/oswald_eff)

    if n_eng == 2:
        climb_gradient = 0.024
    elif n_eng == 3:
        climb_gradient = 0.027
    elif n_eng == 4:
        climb_gradient = 0.030

    T_to_W_secseg = n_eng/(n_eng-1)*(1/E_TO+climb_gradient)    

    return T_to_W_secseg

def missed_approach(CL_max_L,CD0,AR,oswald_eff,n_eng,mML_mTO,Certification=None):
    CL_L = CL_max_L/(1.3**2)

    delta_CD_flap  = 0.05*(CL_L-1.3)+0.01

    if delta_CD_flap < 0:
        delta_CD_flap = 0
    else:
        delta_CD_flap
    
    delta_CD_slats = 0.000

    if Certification == 'CS25':
        delta_CD_gear = 0.000
    else:
        delta_CD_gear = 0.015

    CD_p = CD0+delta_CD_flap+delta_CD_slats+delta_CD_gear

    E_L = CL_L/(CD_p+CL_L**2/np.pi/AR/oswald_eff)

    if n_eng == 2:
        climb_gradient = 0.021
    elif n_eng == 3:
        climb_gradient = 0.024
    elif n_eng == 4:
        climb_gradient = 0.027

    T_to_W_missapp = n_eng/(n_eng-1)*(1/E_L+climb_gradient)*mML_mTO

    return T_to_W_missapp 

def max_glide_ratio_cruise(oswald_eff,Cf_eqv,Swet_Sw,AR):

    k_E = 1/2*np.sqrt(np.pi*oswald_eff/Cf_eqv)

    E_max = k_E*np.sqrt(AR/Swet_Sw)

    return E_max

    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



def matching_chart(BPR,oswald_eff,M_CR,V_Vmd,E_max,AR,T_to_W_secseg,T_to_W_missapp,slope,mMTOW_Sw, T_to_W,W_to_S):

    CD0 = np.pi*AR*oswald_eff/4/E_max**2
    CLmd = np.sqrt(CD0*np.pi*AR*oswald_eff)
    CL_CLmd = 1/V_Vmd**2
    CL = CL_CLmd*CLmd

    E = E_max*2/(1/CL_CLmd+CL_CLmd)

    gamma = 1.4
    g = 9.81
    p0 = 101325
    euler = 2.718282

    altitude = np.linspace(0,15,501)
    altitude_ft = altitude*1000*3.281 

    T_CR_T_TO = [(0.0013*BPR-0.0397)*x-0.0248*BPR+0.7125 for x in altitude]
    T_TO_m_MTOg = [1/(x*E) for x in T_CR_T_TO]
    

    p_h = [p0*np.power(1-0.02256*x,5.256) if x < 12 else p0*0.2232*np.power(euler,-0.1577*(x-11)) for x in altitude]


    mMTO_Sw = [CL*M_CR*M_CR/g*gamma/2*x for x in p_h]

    T_to_W_secseg = [T_to_W_secseg]*len(altitude)

    T_to_W_missapp  = [T_to_W_missapp]*len(altitude)

    T_to_W_TO =[slope*x for x in mMTO_Sw]

    T_to_W_L = [0,0.5]

    mMTOW_Sw = [mMTOW_Sw]*len(T_to_W_L)

    df = pd.DataFrame(list(zip(altitude, mMTO_Sw, T_to_W_secseg, T_to_W_missapp, T_to_W_TO, T_TO_m_MTOg)), columns =['altitude', 'MTO_Sw','T_to_W_secseg','T_to_W_missapp','T_to_W_TO','T_TO_m_MTOg']) 


    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend',fontsize=12) # using a size in points
    plt.rc('legend',fontsize='medium') # using a named size
    plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(mMTO_Sw,T_to_W_secseg,label='2nd segment')
    ax.plot(mMTO_Sw,T_to_W_missapp,label='Missed appr.')
    ax.plot(mMTO_Sw,T_to_W_TO,label='Take-off')
    ax.plot(mMTO_Sw,T_TO_m_MTOg,label='Cruise')
    ax.plot(mMTOW_Sw,T_to_W_L,label='Landing')
    ax.plot(W_to_S,T_to_W,'o',label='Design pont')

    ax.set_xlabel('Wing loading [kg/m2]')
    ax.set_ylabel('Thrust-to-weight ration')
    ax.set_title('Matching chart')

    # ax.set_xlim([0,None])
    # ax.set_ylim([0,None])
    ax.legend()

    plt.grid(True)
    plt.show()



    return df


def all_checks(S_LFL,S_TOFL,CL_max_L,mML_mTO,vehicle,CD0,T_to_W,W_to_S):

    delta_ISA = 0

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    engine = vehicle['engine']
    operations = vehicle['operations']

    AR = wing['aspect_ratio']

    n_eng = aircraft['number_of_engines']
    oswald_eff = 0.7
    Certification = 'FAR25'
    Cf_eqv = 0.003
    Swet_Sw = aircraft['wetted_area']/wing['area']

    BPR =  engine['bypass']
    M_CR = operations['mach_cruise']

    



    V_APP, S_LFL = approach(S_LFL)
    mMTOW_Sw = landing(S_LFL,delta_ISA,CL_max_L,mML_mTO)

    T_to_W_L,slope = takeoff(S_TOFL,delta_ISA,CL_max_L, mMTOW_Sw)

    T_to_W_secseg = second_segment(AR,CD0,CL_max_L,n_eng,oswald_eff)

    T_to_W_missapp = missed_approach(CL_max_L,CD0,AR,oswald_eff,n_eng,mML_mTO,Certification=None)

    oswald_eff = 0.85
    
    E_max = max_glide_ratio_cruise(oswald_eff,Cf_eqv,Swet_Sw,AR)

    V_Vmd = np.sqrt(np.sqrt(3))


    df = matching_chart(BPR,oswald_eff,M_CR,V_Vmd,E_max,AR,T_to_W_secseg,T_to_W_missapp,slope,mMTOW_Sw,T_to_W,W_to_S)

    point_WtoS = df.iloc[(df['MTO_Sw']-W_to_S).abs().argsort()[:1]]
    print(point_WtoS) 

    T_to_W_cruise = float(point_WtoS['T_TO_m_MTOg'])

    # Wing loading at max. take-off mass
    if  W_to_S > mMTOW_Sw:
        flag_landing = 1
    else:
        flag_landing = 0

    # Take-off T/W
    if  T_to_W < T_to_W_L:
        flag_takeoff = 1
    else:
        flag_takeoff = 0

    # Second segment T/W
    if  T_to_W < T_to_W_secseg:
        flag_climb_second_segment = 1
    else:
        flag_climb_second_segment = 0

    # Missed approach T/W
    if  T_to_W < T_to_W_missapp:
        flag_missed_approach = 1
    else:
        flag_missed_approach = 0

    # Cruise T/W
    if  T_to_W <  T_to_W_cruise:
        flag_cruise = 1
    else:
        flag_cruise = 0

    flags = [flag_landing, flag_takeoff, flag_climb_second_segment, flag_missed_approach, flag_cruise]
    print('flags:',flags )

    print('Landing field:', S_LFL)

    print('Landing W/S:', mMTOW_Sw)

    print('Takeoff T/W:', T_to_W_L)

    print('Second segment T/W:', T_to_W_secseg)

    print('Missed approach T/W:', T_to_W_missapp)

    print('Max. glide ratio:',  E_max)

    return






