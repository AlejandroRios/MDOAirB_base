"""
MDOAirB

Description:
    - This module is used to convert SPL to NOY's

Reference:
    - 

TODO's:
    -

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
from numba import jit
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def calculate_NOY(f, SPL):
    """
    Description:
        - This function is used to convert SPL to NOY's
    Inputs:
        - f - standard frequencies [Hz]
        - SPL - noise level in relation to reference [dB]
    Outputs:
        - f - standard frequencies [Hz]
        - NOY - noise [NOY]
    """

    ## CORPO DA FUNÇÃO ##
    ## Tabelas para cálculo ##
    fref                = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
    SPLa                = np.array([91.00, 85.90, 87.30, 79.90, 79.80, 76.00, 74.00, 74.90, 94.60, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 44.30, 50.70])
    SPLb                = np.array([64.00, 60.00, 56.00, 53.00, 51.00, 48.00, 46.00, 44.00, 42.00, 40.00, 40.00, 40.00, 40.00, 40.00, 38.00, 34.00, 32.00, 30.00, 29.00, 29.00, 30.00, 31.00, 37.00, 41.00])
    SPLc                = np.array([52.00, 51.00, 49.00, 47.00, 46.00, 45.00, 43.00, 42.00, 41.00, 40.00, 40.00, 40.00, 40.00, 40.00, 38.00, 34.00, 32.00, 30.00, 29.00, 29.00, 30.00, 31.00, 34.00, 37.00])
    SPLd                = np.array([49.00, 44.00, 39.00, 34.00, 30.00, 27.00, 24.00, 21.00, 18.00, 16.00, 16.00, 16.00, 16.00, 16.00, 15.00, 12.00, 09.00, 05.00, 04.00, 05.00, 06.00, 10.00, 17.00, 21.00])
    SPLe                = np.array([55.00, 51.00, 46.00, 42.00, 39.00, 36.00, 33.00, 30.00, 27.00, 25.00, 25.00, 25.00, 25.00, 25.00, 23.00, 21.00, 18.00, 15.00, 14.00, 14.00, 15.00, 17.00, 23.00, 29.00])
    Mb                  = np.array([0.043478, 0.040570, 0.036831, 0.036831, 0.035336, 0.033333, 0.033333, 0.032051, 0.030675, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.029960, 0.029960, 0.029960, 0.029960, 0.029960, 0.029960, 0.029960, 0.042285, 0.042285])
    Mc                  = np.array([0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.029960, 0.029960])
    Md                  = np.array([0.079520, 0.068160, 0.068160, 0.059640, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.053103, 0.059640, 0.053103, 0.053103, 0.047712, 0.047712, 0.053103, 0.053103, 0.068160, 0.079520, 0.059640])
    Me                  = np.array([0.058098, 0.058098, 0.052288, 0.047534, 0.043573, 0.043573, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.037349, 0.037349, 0.043573])
    ## Comparação das frequencias ##

    ## Definições dos loops ##
    a1                  = max(np.shape(f))
    a3, a2            = SPL.shape
    ## Eliminação dos pontos nã

    ## Conversão de SPL em NOY ##
    n = []
    for i2 in range(a3):
        n_aux = []
        for i1 in range(0,24):

            if SPL[i2][i1]>=SPLa[i1]:
                aux1 = 10**(Mc[i1]*(SPL[i2][i1]-SPLc[i1]))

            elif (SPL[i2][i1] < SPLa[i1] and SPL[i2][i1]>=SPLb[i1]):
                aux1 = 10**(Mb[i1]*(SPL[i2][i1]-SPLb[i1]))

            elif (SPL[i2][i1]<SPLb[i1] and SPL[i2][i1]>=SPLe[i1]):
                aux1 = 0.3*10**(Me[i1]*(SPL[i2][i1]-SPLe[i1]))

            elif (SPL[i2][i1]<SPLe[i1] and SPL[i2][i1]>=SPLd[i1]):
                aux1 = 0.1*10**(Md[i1]*(SPL[i2][i1]-SPLd[i1]))
            
            else:
                aux1 = 0

            

            n_aux.append(np.asarray(aux1))
        n.append(n_aux)

    ## DADOS DE SAIDA ##
    NOY                 = np.asarray(n)

    return f, NOY

@jit(nopython=True)
def calculate_PNL(f,NOY):
    """
    Description:
        - This function is used to calculate the Perceived Noise Levels
    Inputs:
        - f - standard frequencies [Hz]
        - NOY - noise [NOY]
    Outputs:
        -PNL - perceived noise level [PNdB]
    """
    ## CORPO DA FUNÇÃO ##
    ## Definições dos loops ##
    a1       = max(np.shape(f))
    a3,a2             = NOY.shape

    ## Conversão de NOY em PNdB##
    nmax = np.zeros(a3)
    N = np.zeros(a3)
    PNL = np.zeros(a3)
    for i2 in range(a3):
        for i1 in range(a1):

            nmax[i2] = max(NOY[:][i2])
            N[i2] =(nmax[i2]+0.15*(np.sum(NOY[i2])-nmax[i2]))
            PNL[i2] =(40+10/np.log10(2)*np.log10(N[i2]))
    
    # PNL = np.asarray(PNL)

    return PNL

@jit(nopython=True)
def calculate_PNLT(f,SPL):
    """
    Description:
        - This function is used to calculate the perceived noise level, tone corrected
    Inputs:
        - f - standard frequencies [Hz]
        - SPL - noise level in relation to reference [dB]
    Outputs:
        - Cfin - noise correction [dB]
    """
    #step 1
    s = np.zeros(24)
    # s = np.array([])
    for i1 in range(3,23):

        s[i1] =SPL[i1]-SPL[i1-1]
    
    #step 2
    dels = np.zeros(24)

    for i1 in range(5,23):
        dels[i1] =np.abs(s[i1]-s[i1-1])

    sc = np.zeros(24)
    for i1 in range(len(sc)):
        if dels[i1]>5:
            sc[i1]      = 1
        
    
    #step 3
    SPLC = np.zeros(24)
    for i1 in range(len(SPLC)):
        if sc[i1]==1:
            if s[i1]>0 and s[i1]>s[i1-1]:
                SPLC[i1] = 1
            
            if s[i1]<=0 and s[i1-1]>0:
                SPLC[i1-1] = 1
            
    
    #step 4
    SPLl = np.zeros(24)
    for i1 in range(len(SPLl)):
        if SPLC[i1] != 1:
            SPLl[i1]    = SPL[i1]
        else:
            if i1<24:
                SPLl[i1] = 0.5*(SPL[i1-1]+SPL[i1+1])
            else:
                SPLl[i1] = SPL[i1]+s[i1]
            
        
    
    #step 5
    sl = np.zeros(25)
    for i1 in range(3,23):
        sl[i1]          = SPLl[i1]-SPLl[i1-1]
    
    sl[2]               = sl[3]
    sl[24]              = sl[23]
    #step 6

    sbar = np.zeros(23)
    for i1 in range(2,22):
        sbar[i1]        = 1/3*(sl[i1]+sl[i1+1]+sl[i1+2])
    
    #step 7
    SPLll = np.zeros(24)
    SPLll[2]            = SPL[2]
    for i1 in range(3,24):
        SPLll[i1]       = SPLll[i1-1]+sbar[i1-1]
    
    #step 8
    F = np.zeros(24)

    for i1 in range(3,23):
        if (SPL[i1]-SPLll[i1])>0:
            F[i1]       = SPL[i1]-SPLll[i1]
        
    

    ## Definição do valor da correção ##
    C = np.zeros(24)
    for i1 in range(len(C)):
        if F[i1]>=1.5:
            if (f[i1]>=50 and f[i1]<500):
                if (F[i1] >= 1.5 and F[i1] < 3.0):
                    C[i1] = F[i1]/3-0.5
                
                if (F[i1] >= 3.0 and F[i1] < 20.0):
                    C[i1] = F[i1]/6
                
                if F[i1] >= 20.0:
                    C[i1] = 3+1/3
                
            
            if (f[i1]>=500 and f[i1]<5000):
                if (F[i1] >= 1.5 and F[i1] < 3.0):
                    C[i1] = 2*F[i1]/3-1.0
                
                if (F[i1] >= 3.0 and F[i1] < 20.0):
                    C[i1] = F[i1]/3
                
                if F[i1] >= 20.0:
                    C[i1] = 6+2/3
                
            
            if (f[i1]>=5000 and f[i1]<10000):
                if (F[i1] >= 1.5 and F[i1] < 3.0):
                    C[i1] = F[i1]/3-0.5
                
                if (F[i1] >= 3.0 and F[i1] < 20.0):
                    C[i1] = F[i1]/6
                
                if F[i1] >= 20.0:
                    C[i1] = 3+1/3

    ## DADOS DE SAIDA ##
    Cfin                = max(C)


    return Cfin
@jit(nopython=True)
def calculate_EPNdB(tempo,PNLT):
    """
    Description:
        - This function is used to calculate the effective prerceibed noise (EPNdB)
    Inputs:
        - tempo
        -PNL - perceived noise level [PNdB]
    Outputs:
        - EPNdB - effectively perceived noise [EPNdB]
    """

    ## CORPO DA FUNÇÃO ##
    PNLTM               = max(PNLT)                                            # Determinação do PNLT máximo
    PNLTL               = PNLTM-10                                             # Determinação do nível mínimo a ser considerado
    
    a1                  = max(np.shape(PNLT))                                       # Determinação das variáveis de controle do cálculo
    dt                  = 0.5                                  # Determinação das variáveis de controle do cálculo
    for i1 in range(0,a1-1):                                                               # Separação dos valores de PNLT que serão usados
        if PNLT[i1+1]>PNLTL and PNLT[i1]<PNLTL:
            ind1        = i1+1
        else:
            ind1        = 1

        if PNLT[i1+1]<PNLTL and PNLT[i1]>PNLTL:
            ind2        = i1+1
        else:
            ind2        = a1

    # Somatório do ruído
    termo1              = 0
    for i1 in range(ind1,ind2):
        termo1          = termo1+dt*10**(PNLT[i1]/10)

    D                   = 10*np.log10(0.1*termo1)-PNLTM-13                        # Determinação do fator de correção
    EPNdB               = PNLTM+D                                              # Determinação de EPNdB

    return EPNdB
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
