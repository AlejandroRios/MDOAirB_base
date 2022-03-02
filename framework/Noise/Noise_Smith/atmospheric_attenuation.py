"""
MDOAirB

Description:
    - This module is used to estimate the atmospheric attenuation

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
from scipy import interpolate
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def atmospheric_attenuation(T_source,noise_parameters,R,f):
    """
    Description:
        - This function is used to estimate the atmospheric attenuation
    Inputs:
        - T_source - ambient temperature [deg C]
        - noise_parameters - noise constant parameters
        - R - distance [m]
        - f - frequencies [Hz]
    Outputs:
        - ft - frequencies [Hz]
        - alfaamortt -
        - amorttott 
        - amortatmt
        - SPLrt - noise atennuation
    """
    HR = noise_parameters['relative_humidity']   
    T                   = T_source                                                   # temperatura para avaliação da atenuação [ºC]
    H                   = HR                                                   # umidade relativa do ar para avaliação da atenuação [#]
    radialdistance      = R                                                    # distância para avaliação do ruído [m]
    Terzbandfreq        = f                                                    # freqüências para cálculo da atenuação [Hz]
    
    ## Declaração de constantes ##
    deltaamorttab       = np.array([0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.3, 2.5, 2.8, 3, 3.3, 3.6, 4.15, 4.45, 4.8, 5.25, 5.7, 6.05, 6.5, 7])
    etaamorttab         = np.array([0, 0.315, 0.7, 0.84, 0.93, 0.975, 0.996, 1, 0.97, 0.9, 0.84, 0.75, 0.67, 0.57, 0.495, 0.45, 0.4, 0.37, 0.33, 0.3, 0.26, 0.245, 0.23, 0.22, 0.21, 0.205, 0.2, 0.2])
    f0tab               = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 4500, 5600, 7100, 9000])
    
    ## Cálculo da atenuação atmosférica ##
    deltaamort          = np.sqrt(1010./f0tab)*10**(np.log10(H)-1.328924+3.179768e-2*(T-273)-2.173716e-4*(T-273)**2+1.7496e-6*(T-273)**3)
    etaamort            = interpolate.interp1d(deltaamorttab,etaamorttab,fill_value='extrapolate')(deltaamort)

    # interp1d(x,y)(new_x)
    alfaamort           = 10.**(2.05*np.log10(f0tab/1000)+1.1394e-3*(T-273)-1.916984)+etaamort*10**(np.log10(f0tab)+8.42994e-3*(T-273)-2.755624)
    deltaLamort         = alfaamort*radialdistance/100
    amortatm            = interpolate.interp1d(Terzbandfreq,deltaLamort,fill_value='extrapolate')(f)
    
    ## Cálculo da atenuação devido à distância ##
    SPLr                = 20*np.log10(radialdistance)                             # dB
    
    ## Cálculo da atenuação total ##
    amorttot            = amortatm+SPLr
    # amorttot            = 10*np.log10(10.**(0.1*amortatm)+10.**(0.1*SPLr))
    
    
    ## DADOS DE SAIDA ##
    ft                  = f
    alfaamortt          = alfaamort
    amorttott           = amorttot
    amortatmt           = amortatm
    SPLrt               = SPLr

    
    return ft, alfaamortt, amorttott, amortatmt, SPLrt
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
