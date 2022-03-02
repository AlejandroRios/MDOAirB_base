"""
Function  :
Title     :
Written by: 
Email     : aarc.88@gmail.com
Date      : 
Last edit :
Language  : Python 3.8 or >
Aeronautical Institute of Technology - Airbus Brazil

Description:
    -
Inputs:
    -
Outputs:
    -
TODO's:
    -

"""
# =============================================================================
# IMPORTS
# =============================================================================
from framework.Stability.Dynamic.Cmat import Cmat
import numpy as np
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================


def prop_loads(state, control):

    # state =state.squeeze().T
    # control = control.squeeze().T
    Tle = control[0]
    Tre = control[1]

    ## ---------------------------Parâmetros do Motor-------------------------#
    ile = 2
    ire = 2
    tau_le = 1.5
    tau_re = -1.5

    ## -------------------------Matriz de Transformação-----------------------#
    Clb_e = (Cmat(2, np.deg2rad(ile)).transpose()).dot(
        Cmat(3, np.deg2rad(tau_le)).transpose())
    Crb_e = (Cmat(2, np.deg2rad(ire)).transpose()).dot(
        Cmat(3, np.deg2rad(tau_re)).transpose())

    ## -------------------Ponto de aplicação da Força Propulsiva--------------#
    rle_b = np.array([4.899, -5.064, 1.435])
    rre_b = np.array([4.899, 5.064, 1.435])

    ## ---------------------------Decomposição das Forças---------------------#
    Txle = Tle*np.cos(np.deg2rad(ile))
    Tyle = Tle*np.sin(np.deg2rad(tau_le))
    Tzle = Tle*np.sin(np.deg2rad(ile))
    Txre = Tre*np.cos(np.deg2rad(ire))
    Tyre = Tre*np.sin(np.deg2rad(tau_re))
    Tzre = Tre*np.sin(np.deg2rad(ire))
    Tl = np.array([Txle, Tyle, Tzle])
    Tr = np.array([Txre, Tyre, Tzre])
    ## -----------------------------Forças Propulsivas------------------------#
    F_esq = Clb_e.dot(Tl)

    # print(F_esq)

    F_dir = Crb_e.dot(Tr)
    Fprop_b = F_esq+F_dir

    # print(Fprop_b)

    ## ----------------------------Momentos Propulsivos-----------------------#
    M_esq = F_esq*rle_b
    M_dir = F_dir*rre_b
    Mprop_O_b = M_dir-M_esq

    ## ---------------------------------Saidas--------------------------------#
    Yprop = np.array([F_esq.T, F_dir.T, M_esq.T, M_dir.T, Tl.T, Tr.T])
    return Fprop_b, Mprop_O_b, Yprop
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
