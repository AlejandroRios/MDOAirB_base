"""
MDOAirB

Description:
    - This module computes the wing structural layout and calculates
    the fuel storage capacity in the wings

Reference:
    - Prof. Bentos codfe

TODO's:
    - Clean code
    - Rename variables

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
import os

from framework.Attributes.Atmosphere.atmosphere_ISA_deviation import \
    atmosphere_ISA_deviation
from framework.utilities.logger import get_logger
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
log = get_logger(__file__.split('.')[0])


def wing_structural_layout(vehicle, xutip, yutip,
                           yltip, xukink, xlkink, yukink, ylkink, xuroot, xlroot, yuroot, ylroot):
    """
    Description:
        - This function generates wing structural layout and estimates fuel storage.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
        - xutip - upper surface x tip chord coordinate
        - yutip - upper surface y tip chord coordinate
        - xltip - lower surface x tip chord coordinate
        - yltip - lower surface y tip chord coordinate
        - xubreak - upper surface x break chord coordinate
        - yubreak - upper surface y break chord coordinate
        - xlbreak - lower surface x break chord coordinate
        - ylbreak - lower surface y break chord coordinate
        - xuroot - upper surface x root chord coordinate
        - yuroot - upper surface y root chord coordinate
        - xlroot - lower surface x root chord coordinate
        - ylroot - lower surface y root chord coordinate
    Outputs:
        - vehicle - dictionary containing aircraft parameters
    """

    # OBS: Wing leading edge must be of constant sweep
    # Input:
    # wing['trunnion_xposition'] = Posicao da longarina traseira (fracional)
    # AR = Wing aspect ratio
    # Chord at fuselage centerline (CRaiz)
    # Chord at wingtip (wing['tip_chord'])
    # Chord at kink station (wing['kink_chord'])
    # PSILE: Leading-edge sweepback angle
    # Aileron basis as semi-span fraction (posaileron)
    # Location of the kink station as semi-span fraction (wing.ybreak)
    
    kg_l_to_kg_m3 = 1000
    rad = np.pi/180
    # Initialization

    aircraft = vehicle['aircraft']
    wing = vehicle['wing']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    operations = vehicle['operations']
    nose_landing_gear = vehicle['nose_landing_gear']

    
    wing['fuel_capacity'] = 0
    nukink = len(yukink)
    #
    ribs_spacing = wing['ribs_spacing']   # (pol) Roskam Vol III pg 220 suggests 24
    nervspacm = ribs_spacing * 0.0254  # cm)
    querosene_density = operations['fuel_density']*kg_l_to_kg_m3  # jet A1 density

    angquebralongtras = 0
    #
    diamfus = fuselage['width'] 

    fquebra = wing['semi_span_kink']


    # Variaveis auxiliares
    bdiv2 = 0.50*wing['span']
    xquebraBA = bdiv2*fquebra*np.tan(rad*wing['sweep_leading_edge'])
    yquebra = fquebra*bdiv2

    if aircraft['slat_presence'] > 0:
        fraclongdi = 0.25
    else:
        fraclongdi = 0.15

    limited = fraclongdi

    # Dados do trem de pouso:
    pneu_diam = nose_landing_gear['tyre_diameter'] # diam do pneu em metros
    pneu_height = nose_landing_gear['tyre_height']  # largura do pneu (m)
    lmunhao = nose_landing_gear['trunnion_length']  # Comprimento do munhao (m)

    # Intersecao asa-fuselagem
    yfusjunc = diamfus/2

    # Vetor para Aramzernar Todas as Nervuras
    Nerv = np.zeros((0, 4))

    # Calcula a corda na intersecao
    xbainter = yfusjunc*np.tan(rad*wing['sweep_leading_edge'])  # coord do ba na intersecao

    xbfquebra = xquebraBA + wing['kink_chord']

    if xbfquebra == wing['center_chord']:
        xbfinter = xbfquebra
    else:
        inclinabf = (yquebra-0)/((xquebraBA + wing['kink_chord']) - wing['center_chord'])
        xbfinter = wing['center_chord']+(yfusjunc-0)/inclinabf  # coord do bf na intersecao

    Cinter = xbfinter - xbainter
    aux1 = bdiv2*np.tan(rad*wing['sweep_leading_edge'])
    aux2 = xquebraBA

    # *** Forma em planta da asa***
    xw = [0, xbainter, aux1, (aux1+wing['tip_chord']), (aux2+wing['kink_chord']), xbfinter, wing['center_chord']]
    yw = [0, yfusjunc, bdiv2, bdiv2, yquebra, yfusjunc, 0]



    xfus = [-2, wing['center_chord']+2]
    yfus = [diamfus/2, diamfus/2]

    # *** Ponta da asa ***
    xcontrolpoint2 = bdiv2*np.tan(rad*wing['sweep_leading_edge'])
    xcontrolpoint3 = bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['tip_chord']
    xcontrolpoint4 = aux2+wing['kink_chord']
    ycontrolpoint3 = bdiv2
    ycontrolpoint4 = fquebra*bdiv2
    inclinabf = (ycontrolpoint4-ycontrolpoint3)/(xcontrolpoint4-xcontrolpoint3)
    xprojbfponta = xcontrolpoint3+(1.05*bdiv2-ycontrolpoint3)/inclinabf

    # projecao do BF da ponta
    xpp = [xcontrolpoint2, (xcontrolpoint2+0.15*wing['tip_chord']),
           xprojbfponta, (xcontrolpoint2+wing['tip_chord'])]
    ypp = [bdiv2, (bdiv2+0.025*bdiv2), (bdiv2+0.05*bdiv2), bdiv2]


    # *** Longarina dianteira
    xld = [(yfusjunc*np.tan(rad*wing['sweep_leading_edge'])+fraclongdi*Cinter),
           (bdiv2*np.tan(rad*wing['sweep_leading_edge'])+fraclongdi*wing['tip_chord'])]
    yld = [yfusjunc, bdiv2]

    # *** Fim longarina dianteira ***

    # *** Longarina traseira (LT) ***

    # LT externa
    x1aux = (wing['trunnion_xposition']*wing['kink_chord']+bdiv2*fquebra*np.tan(rad*wing['sweep_leading_edge']))
    xlte = [x1aux, (bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['trunnion_xposition']*wing['tip_chord'])]
    ylte = [bdiv2*fquebra,  bdiv2]

    # LT Interna

    inclnt = x1aux-(bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['trunnion_xposition']*wing['tip_chord'])
    inclnt = (bdiv2*fquebra - bdiv2)/inclnt

    # adiciona angulo graus para aumentar o tamanho do caixao central
    anglti = np.arctan(inclnt)+angquebralongtras*np.pi/180
    xltintern = x1aux+((yfusjunc-bdiv2*fquebra)/np.tan(anglti))
    xlti = [xltintern, x1aux]
    ylti = [yfusjunc, bdiv2*fquebra]


    xlt = []
    xlt = xlti
    xlt[1] = xlte[1]

    ylt = []
    ylt = ylti
    ylt[1] = ylte[1]

    # *** Fim da longarina traseita ***

    # ************************   Nervuras ********************************
    # ++++  Na quebra
    #
    x1aux = yfusjunc*np.tan(rad*wing['sweep_leading_edge']) + fraclongdi*Cinter
    y1aux = yfusjunc
    x2aux = bdiv2*np.tan(rad*wing['sweep_leading_edge']) + fraclongdi*wing['tip_chord']
    y2aux = bdiv2

    xnq = []
    ynq = []

    if x1aux == x2aux:
        xnq.append(x1aux)
    else:
        inclinald = (y2aux-y1aux)/(x2aux-x1aux)
        x0 = x1aux
        y0 = y1aux
        xnq.append((yquebra-y0)/inclinald + x0)


    xnq.append(xquebraBA+wing['trunnion_xposition']*wing['kink_chord'])
    ynq.append(yquebra)
    ynq.append(ynq[0])

    Nerv = np.hstack((xnq, ynq))

    # Coordendas da nervura na quebra
    xtnervq = xnq[1]
    ytnervq = ynq[1]
    xdnervq = xnq[0]
    ydnervq = ynq[0]

    # Nervura da asa externa com origem na quebra(eh perpendicular ah longarina
    # traseira)
    x1aux = xquebraBA + wing['trunnion_xposition']*wing['kink_chord']
    y1aux = yquebra

    x2aux = bdiv2*np.tan(rad*wing['sweep_leading_edge']) + wing['trunnion_xposition']*wing['tip_chord']
    y2aux = bdiv2

    nervkinknormal = 0  # a principio esta nervura nao existe
    nnervext = 1  # Por enquanto, apenas a nervura padrao da quebra eh levada em conta
    angnev = 0 # REVIEEEEEEEEEEEEEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    anglte = 0
    if x2aux == x1aux:
        anglte = np.pi/2
        angnev = 0
    else:
        inclinalt = (y2aux-y1aux)/(x2aux-x1aux)
        if inclinalt > 0:
            anglte = np.arctan(inclinalt)
            angnev = np.pi/2 + anglte
            # testa se o angulo entrea nerv da quebra e esta nervura eh maior
            # que 5 graus
            if (np.pi-angnev) > 5*rad:
                nervkinknormal = 1  # nervura serah considerada
                nnervext = 2

    x02 = xnq[1]
    y02 = ynq[1]
    inclinanerv = np.tan(angnev)
    x01 = yfusjunc*np.tan(rad*wing['sweep_leading_edge']) + fraclongdi*Cinter
    y01 = yfusjunc
    #

    xnqa = []
    ynqa = []

    if nervkinknormal == 1:
        # acha intersecao com a longarina dianteira
        xnqa.append(xnq[1])  # coord x do ponto da longarina traseira na quebra
        ynqa.append(ynq[1])  # coord y do ponto da longarina traseira na quebra

        term1 = (y02-y01)-inclinanerv*x02+inclinald*x01
        xinerv = term1/(inclinald-inclinanerv)
        yinerv = y01+inclinald*(xinerv-x01)
        xnqa.append(xinerv)
        ynqa.append(yinerv)
        Nerv = np.vstack((Nerv, np.hstack((xnqa, ynqa))))

    else:
        yinerv = yquebra

    # Restante das nervuras do caixao central externo
    ytnerv = y02
    xtnerv = x02
    ydnerv = yinerv
    #
    jnervext = 0
    deltay = nervspacm
    if anglte == 0:
        deltax = 0
    else:
        deltax = deltay/np.tan(anglte)

    xnervext_aux1 = []
    ynervext_aux1 = []
    xnervext_aux2 = []
    ynervext_aux2 = []
    # parte 1 ateh o flape
    while (ydnerv+deltay) < bdiv2:
        # descobre coord da nova nervura na LT
        ytnerv = ytnerv+deltay
        xtnerv = xtnerv+deltax
        jnervext = jnervext+1
        xnervext_aux1.append(xtnerv)
        ynervext_aux1.append(ytnerv)
        xnq[1] = xtnerv
        ynq[1] = ytnerv
        # Calcula intersecao com a LD
        ydnerv = ydnerv+deltay
        xdnerv = x01+(ydnerv-y01)/inclinald
        xnervext_aux2.append(xdnerv)
        ynervext_aux2.append(ydnerv)
        xnq[0] = xdnerv
        ynq[0] = ydnerv
        nnervext = nnervext+1
        Nerv = np.vstack((Nerv, np.hstack((xnq, ynq))))

    # Ultima nervura (fracionaria)

    if (ytnerv+deltay) < bdiv2:
        jnervext = jnervext+1
        ytnerv = ytnerv+deltay
        xtnerv = xtnerv+deltax
        xnq[0] = xtnerv
        ynq[0] = ytnerv
        xdnerv = (ytnerv-y01 + x01*inclinald-inclinanerv*xtnerv) / \
            (inclinald-inclinanerv)
        ydnerv = y01 + inclinald*(xdnerv-x01)
        yinerv = ydnerv
        sinerv = xdnerv

        if ydnerv > bdiv2:
            yinerv = bdiv2
            y01p = bdiv2
            x01p = bdiv2*np.tan(rad*wing['sweep_leading_edge'])
            xinerv = (ytnerv-y01p + x01p*0-inclinanerv*xtnerv) / \
                (0-inclinanerv)  # inclinacao d apont eh zero

        xnervext_aux1.append(xtnerv)
        ynervext_aux1.append(ytnerv)
        xnervext_aux2.append(xinerv)
        ynervext_aux2.append(yinerv)

        xnq[1] = xdnerv
        ynq[1] = yinerv
        nnervext = nnervext+1

        Nerv = np.vstack((Nerv, np.hstack((xnq, ynq))))

    xnervext = np.vstack((xnervext_aux1, xnervext_aux2))
    ynervext = np.vstack((ynervext_aux1, ynervext_aux2))

    # Nervuras no caixao central interno
    xtnerv = xtnervq
    ytnerv = ytnervq
    xdnerv = xdnervq
    ydnerv = ydnervq

    nni = 0
    deltay = 0.60*deltay

    xnervint_aux1 = []
    ynervint_aux1 = []
    xnervint_aux2 = []
    ynervint_aux2 = []

    while (ytnerv-yfusjunc-deltay) > (nervspacm/2):
        ytnerv = ytnerv - deltay
        xtnerv = xtnervq + (ytnerv-ytnervq)/np.tan(anglti)
        ydnerv = ytnerv
        xdnerv = x01+(ydnerv-y01)/inclinald
        xnq[0] = xdnerv
        ynq[0] = ydnerv
        nni = nni + 1
        xnervint_aux1.append(xdnerv)
        ynervint_aux1.append(ydnerv)
        xnq[1] = xtnerv
        ynq[1] = ytnerv
        xnervint_aux2.append(xtnerv)
        ynervint_aux2.append(ytnerv)

        Nerv = np.vstack((Nerv, np.hstack((xnq, ynq))))

    nni = nni+1

    # nervura na juncao asa-fuselagem
    xnq[0] = xld[0]
    ynq[0] = yld[0]
    xnervint_aux1.append(xnq[0])
    ynervint_aux1.append(ynq[0])
    xnq[1] = xlti[0]
    ynq[1] = ylti[0]
    xnervint_aux2.append(xnq[1])
    ynervint_aux2.append(ynq[1])
    #
    xnervint = np.vstack((xnervint_aux1, xnervint_aux2))
    ynervint = np.vstack((ynervint_aux1, ynervint_aux2))

    Nerv = np.vstack((Nerv, np.hstack((xnq, ynq))))

    # *** Aileron ***
    # comprimento estimado do aileron: aprox. 25# da semi-envergadura

    xail = []
    yail = []
    xail.append(bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['trunnion_xposition']*wing['tip_chord'])
    yail.append(bdiv2)
    xail.append(bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['tip_chord'])
    yail.append(bdiv2)
    #
    yinfaileron = 0.75*bdiv2
    # Procura a nervura mais proxima de yinfaileron
    minnerv = 1e06

    for j in range(jnervext):
        if abs(yinfaileron-ynervext[0, j]) < minnerv:

            minnerv = abs(yinfaileron-ynervext[0, j])

            mem = j

    ytop_tanque_externo = 0.85*bdiv2
    minnerv = 1e06
    for j in range(jnervext):
        if abs(ytop_tanque_externo-ynervext[0, j]) < minnerv:
            minnerv = abs(ytop_tanque_externo-ynervext[0, j])
            memtqe = j

    # calcula intesecao com BF
    tg1 = inclinabf
    tg2 = 0
    x1 = xcontrolpoint3
    y1 = ycontrolpoint3
    x2 = xnervext[0, mem]
    y2 = ynervext[0, mem]
    #
    term1 = (y2-y1)-tg2*x2+tg1*x1

    xail.append(term1/(tg1-tg2))
    yail.append(y1+tg1*(xail[2]-x1))
    #
    xail.append(xnervext[1, mem])
    yail.append(ynervext[1, mem])

    xcombe = []
    ycombe = []
    # Tanque de combust�vel na asa externa
    if engine['position'] == 2:
        xcombe.append(xtnervq)
        ycombe.append(ytnervq)
        xcombe.append(xdnervq)
        ycombe.append(ydnervq)
        xcombe.append(xnervext[1, memtqe])
        ycombe.append(ynervext[1, memtqe])
        xcombe.append(xnervext[1, memtqe])
        ycombe.append(ynervext[1, mem])
    else:
        xcombe.append(xnervext[0, 0])
        ycombe.append(ynervext[0, 0])
        xcombe.append(xnervext[1, 0])
        ycombe.append(ynervext[1, 0])
        xcombe.append(xnervext[1, memtqe])
        ycombe.append(ynervext[1, memtqe])
        xcombe.append(xnervext[1, memtqe])
        ycombe.append(ynervext[1, memtqe])


    xcgtqe = sum(xcombe)/4  # CG do tanque externo
    ycgtqe = sum(ycombe)/4

    # Calcula capacidade dos tanques (duas semi-asas)
    limitepe = wing['trunnion_xposition']

    # estacoes da base inf e sup do tanque externo (ymed1 e ymed2)
    ymed1 = 0.50*(ynervext[0, 0]+ynervext[1, 0])
    ymed2 = 0.50*(ynervext[0, memtqe]+ynervext[1, memtqe])

    # estacao superior do tanque interno da asa
    if engine['position'] == 2:
        ymed3 = ytnervq
        limitepi1 = wing['trunnion_xposition']

        aux = max(xnervint[0, nni-1], xnervint[1, nni-1])
        limitepi2 = (aux-yfusjunc*np.tan(rad*wing['sweep_leading_edge']))/Cinter
    else:
        ymed3 = ynervint[1, 0]
        deltaysta = yquebra-yfusjunc
        Csupint = Cinter + ((ymed3-yfusjunc)/deltaysta)*(wing['kink_chord']-Cinter)
        xbainter = yfusjunc*np.tan(rad*wing['sweep_leading_edge'])
        xbaint = xbainter + ((ymed3-yfusjunc)/deltaysta)*(xquebraBA-xbainter)
        aux = max(xnervint[0, 0], xnervint[1, 0])
        limitepi1 = (aux-xbaint)/Csupint
        aux = max(xnervint[0, nni-1], xnervint[1, nni-1])
        limitepi2 = (aux-yfusjunc*np.tan(rad*wing['sweep_leading_edge']))/Cinter


    yinterno = ymed1
    yexterno = ymed2
    # acha perfil externo
    deltay = wing['span']/2-yinterno

    xuperfilext = xutip
    yuperfilext = yukink + ((yexterno-yinterno)/deltay)*(yutip-yukink)
    ylperfilext = ylkink + ((yexterno-yinterno)/deltay)*(yltip-ylkink)
    Cext = wing['kink_chord'] + ((yexterno-yinterno)/deltay)*(wing['tip_chord']-wing['kink_chord'])
    heighte = ymed2-ymed1
    icount = 0

    xpolye = []
    ypolye = []
    xpolyi = []
    ypolyi = []
    for i in range(0, nukink, 1):
        if xuperfilext[i] < limitepe and xuperfilext[i] >= limited:
            icount = icount+1
            xpolye.append(xuperfilext[i])
            ypolye.append(yuperfilext[i])
            xpolyi.append(xukink[i])
            ypolyi.append(yukink[i])

    for i in range(nukink-1, 0, -1):
        if xuperfilext[i] < limitepe and xuperfilext[i] >= limited:
            icount = icount+1
            xpolye.append(xuperfilext[i])
            ypolye.append(ylperfilext[i])
            xpolyi.append(xlkink[i])
            ypolyi.append(ylkink[i])

    # a percentagem da corda na secao da raiz nao eh a memsma da long traseira
    yinterno = yfusjunc
    yexterno = ymed3
    # acha perfil externo
    deltay = yquebra-yinterno

    gradaux = (yexterno-yinterno)/deltay

    xuperfilint = xuroot

    yuperfilint = []
    ylperfilint = []

    for j in range(len(xuroot)):
        yuperfilint.append(yuroot[j] + gradaux*(yukink[j]-yuroot[j]))
        ylperfilint.append(ylroot[j] + gradaux*(ylkink[j]-ylroot[j]))

    Csupint = Cinter + ((yexterno-yinterno)/deltay)*(wing['kink_chord']-Cinter)

    limitedr = limited

    icount = 0

    xpolyroot = []
    ypolyroot = []

    for i in range(0, len(xuroot), 1):
        if xuperfilint[i] <= limitepi1 and xuperfilint[i] >= limitedr:
            icount = icount+1
            xpolyroot.append(xuroot[i])
            ypolyroot.append(yuperfilint[i])
    # CHECKKKKKKKKKK THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    for i in range(len(xuroot)-1, 0, -1):

        if xuperfilint[i-1] <= limitepi1 and xuperfilint[i-1] >= limitedr:
            icount = icount+1
            xpolyroot.append(xuroot[i])
            ypolyroot.append(ylperfilint[i])

    # Area molhada no perfil da interseccao asa-fuselagem
    icount = 0

    xpolyroot1 = []
    ypolyroot1 = []
    for i in range(0, len(xuroot), 1):
        if xuroot[i] <= limitepi2 and xuroot[i] >= limitedr:
            icount = icount+1
            xpolyroot1.append(xuroot[i])
            ypolyroot1.append(yuroot[i])

    # CHECKKKKKKKKKK THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    for i in range(len(xuroot)-1, 0, -1):
        if xuroot[i-1] <= limitepi2 and xuroot[i-1] >= limitedr:
            icount = icount+1
            xpolyroot1.append(xuroot[i])
            ypolyroot1.append(ylroot[i])

    def PolyArea(x, y):
        return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    areae = PolyArea(xpolye, ypolye)
    areae = areae*Cext*Cext
    areai = PolyArea(xpolyi, ypolyi)
    areai = areai*wing['kink_chord']*wing['kink_chord']

    arearootsup = PolyArea(xpolyroot, ypolyroot)
    arearootsup = arearootsup*Csupint*Csupint
    arearootinf = PolyArea(xpolyroot1, ypolyroot1)
    arearootinf = arearootinf*Cinter*Cinter
    # Calculo dos volumes
    # 2# de perdas devido a nervuras, revestimento e outros equip
    voltanqueext = 0.98*(heighte/3)*(areai + areae + np.sqrt(areai*areae))
    # 2# de perdas devido a nervuras, revestimento e outros equip
    voltanqueint = 0.98*(deltay/3)*(arearootinf +
                                    arearootsup + np.sqrt(arearootinf*arearootsup))

    capacidadete = 2*voltanqueext*querosene_density

    # Capacidade dos tanques da asa interna
    xcombi = []
    ycombi = []
    if engine['position'] == 2:
        xcombi.append(xdnervq)
        ycombi.append(ydnervq)
        xcombi.append(xtnervq)
        ycombi.append(ytnervq)
        xcombi.append(xnervint[1, nni-1])
        ycombi.append(ynervint[1, nni-1])
        xcombi.append(xnervint[0, nni-1])
        ycombi.append(ynervint[0, nni-1])
    else:
        xcombi.append(xnervint[0, 0])
        ycombi.append(ynervint[0, 0])
        xcombi.append(xnervint[1, 0])
        ycombi.append(ynervint[1, 0])
        xcombi.append(xnervint[1, nni-1])
        ycombi.append(ynervint[1, nni-1])
        xcombi.append(xnervint[0, nni-1])
        ycombi.append(ynervint[0, nni-1])


    xcgtqi = sum(xcombi)/4  # CG do tanque interno
    ycgtqi = sum(ycombi)/4

    capacidadeti = 2*voltanqueint*querosene_density

    # Capacidade total dos tanques
    # Considera perdas devido a nervuras, longarinas, revestimento, bombas
    # etc...
    if capacidadeti > 0 and capacidadete > 0:
        wing['fuel_capacity'] = capacidadeti + capacidadete

    # Localizacao do CG dos tanques de combustivel
    wing['tank_center_of_gravity_xposition'] = (xcgtqe*capacidadete + xcgtqi*capacidadeti) / \
        (capacidadeti+capacidadete)

    # *** Flape externo ***
    xflape = []
    yflape = []
    xflape.append(xail[2]+1)
    yflape.append(yail[2] - 0.10)

    x1aux = xquebraBA + wing['kink_chord']
    y1aux = yquebra
    x2aux = bdiv2*np.tan(rad*wing['sweep_leading_edge']) + wing['tip_chord']
    y2aux = bdiv2

    if x1aux == x2aux:
        xflape[0] = x1aux
    else:
        gradbfaux = (y2aux-y1aux)/(x2aux-x1aux)
        xflape[0] = (yflape[0]-y1aux)/gradbfaux + x1aux

    # Intersecao com a LT
    tg1 = 0
    tg2 = inclnt
    x1 = xail[2]
    y1 = yail[2]-0.10
    x2 = xnervext[0, mem]
    y2 = ynervext[0, mem]
    term1 = (y2-y1)-tg2*x2+tg1*x1
    xflape.append(term1/(tg1-tg2))
    yflape.append(y1+tg1*(xflape[1]-x1))
    xflape.append(x02)
    yflape.append(y02)
    xflape.append(fquebra*bdiv2*np.tan(rad*wing['sweep_leading_edge'])+wing['kink_chord'])
    yflape.append(fquebra*bdiv2)

    def PolyArea(x, y):
        return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    wing['flap_area'] = PolyArea(xflape, yflape)

    wing['flap_chord'] = wing['flap_area']/(wing['flap_span']*(wing['span']/2))


    # Insere nervura auxiliar para o flape externo
    xnervaf = []
    ynervaf = []
    xnervaf.append(xflape[1])
    ynervaf.append(yflape[1])

    x1aux = yfusjunc*np.tan(rad*wing['sweep_leading_edge']) + fraclongdi*Cinter
    y1aux = yfusjunc
    x2aux = bdiv2*np.tan(rad*wing['sweep_leading_edge']) + fraclongdi*wing['tip_chord']
    y2aux = bdiv2

    if x1aux == x2aux:
        xnervaf.append(x1aux)
        ynervaf.append(yflape[1])
    else:
        tg1 = (y2aux-y1aux)/(x2aux-x1aux)
        tg2 = np.tan(angnev)
        x1 = x1aux
        y1 = y1aux
        x2 = xflape[1]
        y2 = yflape[1]
        term1 = (y2-y1)-tg2*x2+tg1*x1
        xnervaf.append(term1/(tg1-tg2))
        ynervaf.append(y2+tg2*(xnervaf[1]-x2))

    xflapi = []
    yflapi = []

    # Flape interno
    xflapi.append(xflape[3])
    yflapi.append(yflape[3])
    xflapi.append(xflape[2])
    yflapi.append(yflape[2])
    cordafi = xflape[3]-xflape[2]
    xflapi.append(yfusjunc*(np.tan(rad*wing['sweep_leading_edge'])) + Cinter - cordafi)
    yflapi.append(yfusjunc)
    xflapi.append(yfusjunc*(np.tan(rad*wing['sweep_leading_edge'])) + Cinter)
    yflapi.append(yfusjunc)

    # posicao em x do munhao
    wing['trunnion_xposition'] = (xflapi[2]+xltintern)/2

    if os.path.exists('wlayout.jpg'):
        os.remove('wlayout.jpg')


    if os.path.exists('tankprofiles.jpg'):
        os.remove('tankprofiles.jpg')

    # Check de consistencia
    if wing['fuel_capacity'] <= 0 or wing['flap_area'] <= 0:
        checkconsistency = 1  # fail
    else:
        checkconsistency = 0  # ok

    return vehicle
