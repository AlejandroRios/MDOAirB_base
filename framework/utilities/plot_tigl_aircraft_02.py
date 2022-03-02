from framework.CPACS_update.cpacsfunctions import *
import numpy as np
def plot3d_tigl(vehicle):

    MODULE_DIR = 'c:/Users/aarc8/Documents/github\MDOAirB/framework/CPACS_update'
    cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'baseline_in.xml')
    cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'baseline_out.xml')
    tixi = open_tixi(cpacs_out_path)
    tigl = open_tigl(tixi)

    tixi_out = open_tixi(cpacs_out_path)
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']
    aircraft = vehicle['aircraft']


    delta_x = (wing['center_chord'] - wing['tip_chord'])/4 + wing['semi_span']*np.tan((wing['sweep_c_4']*np.pi)/180)
    delta_x_kink = (wing['root_chord'] - wing['kink_chord'])/4 + wing['semi_span_kink']*wing['semi_span']*np.tan((wing['sweep_c_4']*np.pi)/180)
    delta_x_tip = (wing['root_chord'] - wing['tip_chord'])/4 + wing['semi_span']*np.tan((wing['sweep_c_4']*np.pi)/180)

    # Wing ------------------------------------
    # center chord
    cc_w = wing['center_chord']
    xc_w = wing['leading_edge_xposition']
    yc_w = 0
    zc_w = 0
    # root chord
    cr_w = wing['root_chord']
    xr_w = wing['leading_edge_xposition']
    yr_w = wing['root_chord_yposition']
    zr_w = 0
    # kink chord
    ck_w = wing['kink_chord']
    xk_w = wing['leading_edge_xposition'] + delta_x_kink
    yk_w = wing['semi_span_kink']*wing['semi_span']
    zk_w = 0
    # tip chord
    ct_w = wing['tip_chord']
    xt_w = xr_w + delta_x
    yt_w = wing['semi_span']
    zt_w = 0 + wing['semi_span']*(np.tan((wing['dihedral']*np.pi)/180))

    # EH ------------------------------------
    # root chord
    cr_h = horizontal_tail['center_chord']
    xr_h = horizontal_tail['leading_edge_xposition']
    yr_h = 0
    zr_h = 0
    # tip chord
    ct_h = horizontal_tail['taper_ratio']*horizontal_tail['center_chord']
    yt_h = (horizontal_tail['span']/2)
    xt_h = xr_h + (cr_h - ct_h)/4 + yt_h*np.tan((horizontal_tail['sweep_c_4']*np.pi)/180)
    zt_h = 0

    # EV ------------------------------------
    # root chord
    cr_v = vertical_tail['center_chord']
    xr_v = vertical_tail['leading_edge_xposition']
    yr_v = 0
    zr_v = 0
    
    # tip chord
    ct_v = vertical_tail['tip_chord']
    xt_v = xr_v + cr_v/4 + vertical_tail['span']*np.tan((vertical_tail['sweep_c_4']*np.pi)/180) - ct_v/4
    yt_v = 0
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




    # fuselage_xpath = '/cpacs/vehicles/aircraft/model/fuselages/fuselage[1]/'
    
    # # Update leading edge position
    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[8]/length',fuselage['cabine_length']/4, '%g')
    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[9]/length',fuselage['cabine_length']/4, '%g')
    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[10]/length',fuselage['cabine_length']/4, '%g')
    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[11]/length',fuselage['cabine_length']/4, '%g')

    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[12]/length',fuselage['tail_length']/2, '%g')
    # tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[13]/length',fuselage['tail_length']/2, '%g')

    # nominal_diameter = 2.0705*2
    # scale_factor = fuselage['diameter']/nominal_diameter

    # tixi_out.updateDoubleElement(fuselage_xpath+'transformation/scaling/y',scale_factor, '%g')
    # tixi_out.updateDoubleElement(fuselage_xpath+'transformation/scaling/z',scale_factor, '%g')


    wing_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[1]/'
    
    # Update leading edge position
    # tixi_out.updateDoubleElement(wing_xpath+'transformation/translation/x', wing['leading_edge_xposition'], '%g')
    # Update center chord 
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/x', xc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/y', yc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/z', zc_w, '%g')


    # Update root chord 
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/x', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/y', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/z', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/x', xr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/y', yr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/z', zr_w, '%g')

    # Update kink chord 

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/x', ck_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/y', ck_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/z', ck_w, '%g')

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/x', xk_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/y', yk_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/z', zk_w, '%g')
    
    # Update tip chord 

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/x', ct_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/y', ct_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/z', ct_w, '%g')

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/x', xt_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/y', yt_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/z', zt_w, '%g')



    # vertical_tail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[3]/'

    # # Update leading edge position
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'transformation/translation/x', vertical_tail['leading_edge_xposition'], '%g')
    # # Update center chord 
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', vertical_tail['center_chord'], '%g')
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', vertical_tail['center_chord'], '%g')
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', vertical_tail['center_chord'], '%g')

    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', vertical_tail['tip_chord'], '%g')
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', vertical_tail['tip_chord'], '%g')
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', vertical_tail['tip_chord'], '%g')
    # # Update root chord 
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/length',vertical_tail['span'], '%g')
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/sweepAngle',vertical_tail['sweep_leading_edge'], '%g')



    horizontal_thail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[2]/'

    # Update center chord 
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/x', xr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/y', yr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/z', zr_h, '%g')


    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/x', xt_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/y', yt_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/z', zt_h, '%g')

    # Update root chord 

    vertical_tail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[3]/'
    # vertical_tail['leading_edge_xposition'] = vertical_tail['aerodynamic_center_xposition'] - vertical_tail['center_chord']*0.25
    

    # # Update leading edge position
    # tixi_out.updateDoubleElement(vertical_tail_xpath+'transformation/translation/x', vertical_tail['leading_edge_xposition'], '%g')
    # # Update center chord 
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cr_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cr_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cr_v, '%g')

    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', ct_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', ct_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', ct_v, '%g')
    # Update root chord 
    tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/length',vertical_tail['span'], '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/sweepAngle',vertical_tail['sweep_leading_edge'], '%g')

    tixi_out = close_tixi(tixi_out, cpacs_out_path)

        # Reference parameters
    Cref = tigl.wingGetMAC(tigl.wingGetUID(1))
    Sref = tigl.wingGetReferenceArea(1,1)
    b    = tigl.wingGetSpan(tigl.wingGetUID(1))

    print(Cref)
    print(2*Sref)
    print(b)

    return


