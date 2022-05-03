import pickle
import numpy as np


import tixi3.tixi3wrapper as tixi3wrapper
import tigl3.tigl3wrapper as tigl3wrapper
from tixi3.tixi3wrapper import Tixi3Exception
from tigl3.tigl3wrapper import Tigl3Exception

import os
# import sys
#==============================================================================
#   CLASSES
#==============================================================================


#==============================================================================
#   FUNCTIONS
#==============================================================================

def open_tixi(cpacs_path):
    """ Create TIXI handles for a CPACS file and return this handle.

    Function 'open_tixi' return the TIXI Handle of a CPACS file given as input
    by its path. If this operation is not possible, it returns 'None'

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        cpacs_path (str): Path to the CPACS file

    Returns::
        tixi_handle (handles): TIXI Handle of the CPACS file
    """

    tixi_handle = tixi3wrapper.Tixi3()
    tixi_handle.open(cpacs_path)

    # log.info('TIXI handle has been created.'+cpacs_path)

    return tixi_handle


def open_tigl(tixi_handle):
    """ Create TIGL handles for a CPACS file and return this handle.

    Function 'open_tigl' return the TIGL Handle from its TIXI Handle.
    If this operation is not possible, it returns 'None'

    Source :
        * TIGL functions http://tigl.sourceforge.net/Doc/index.html

    Args:
        tixi_handle (handles): TIXI Handle of the CPACS file

    Returns:
        tigl_handle (handles): TIGL Handle of the CPACS file
    """

    tigl_handle = tigl3wrapper.Tigl3()
    tigl_handle.open(tixi_handle, '')

    tigl_handle.logSetVerbosity(1)  # 1 - only error, 2 - error and warnings

    # log.info('TIGL handle has been created.')
    return tigl_handle


def close_tixi(tixi_handle, cpacs_out_path):
    """ Close TIXI handle and save the CPACS file.

    Function 'close_tixi' close the TIXI Handle and save the CPACS file at the
    location given by 'cpacs_out_path' after checking if the directory path
    exists

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi_handle (handles): TIXI Handle of the CPACS file
        cpacs_out_path (str): Path to the CPACS output file

    """

    # Check if the directory of 'cpacs_out_path' exist, if not, create it
    path_split = cpacs_out_path.split('\\')[:-1]
    dir_path = '\\'.join(str(m) for m in path_split)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        # log.info(str(dir_path) + ' directory has been created.')

    # Save CPACS file
    tixi_handle.save(cpacs_out_path)
    # log.info("Output CPACS file has been saved at: " + cpacs_out_path)

    # Close TIXI handle
    tixi_handle.close()
    # log.info("TIXI Handle has been closed.")


def create_branch(tixi, xpath, add_child=False):
    """ Function to create a CPACS branch.

    Function 'create_branch' create a branch in the tixi handle and also all
    the missing parent nodes. Be careful, the xpath must be unique until the
    last element, it means, if several element exist, its index must be precised
    (index start at 1).
    e.g.: '/cpacs/vehicles/aircraft/model/wings/wing[2]/name'

    If the entire xpath already exist, the option 'add_child' (True/False) lets
    the user decide if a named child should be added next to the existing
    one(s). This only valid for the last element of the xpath.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi (handles): TIXI Handle of the CPACS file
        xpath (str): xpath of the branch to create
        add_child (boolean): Choice of adding a name child if the last element
                             of the xpath if one already exists

    Returns:
        tixi (handles): Modified TIXI Handle (with new branch)
    """

    xpath_split = xpath.split("/")
    xpath_count = len(xpath_split)

    for i in range(xpath_count-1):
        xpath_index = i + 2
        xpath_partial = '/'.join(str(m) for m in xpath_split[0:xpath_index])
        xpath_parent = '/'.join(str(m) for m in xpath_split[0:xpath_index-1])
        child = xpath_split[(xpath_index-1)]
        if tixi.checkElement(xpath_partial):
            # log.info('Branch "' + xpath_partial + '" already exist')

            if child == xpath_split[-1] and add_child:
                namedchild_nb = tixi.getNamedChildrenCount(xpath_parent, child)
                tixi.createElementAtIndex (xpath_parent, child, namedchild_nb+1)
                log.info('Named child "' + child
                         + '" has been added to branch "'
                         + xpath_parent + '"')
        else:
            tixi.createElement(xpath_parent, child)
            log.info('Child "' + child + '" has been added to branch "'
                     + xpath_parent + '"')


def copy_branch(tixi, xpath_from, xpath_to):
    """ Function to copy a CPACS branch.

    Function 'copy_branch' copy the branch (with sub-branches) from
    'xpath_from' to 'xpath_to' by using recursion. The new branch should
    be identical (uiD, attribute, etc). There is no log in this function
    because of its recursivity.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi_handle (handles): TIXI Handle of the CPACS file
        xpath_from (str): xpath of the branch to copy
        xpath_to (str): Destination xpath

    Returns:
        tixi (handles): Modified TIXI Handle (with copied branch)
    """

    if not tixi.checkElement(xpath_from):
        raise ValueError(xpath_from + ' XPath does not exist!')
    if not tixi.checkElement(xpath_to):
        raise ValueError(xpath_to + ' XPath does not exist!')

    child_nb = tixi.getNumberOfChilds(xpath_from)

    if child_nb:
        xpath_to_split = xpath_to.split("/")
        xpath_to_parent = '/'.join(str(m) for m in xpath_to_split[:-1])

        child_list = []
        for i in range(child_nb):
            child_list.append(tixi.getChildNodeName(xpath_from, i+1))

        # If it is a text Element --> no child
        if "#" in child_list[0]:
            elem_to_copy = tixi.getTextElement(xpath_from)
            tixi.updateTextElement(xpath_to, elem_to_copy)

        else:
            # If child are named child (e.g. wings/wing)
            if all(x == child_list[0] for x in child_list):
                namedchild_nb = tixi.getNamedChildrenCount(xpath_from, 
                                                           child_list[0])

                for i in range(namedchild_nb):
                    new_xpath_from = xpath_from + "/" + child_list[0] \
                                     + '[' + str(i+1) + ']'
                    new_xpath_to = xpath_to + "/" + child_list[0] \
                                   + '[' + str(i+1) + ']'
                    tixi.createElement(xpath_to, child_list[0])

                    # Call the function itself for recursion
                    copy_branch(tixi, new_xpath_from, new_xpath_to)

            else:
                for child in child_list:
                    new_xpath_from = xpath_from + "/" + child
                    new_xpath_to = xpath_to + "/" + child

                    # Create child
                    tixi.createElement(xpath_to, child)

                    # Call the function itself for recursion
                    copy_branch(tixi, new_xpath_from, new_xpath_to)

        # Copy attribute(s) if exists
        last_attrib = 0
        attrib_index = 1
        while not last_attrib:
            try:
                attrib_name = tixi.getAttributeName(xpath_from, attrib_index)
                attrib_text = tixi.getTextAttribute(xpath_from, attrib_name)
                tixi.addTextAttribute(xpath_to, attrib_name, attrib_text)
                attrib_index = attrib_index + 1
            except:
                last_attrib = 1


def get_uid(tixi, xpath):
    """ Function to get uID from a specific XPath.

    Function 'get_uid' checks the xpath and get the corresponding uID.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi (handles): TIXI Handle of the CPACS file
        xpath (str): xpath of the branch to add the uid

    Returns:
        uid (str): uid to add at xpath
    """


    if not tixi.checkElement(xpath):
        raise ValueError(xpath + ' XPath does not exist!')

    if tixi.checkAttribute(xpath, 'uID'):
        uid = tixi.getTextAttribute(xpath, 'uID')
        return uid
    else:
        raise ValueError("No uID found for: " + xpath)



def add_uid(tixi, xpath, uid):
    """ Function to add UID at a specific XPath.

    Function 'add_uid' checks and add UID to a specific path, the function will
    automatically update the chosen UID if it exists already.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi (handles): TIXI Handle of the CPACS file
        xpath (str): xpath of the branch to add the uid
        uid (str): uid to add at xpath

    """

    exist = True
    uid_new = uid
    i = 0
    while exist is True:
        if not tixi.uIDCheckExists(uid_new):
            tixi.uIDSetToXPath(xpath, uid_new)
            exist = False
        else:
            i = i + 1
            uid_new = uid + str(i)
            log.warning('UID already existing changed to: ' + uid_new)


def get_value(tixi, xpath):
    """ Function to get value from a CPACS branch if this branch exist.

    Function 'get_value' check first if the the xpath exist and a value is store
    at this place. Then, it gets and returns this value. If the value or the
    xpath does not exist it raise an error and return 'None'.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi_handle (handles): TIXI Handle of the CPACS file
        xpath (str): xpath of the value to get

    Returns:
         value (float or str): Value found at xpath
    """

    # Try to get the a value at xpath
    try:
        value = tixi.getTextElement(xpath)
    except:
        value = None

    if value:
        try: # check if it is a 'float'
            is_float = isinstance(float(value), float)
            value = float(value)
        except:
            pass
    else:
        # check if the path exist
        if tixi.checkElement(xpath):
            log.error('No value has been found at ' + xpath)
            raise ValueError('No value has been found at ' + xpath)
        else:
            log.error(xpath + ' cannot be found in the CPACS file')
            raise ValueError(xpath + ' cannot be found in the CPACS file')

    # Special return for boolean
    if value == 'True':
        return True
    elif value == 'False':
        return False

    return value


def get_value_or_default(tixi, xpath, default_value):
    """ Function to get value from a CPACS branch if this branch exist, if not
        it returns the default value.

    Function 'get_value_or_default' do the same than the function 'get_value'
    but if no value is found at this place it returns the default value and add
    it at the xpath. If the xpath does not exist, it is created.

    Source :
        * TIXI functions: http://tixi.sourceforge.net/Doc/index.html

    Args:
        tixi_handle (handles): TIXI Handle of the CPACS file
        xpath (str): xpath of the value to get
        default_value (str, float or int): Default value

    Returns:
        tixi (handles): Modified TIXI Handle (with added default value)
        value (str, float or int): Value found at xpath
    """

    value = None
    try:
        value = get_value(tixi, xpath)
    except:
        pass

    if value is None:
        log.info('Default value will be used instead')
        value = default_value

        xpath_parent = '/'.join(str(m) for m in xpath.split("/")[:-1])
        value_name = xpath.split("/")[-1]
        create_branch(tixi, xpath_parent, False)

        is_int = False
        is_float = False
        is_bool = False
        try: # check if it is an 'int' or 'float'
            is_int = isinstance(float(default_value), int)
            is_float = isinstance(float(default_value), float)
            is_bool = isinstance(default_value, bool)
        except:
            pass
        if is_bool:
           tixi.addTextElement(xpath_parent, value_name, str(value))
        elif is_float or is_int:
            value = float(default_value)
            tixi.addDoubleElement(xpath_parent, value_name, value, '%g')
        else:
            value = str(value)
            tixi.addTextElement(xpath_parent, value_name, value)
        log.info('Default value has been added to the cpacs file at: ' + xpath)
    else:
        log.info('Value found at ' + xpath + ', default value will not be used')

        # Special return for boolean
        if value == 'True':
            return True
        elif value == 'False':
            return False
        elif isinstance(value, bool):
            return value

    return value


def add_float_vector(tixi, xpath, vector):
    """ Add a vector (of float) at given CPACS xpath

    Function 'add_float_vector' will add a vector (composed by float) at the
    given XPath, if the node does not exist, it will be created. Values will be
    overwritten if paths exists.

    Args:
        tixi (handle): Tixi handle
        xpath (str): XPath of the vector to add
        vector (list, tuple): Vector of floats to add
    """

    # Strip trailing '/' (has no meaning here)
    if xpath.endswith('/'):
        xpath = xpath[:-1]

    # Get the field name and the parent CPACS path
    xpath_child_name = xpath.split("/")[-1]
    xpath_parent = xpath[:-(len(xpath_child_name)+1)]

    if not tixi.checkElement(xpath_parent):
        create_branch(tixi, xpath_parent)

    if tixi.checkElement(xpath):
        tixi.updateFloatVector(xpath, vector, len(vector), format='%g')
        tixi.addTextAttribute(xpath, 'mapType', 'vector')
    else:
        tixi.addFloatVector(xpath_parent, xpath_child_name, vector, \
                            len(vector), format='%g')
        tixi.addTextAttribute(xpath, 'mapType', 'vector')


def get_float_vector(tixi, xpath):
    """ Get a vector (of float) at given CPACS xpath

    Function 'get_float_vector' will get a vector (composed by float) at the
    given XPath, if the node does not exist, an error will be raised.

    Args:
        tixi (handle): Tixi handle
        xpath (str): XPath of the vector to get
    """

    if not tixi.checkElement(xpath):
        raise ValueError(xpath + ' path does not exist!')

    float_vector_str = tixi.getTextElement(xpath)

    if float_vector_str == '':
        raise ValueError('No value has been fournd at ' + xpath)

    if float_vector_str.endswith(';'):
        float_vector_str = float_vector_str[:-1]
    float_vector_list = float_vector_str.split(';')
    float_vector = [float(elem) for elem in float_vector_list]

    return float_vector


def add_string_vector(tixi, xpath, vector):
    """ Add a vector (of string) at given CPACS xpath

    Function 'add_string_vector' will add a vector (composed by stings) at the
    given XPath, if the node does not exist, it will be created. Values will be
    overwritten if paths exists.

    Args:
        tixi (handle): Tixi handle
        xpath (str): XPath of the vector to add
        vector (list): Vector of string to add
    """

    # Strip trailing '/' (has no meaning here)
    if xpath.endswith('/'):
        xpath = xpath[:-1]

    # Get the field name and the parent CPACS path
    xpath_child_name = xpath.split("/")[-1]
    xpath_parent = xpath[:-(len(xpath_child_name)+1)]

    vector_str = ";".join([str(elem) for elem in vector])

    if not tixi.checkElement(xpath_parent):
        create_branch(tixi, xpath_parent)

    if tixi.checkElement(xpath):
        tixi.updateTextElement(xpath, vector_str)
    else:
        tixi.addTextElement(xpath_parent, xpath_child_name, vector_str)


def get_string_vector(tixi, xpath):
    """ Get a vector (of string) at given CPACS xpath

    Function 'get_string_vector' will get a vector (composed by string) at the
    given XPath, if the node does not exist, an error will be raised.

    Args:
        tixi (handle): Tixi handle
        xpath (str): XPath of the vector to get
    """

    if not tixi.checkElement(xpath):
        raise ValueError(xpath + ' path does not exist!')

    string_vector_str = tixi.getTextElement(xpath)

    if string_vector_str == '':
        raise ValueError('No value has been fournd at ' + xpath)

    if string_vector_str.endswith(';'):
        string_vector_str = string_vector_str[:-1]
    string_vector_list = string_vector_str.split(';')
    string_vector = [str(elem) for elem in string_vector_list]

    return string_vector


def get_path(tixi, xpath):
    """ Get a path with your os system format

    Function 'get_path' will get a get the path in the CPACS file and return
    a path with the format corresponding to your os ('/' for Linux and MacOS
    and '\' for Windows). All paths to store in the CPACS file could be saved as
    normal strings as long as this function is used to get them back.

    Args:
        tixi (handle): Tixi handle
        xpath (str): XPath of the path to get

    Returns:
        correct_path (str): Path with the correct separators

    """

    path_str = get_value(tixi, xpath)

    if ('/' in path_str and '\\' in path_str):
        raise ValueError('Request path format is unrecognized!')
    elif '/' in path_str:
        path_list = path_str.split('/')
    elif '\\' in path_str:
        path_list = path_str.split('\\')
    else:
        raise ValueError('No path has been recognized!')

    correct_path = os.path.join('', *path_list)

    return correct_path


def aircraft_name(tixi_or_cpacs):
    """ The function get the name of the aircraft from the cpacs file or add a
        default one if non-existant.

    Args:
        cpacs_path (str): Path to the CPACS file

    Returns:
        name (str): Name of the aircraft.
    """

    # TODO: MODIFY this funtion, temporary it could accept a cpacs path or tixi handle
    # check xpath
    # *modify corresponding test

    if isinstance(tixi_or_cpacs, str):

        tixi = open_tixi(tixi_or_cpacs)

        aircraft_name_xpath = '/cpacs/header/name'
        name = get_value_or_default(tixi, aircraft_name_xpath, 'Aircraft')

        close_tixi(tixi, tixi_or_cpacs)

    else:

        aircraft_name_xpath = '/cpacs/header/name'
        name = get_value_or_default(tixi_or_cpacs, aircraft_name_xpath, 'Aircraft')

    name = name.replace(' ', '_')
    # log.info('The name of the aircraft is : ' + name)

    return(name)


def plot3d_tigl(vehicle):

    MODULE_DIR = 'c:/Users/aarc8/Documents/github\MDOAirB_base/framework/CPACS_update'
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



with open('Database/Family/40_to_100/all_dictionaries/'+str(1)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)


vehicle = all_info_acft1['vehicle']
plot3d_tigl(vehicle)