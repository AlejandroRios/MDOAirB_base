"""
MDOAirB

Description:
    - This module calculates the wetted area of the fuselage.
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

from framework.Sizing.Geometry.area_triangle_3d import area_triangle_3d
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def wetted_area_forward_fuselage(vehicle):
    """
    Description:
        - This function calculates the wetted area of the forward part of the fuselage.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - total_area - [m2]
    """
    from stl import mesh

    fuselage = vehicle['fuselage']

    mesh = mesh.Mesh.from_file('Database/Fuselage/forwardfus_short.stl')
    # mesh = meshio.read('forwardfus_short.stl')('forwardfus_short.stl')

    xmin = mesh.min_[0]
    xmax = mesh.max_[0]
    ymin = mesh.min_[1]
    ymax = mesh.max_[1]
    zmin = mesh.min_[2]
    zmax = mesh.max_[2]

    # Proceed to scale the mesh to fit into the current airplane configuration
    #xmed = 0.50*(xmin+xmax)
    ymed = 0.50*(ymin+ymax)
    zmed = 0.50*(zmin+zmax)

    dx = xmax-xmin
    dy = ymax-ymin
    dz = zmax-zmin

    scalex = fuselage['cockpit_length']/dx
    scaley = fuselage['width']/dz
    scalez = fuselage['height']/dy

    vertx = mesh.x - xmin
    vertx = vertx * scalex

    verty = mesh.y - ymed
    verty = verty * scalez

    vertz = mesh.z - zmed
    vertz = vertz * scaley

    # verts = np.column_stack((vertx,vertz,verty))
    nfac = len(mesh.vectors)
    areas = []

    # check this point
    for i in range(nfac-1):
        x = vertx[i]
        y = vertz[i]

        z = verty[i]

        area = area_triangle_3d(x, y, z)
        areas.append(area)
        # print(areas)
    total_area = np.sum(areas)

    return(total_area)


def wetted_area_tailcone_fuselage(vehicle):
    """
    Description:
        - This function calculates the wetted area of the tailcone of the fuselage.
    Inputs:
        - vehicle - dictionary containing aircraft parameters
    Outputs:
        - total_area
    """
    fuselage = vehicle['fuselage']

    ai = fuselage['width']/2
    bi = 0.90*fuselage['height']/2
    # Ellipse do final da fuselagem
    af = fuselage['af_ellipse']  # m
    bf = fuselage['bf_ellipse']   # m
    z0f = bi-bf
    #
    n_points = 20
    teta = np.linspace(0, 1, n_points+1)*np.pi

    # #
    xi = np.ones(n_points)*(fuselage['length']-fuselage['tail_length'])
    zi = bi*np.cos(teta)
    yi = ai*np.sin(teta)
    #
    xf = np.ones(n_points)*fuselage['length']
    zf = z0f + bf*np.cos(teta)
    yf = af*np.sin(teta)
    SWET_TC = 0

    xe = []
    ye = []
    ze = []
    xt1 = []
    yt1 = []
    zt1 = []

    xt2 = []
    yt2 = []
    zt2 = []

    areas = []
    for ie in range(n_points-1):
        xe.insert(0, xi[ie])
        xe.insert(1, xf[ie])
        xe.insert(2, xf[ie+1])
        xe.insert(3, xi[ie+1])
        xe.insert(4, xe[0])

        ye.insert(0, yi[ie])
        ye.insert(1, yf[ie])
        ye.insert(2, yf[ie+1])
        ye.insert(3, yi[ie+1])
        ye.insert(4, ye[0])

        ze.insert(0, zi[ie])
        ze.insert(1, zf[ie])
        ze.insert(2, zf[ie+1])
        ze.insert(3, zi[ie+1])
        ze.insert(4, ze[0])

        xt1.insert(0, xe[0])
        xt1.insert(1, xe[1])
        xt1.insert(2, xe[3])

        yt1.insert(0, ye[0])
        yt1.insert(1, ye[1])
        yt1.insert(2, ye[3])

        zt1.insert(0, ze[0])
        zt1.insert(1, ze[1])
        zt1.insert(2, ze[3])

        A1 = area_triangle_3d(xt1, yt1, zt1)

        xt2.insert(0, xe[1])
        xt2.insert(1, xe[2])
        xt2.insert(2, xe[3])

        yt2.insert(0, ye[1])
        yt2.insert(1, ye[2])
        yt2.insert(2, ye[3])

        zt2.insert(0, ze[1])
        zt2.insert(1, ze[2])
        zt2.insert(2, ze[3])

        A2 = area_triangle_3d(xt2, yt2, zt2)

        areas.append(A1+A2)

    total_area = 2*(sum(areas))
    return total_area

# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================