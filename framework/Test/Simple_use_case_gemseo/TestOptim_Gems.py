# -------------------------------------------------------------
# -------------------------------------------------------------
# Test case for deterministic constrained optimization of Rosenbrock function
# -------------------------------------------------------------
# -------------------------------------------------------------

from __future__ import division, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ones
from Rosenbrock import *
from gemseo.api import (
    configure_logger,
    create_design_space,
    create_discipline,
    create_scenario,
)

configure_logger("GEMSEO", "INFO")

def Rosen_Obj_Gems(x_local=np.zeros(2)):
    print("x0=",x_local[0])
    print("x1=",x_local[1])
    ## Rosenbrock function with constraint
    obj = ((1-x_local[0]) ** 2 + 100 * (x_local[1] - x_local[0] ** 2) ** 2)
    C_1 = (x_local[0] + 1) ** 2 + (x_local[1] + 0.3) ** 2 - 1
    C_2 = -0.05-2*x_local[0]-0.8*x_local[1]- 0.01 * ((1-x_local[0]) ** 2 + 100 * (x_local[1] - x_local[0] ** 2) ** 2)

    return obj, C_1, C_2

disc_rosen = create_discipline("AutoPyDiscipline", py_func=Rosen_Obj_Gems)
print(disc_rosen.execute())

Xini = array([-1., -1.])
Xmin = array([-3., -3.])
Xmax = array([3., 3.])

design_space = create_design_space()
design_space.add_variable("x_local", 2, l_b=Xmin, u_b=Xmax, value=Xini)

scenario = create_scenario(
       [disc_rosen],
       design_space=design_space,
       formulation="DisciplinaryOpt",
       objective_name = "obj",
   )

scenario.add_constraint("C_1", "ineq")
scenario.add_constraint("C_2", "ineq")

scenario.execute({"max_iter": 999, "algo": "NLOPT_COBYLA"})

xres = scenario.optimization_result.x_opt

obj_opt, cons1_opt, cons2_opt = Rosen_Obj_Gems(xres)

Mean = ones(4)
print('=====================================================')
print('Solution of the deterministic problem : ({},{}), with Rosen = {}'.format(xres[0],xres[1],obj_opt))
print('Constraints values: cons1 = {}, cons2 = {}'.format(cons1_opt,cons2_opt))
print('=====================================================')

# # -------------------------------------------------------------
# # Function graph
# # -------------------------------------------------------------
n = 100
x1 = np.linspace(-0.7, -0.1, n)
y1 = np.linspace(-1.4, -0.8, n)
[X, Y] = np.meshgrid(x1, y1)

Z = Rosen_Obj([X,Y],Mean)
Zcons1 = Rosen_Cons_1([X,Y],Mean)
Zcons2 = Rosen_Cons_2([X,Y],Mean)

plt.figure()
plt.contour(X,Y,Z,200)
plt.colorbar()
plt.contour(x1,y1,Zcons1, [0])
plt.contour(x1,y1,Zcons2, [0])
plt.plot([xres[0]],[xres[1]],marker='o',markersize=15, color ='r')

plt.show()

