def Rosen_Obj(x,k):
    ## Rosenbrock function 
    Y = (1*k[0]-x[0]) ** 2 + 100*k[1] * (x[1] - x[0] ** 2) ** 2
    return Y


def Rosen_Cons_1(x,k):
    ## F[x] <= 1.5
  C = (x[0] + 1*k[2]) ** 2 + (x[1] + 0.3*k[2]) ** 2 - 1
  return C

def Rosen_Cons_2(x, k):
    ## Constraint function from R² -> R
    ## F[x] <= 1.5
  C = -0.05-2*x[0]-0.8*k[3]*x[1]-k[3]* 0.01 * ((1-x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

  return C

def Rosen_Pyopt(x, **k):
    ## Constraint function from R² -> R
    ## F[x] <= 1.5
    k1 = k['k'][0]
    k2 = k['k'][0]
    k3 = k['k'][0]
    k4 = k['k'][0]
    
    Y = (1*k1-x[0]) ** 2 + 100*k2 * (x[1] - x[0] ** 2) ** 2
    C = [0.0, 0.0]
    C[0] = (x[0] + 1*k3) ** 2 + (x[1] + 0.3*k3) ** 2 - 1
    C[1] = -0.05-2*x[0]-0.8*k4*x[1]-k4* 0.01 * ((1-x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

    fail = 0

    return Y, C, fail
