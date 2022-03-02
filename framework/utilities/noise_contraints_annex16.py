import numpy as np
import matplotlib.pyplot as plt

def noise_constraint(mtow):
    if mtow>=1000 and mtow<=3950:
        EPNdb_constraint = 247
    elif mtow > 3950 and mtow <= 9250:
        m = (265-247)/(9250-3950)
        y = m*(mtow-3950)+247
        EPNdb_constraint = y
    elif mtow >9250 and mtow <= 42000:
        EPNdb_constraint = 265
    elif mtow >42000 and mtow <= 380000:
        m = (295-265)/(380000-42000)
        y = m*(mtow-42000)+265
        EPNdb_constraint = y
    elif mtow >380000:
        EPNdb_constraint = 295

    return EPNdb_constraint


# mtow = np.linspace(1000,500000,10000)

# EPNdb_vec = [noise_constraint(x) for x in mtow]

# plt.plot(mtow,EPNdb_vec)
# plt.plot(75000,280,'o')
# plt.grid()
# plt.show()