"""
MDOAirB

Description:
    - This function takes the outputs of the network optimization and crates
    density plots and network connection plots

Reference:
    -

TODO's:
    - Crate a funtion from this module

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx
# =============================================================================
# CLASSES
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================
fig, ax = plt.subplots()
m = Basemap(resolution='l', projection='merc', llcrnrlat=30,
            urcrnrlat=70, llcrnrlon=-15, urcrnrlon=50)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='1.0', lake_color='aqua')
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(0.0, 81, 10.0)
# labels = [left, right, top, bottom]
m.drawparallels(parallels, labels=[False, True, True, False])
meridians = np.arange(10.0, 351., 20.0)
m.drawmeridians(meridians, labels=[True, False, False, True])


dic = {('CD0', 'CD0'): 0.0, ('CD0', 'CD1'): 1.0, ('CD0', 'CD2'): 6.0, ('CD0', 'CD3'): 0.0, ('CD0', 'CD4'): 0.0, ('CD0', 'CD5'): 0.0, ('CD0', 'CD6'): 0.0, ('CD0', 'CD7'): 8.0, ('CD0', 'CD8'): 0.0, ('CD0', 'CD9'): 0.0, ('CD1', 'CD0'): 3.0, ('CD1', 'CD1'): 0.0, ('CD1', 'CD2'): 10.0, ('CD1', 'CD3'): 2.0, ('CD1', 'CD4'): 10.0, ('CD1', 'CD5'): 0.0, ('CD1', 'CD6'): 2.0, ('CD1', 'CD7'): 0.0, ('CD1', 'CD8'): 0.0, ('CD1', 'CD9'): 2.0, ('CD2', 'CD0'): 10.0, ('CD2', 'CD1'): 0.0, ('CD2', 'CD2'): 0.0, ('CD2', 'CD3'): 1.0, ('CD2', 'CD4'): 0.0, ('CD2', 'CD5'): 1.0, ('CD2', 'CD6'): 3.0, ('CD2', 'CD7'): 2.0, ('CD2', 'CD8'): 9.0, ('CD2', 'CD9'): 0.0, ('CD3', 'CD0'): 0.0, ('CD3', 'CD1'): 2.0, ('CD3', 'CD2'): 0.0, ('CD3', 'CD3'): 0.0, ('CD3', 'CD4'): 4.0, ('CD3', 'CD5'): 0.0, ('CD3', 'CD6'): 0.0, ('CD3', 'CD7'): 0.0, ('CD3', 'CD8'): 0.0, ('CD3', 'CD9'): 6.0, ('CD4', 'CD0'): 0.0, ('CD4', 'CD1'): 0.0, ('CD4', 'CD2'): 10.0, ('CD4', 'CD3'): 0.0, ('CD4', 'CD4'): 0.0, ('CD4', 'CD5'): 0.0, ('CD4', 'CD6'): 0.0, ('CD4', 'CD7'): 0.0, ('CD4', 'CD8'): 0.0, ('CD4', 'CD9'): 0.0,
       ('CD5', 'CD0'): 3.0, ('CD5', 'CD1'): 0.0, ('CD5', 'CD2'): 1.0, ('CD5', 'CD3'): 1.0, ('CD5', 'CD4'): 4.0, ('CD5', 'CD5'): 0.0, ('CD5', 'CD6'): 1.0, ('CD5', 'CD7'): 0.0, ('CD5', 'CD8'): 0.0, ('CD5', 'CD9'): 0.0, ('CD6', 'CD0'): 0.0, ('CD6', 'CD1'): 0.0, ('CD6', 'CD2'): 6.0, ('CD6', 'CD3'): 0.0, ('CD6', 'CD4'): 6.0, ('CD6', 'CD5'): 1.0, ('CD6', 'CD6'): 0.0, ('CD6', 'CD7'): 0.0, ('CD6', 'CD8'): 0.0, ('CD6', 'CD9'): 10.0, ('CD7', 'CD0'): 4.0, ('CD7', 'CD1'): 7.0, ('CD7', 'CD2'): 0.0, ('CD7', 'CD3'): 0.0, ('CD7', 'CD4'): 0.0, ('CD7', 'CD5'): 7.0, ('CD7', 'CD6'): 0.0, ('CD7', 'CD7'): 0.0, ('CD7', 'CD8'): 10.0, ('CD7', 'CD9'): 0.0, ('CD8', 'CD0'): 0.0, ('CD8', 'CD1'): 3.0, ('CD8', 'CD2'): 0.0, ('CD8', 'CD3'): 0.0, ('CD8', 'CD4'): 0.0, ('CD8', 'CD5'): 2.0, ('CD8', 'CD6'): 0.0, ('CD8', 'CD7'): 1.0, ('CD8', 'CD8'): 0.0, ('CD8', 'CD9'): 0.0, ('CD9', 'CD0'): 0.0, ('CD9', 'CD1'): 3.0, ('CD9', 'CD2'): 9.0, ('CD9', 'CD3'): 0.0, ('CD9', 'CD4'): 0.0, ('CD9', 'CD5'): 0.0, ('CD9', 'CD6'): 0.0, ('CD9', 'CD7'): 0.0, ('CD9', 'CD8'): 10.0, ('CD9', 'CD9'): 0.0}
# dic ={('CD0', 'CD0'): 0.0, ('CD0', 'CD1'): 0.0, ('CD0', 'CD2'): 10.0, ('CD0', 'CD3'): 7.0, ('CD0', 'CD4'): 0.0, ('CD0', 'CD5'): 0.0, ('CD0', 'CD6'): 0.0, ('CD0', 'CD7'): 1.0, ('CD0', 'CD8'): 0.0, ('CD0', 'CD9'): 0.0, ('CD1', 'CD0'): 2.0, ('CD1', 'CD1'): 0.0, ('CD1', 'CD2'): 10.0, ('CD1', 'CD3'): 2.0, ('CD1', 'CD4'): 10.0, ('CD1', 'CD5'): 0.0, ('CD1', 'CD6'): 4.0, ('CD1', 'CD7'): 1.0, ('CD1', 'CD8'): 1.0, ('CD1', 'CD9'): 4.0, ('CD2', 'CD0'): 5.0, ('CD2', 'CD1'): 0.0, ('CD2', 'CD2'): 0.0, ('CD2', 'CD3'): 0.0, ('CD2', 'CD4'): 0.0, ('CD2', 'CD5'): 1.0, ('CD2', 'CD6'): 10.0, ('CD2', 'CD7'): 2.0, ('CD2', 'CD8'): 10.0, ('CD2', 'CD9'): 0.0, ('CD3', 'CD0'): 0.0, ('CD3', 'CD1'): 1.0, ('CD3', 'CD2'): 0.0, ('CD3', 'CD3'): 0.0, ('CD3', 'CD4'): 4.0, ('CD3', 'CD5'): 0.0, ('CD3', 'CD6'): 0.0, ('CD3', 'CD7'): 0.0, ('CD3', 'CD8'): 0.0, ('CD3', 'CD9'): 5.0, ('CD4', 'CD0'): 0.0, ('CD4', 'CD1'): 0.0, ('CD4', 'CD2'): 6.0, ('CD4', 'CD3'): 0.0, ('CD4', 'CD4'): 0.0, ('CD4', 'CD5'): 0.0, ('CD4', 'CD6'): 0.0, ('CD4', 'CD7'): 0.0, ('CD4', 'CD8'): 0.0, ('CD4', 'CD9'): 0.0, ('CD5', 'CD0'): 3.0, ('CD5', 'CD1'): 0.0, ('CD5', 'CD2'): 5.0, ('CD5', 'CD3'): 0.0, ('CD5', 'CD4'): 7.0, ('CD5', 'CD5'): 0.0, ('CD5', 'CD6'): 0.0, ('CD5', 'CD7'): 0.0, ('CD5', 'CD8'): 0.0, ('CD5', 'CD9'): 0.0, ('CD6', 'CD0'): 0.0, ('CD6', 'CD1'): 0.0, ('CD6', 'CD2'): 0.0, ('CD6', 'CD3'): 0.0, ('CD6', 'CD4'): 2.0, ('CD6', 'CD5'): 0.0, ('CD6', 'CD6'): 0.0, ('CD6', 'CD7'): 0.0, ('CD6', 'CD8'): 3.0, ('CD6', 'CD9'): 6.0, ('CD7', 'CD0'): 10.0, ('CD7', 'CD1'): 8.0, ('CD7', 'CD2'): 0.0, ('CD7', 'CD3'): 0.0, ('CD7', 'CD4'): 0.0, ('CD7', 'CD5'): 3.0, ('CD7', 'CD6'): 0.0, ('CD7', 'CD7'): 0.0, ('CD7', 'CD8'): 10.0, ('CD7', 'CD9'): 0.0, ('CD8', 'CD0'): 0.0, ('CD8', 'CD1'): 8.0, ('CD8', 'CD2'): 0.0, ('CD8', 'CD3'): 0.0, ('CD8', 'CD4'): 0.0, ('CD8', 'CD5'): 0.0, ('CD8', 'CD6'): 0.0, ('CD8', 'CD7'): 3.0, ('CD8', 'CD8'): 0.0, ('CD8', 'CD9'): 9.0, ('CD9', 'CD0'): 0.0, ('CD9', 'CD1'): 8.0, ('CD9', 'CD2'): 2.0, ('CD9', 'CD3'): 0.0, ('CD9', 'CD4'): 0.0, ('CD9', 'CD5'): 0.0, ('CD9', 'CD6'): 0.0, ('CD9', 'CD7'): 1.0, ('CD9', 'CD8'): 10.0, ('CD9', 'CD9'): 0.0}

departures = ['CD0', 'CD1', 'CD2', 'CD3',
              'CD4', 'CD5', 'CD6', 'CD7', 'CD8', 'CD9']
arrivals = ['CD0', 'CD1', 'CD2', 'CD3',
            'CD4', 'CD5', 'CD6', 'CD7', 'CD8', 'CD9']

# freq_2 = []
# for i in departures:
#     for j in arrivals:
#         freq_2.append(dic[(i,j)])

freq_2 = np.load('Database/Results_FamOpt/acft3.npy')
# freq_2 = np.load('Network/acft2.npy')

# print(freq_2)

freq_matrix = freq_2

freq_matrix = freq_matrix.astype(int)

# freq_matrix = np.ones([15, 15])


# # freq_matrix = freq_matrix.astype(int)

# freq_matrix[np.diag_indices_from(freq_matrix)] = 0


print(freq_matrix)



data = pd.read_csv("Database/Airports/airports.csv")
print(data.head())
number_of_airports = len(data['APT'])
# Creamos ciudades de la 0 a la 9
cities = [i for i in range(number_of_airports)]
arcs = [(i, j) for i in cities for j in cities]

lon_coordinates = data.LON
lat_coordinates = data.LAT

x = lon_coordinates
y = lat_coordinates
x = x.values.tolist()
y = y.values.tolist()
names = data.APT


x, y = m(x, y)
m.scatter(x, y, 50, color="orange", marker="o", edgecolor="r", zorder=3)
for i in range(len(names)):
    plt.text(x[i], y[i], names[i], va="baseline", color='k',
             fontsize=12, family="monospace", weight="bold")
for i, j in arcs:
    if freq_matrix[i][j] > 0:
        x1, y1 = m(lon_coordinates[i], lat_coordinates[i])
        x2, y2 = m(lon_coordinates[j], lat_coordinates[j])
        color_value = plt.cm.viridis(freq_matrix[i][j]/10)
        m.drawgreatcircle(lon_coordinates[i], lat_coordinates[i], lon_coordinates[j],
                          lat_coordinates[j], linewidth=freq_matrix[i][j]/2, color=color_value, alpha=0.9)

departure_airports = ["FRA", "LHR", "CDG", "AMS",
                      "MAD", "BCN", "FCO", "DUB", "VIE", "ZRH",'ARN','DME','HEL','IST','KBP']
arrival_airports = ["FRA", "LHR", "CDG", "AMS",
                      "MAD", "BCN", "FCO", "DUB", "VIE", "ZRH",'ARN','DME','HEL','IST','KBP']

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
fig1, ax1 = plt.subplots()
im = ax1.imshow(freq_matrix)
im.set_clim(0, 10)
print(im)
fig.colorbar(im,orientation="vertical", pad=0.2)
# We want to show all ticks...
ax1.set_xticks(np.arange(len(arrival_airports)))
ax1.set_yticks(np.arange(len(departure_airports)))
# ... and label them with the respective list entries
ax1.set_xticklabels(arrival_airports)
ax1.set_yticklabels(departure_airports)

# Loop over data dimensions and create text annotations.
for i in range(len(departure_airports)):
    for j in range(len(arrival_airports)):
        text = ax1.text(j, i, freq_matrix[i, j],
                        ha="center", va="center", color="w")

ax1.xaxis.set_ticks_position('top')

# ax.set_title("Network frequencies for optimum aircraft (112 seats)")
fig.tight_layout()
plt.show()
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# TEST
# =============================================================================
