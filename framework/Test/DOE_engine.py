'''

'''

from framework.Database.Aircrafts.baseline_aircraft_parameters import initialize_aircraft_parameters
from framework.Performance.Engine.engine_performance import turbofan

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymoo.factory import get_sampling
from pymoo.interface import sample
from aux_tools import corrdot
import pickle
#=========================================

# SETUP



# Give number of input variables
vehicle = initialize_aircraft_parameters()
engine = vehicle['engine']

# Give number of input variables
n_inputs = 5

# Lower and upeer bounds of each input variable
#    D   | BPR | FPR | OPR | TiT
lb = [1.2,   1,  1.4,   12, 1100]
ub = [3.0,   9,  2.0,   40, 2000]

# lb = [1.3, 4, 1, 18, 1200]
# ub = [2, 9 , 2, 32, 2000]


# Desired number of samples
n_samples = 50

# Sampling type
#sampling_type = 'real_random'
sampling_type = 'real_lhs'

# Plot type (0-simple, 1-complete)
plot_type = 1
#=========================================

# EXECUTION

# Set random seed to make results repeatable
np.random.seed(123)

# Initialize sampler
sampling = get_sampling(sampling_type)

# Draw samples
X = sample(sampling, n_samples, n_inputs)

# Samples are originally between 0 and 1,
# so we need to scale them to the desired interval
for ii in range(n_inputs):
    X[:,ii] = lb[ii] + (ub[ii] - lb[ii])*X[:,ii]

# Execute all cases and store outputs
y1_samples = []
y2_samples = []
for ii in range(n_samples):

    # Evaluate sample

    engine['fan_pressure_ratio'] = X[ii,2]
    engine['compressor_pressure_ratio'] = X[ii,3]
    engine['bypass'] = X[ii,1]
    engine['fan_diameter'] = X[ii,0]
    engine['turbine_inlet_temperature'] = X[ii,4]

    engine_thrust, ff , vehicle = turbofan(0, 0.1 , 1, vehicle)

    # Store the relevant information
    y1_samples.append(float(engine_thrust))
    y2_samples.append(float(ff))
import winsound
duration = 1000*60  # milliseconds
freq = 1000 # Hz
winsound.Beep(freq, duration)
# Create a pandas dataframe with all the information
df = pd.DataFrame({'De (m)' : X[:,0],
                   'BPR' : X[:,1],
                   'FPR' : X[:,2],
                   'OPR' : X[:,3],
                   'TIT (K)' : X[:,4],
                   'T (N)' : y1_samples,
                   'FC (kg/hr)' : y2_samples})


df.to_pickle("engines3.pkl")  
print(df.head())
# Plot the correlation matrix
# fig = plt.subplots(figsize=(4, 4))
sns.set(style='white', font_scale=1)

if plot_type == 0:

    # Simple plot
    ax = sns.pairplot(df,corner=True)

elif plot_type == 1:


    # Complete plot
    # based on: https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
    ax = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    ax.map_lower(sns.regplot, lowess=True, line_kws={'color': 'black'})
    ax.map_diag(sns.histplot)
    ax.map_upper(corrdot)

    for ax in ax.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.22,0.5)

# Plot window
plt.tight_layout()
plt.show()

# plt.savefig('doe.pdf', dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=False, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)