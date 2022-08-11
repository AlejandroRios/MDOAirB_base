'''
This script loads an Excel spreadsheet using the pandas package, uses the numpy
package to find trends and plots with the matplotlib package
'''

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pickle as pkl

# 'De (m)' 'BPR' 'FPR' 'OPR' 'TIT (K)' 'T (N)' 'FC (kg/hr)'
df = pd.read_pickle("engines.pkl")
# df =df.reset_index(drop=True)
# df = df.drop('index', 1)

# df = df.reset_index(level=0)

print(df)
print(df.shape[0])




x = 44

# print(df.iloc[[2],['De (m)']])

print(df.loc[x, 'De (m)'])


# df.to_pickle("engines.pkl")  
