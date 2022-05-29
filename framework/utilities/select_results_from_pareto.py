import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1


with open(r"Database/Results_Multi_Optim/functions/case8_profit_pareto.pkl", "rb") as input_file:
    F1_GCD = pickle.load(input_file)
with open(r"Database/Results_Multi_Optim/functions/case8_cost_pareto.pkl", "rb") as input_file:
    F2_GCD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions/functions_multi_obj_DD_profit_cost.pkl", "rb") as input_file:
    df_GCD_all = pickle.load(input_file)


with open(r"Database/Results_Multi_Optim/variables/vars_multi_obj_DD_profit_cost.pkl", "rb") as input_file:
    df_vars = pickle.load(input_file)



df_GCD = pd.DataFrame(
    {'X1': F1_GCD,
     'X2': F2_GCD,})

print(df_GCD_all.head())

print('max prof',np.min(df_GCD['X1']))
print('min cost',np.min(df_GCD['X2']))

print('mean prof',np.mean(df_GCD['X1']))
print('mean cost',np.mean(df_GCD['X2']))



maxprof = np.mean(df_GCD['X1'])
mincost = np.mean(df_GCD['X2'])

input = -1962030

input2 = 70573
print('meanProf',df_GCD_all.iloc[(df_GCD_all['X1']-input).abs().argsort()[:1]])
print('meanCO2 ',df_GCD_all.iloc[(df_GCD_all['X2']-input2).abs().argsort()[:1]])

point_meanProf = df_GCD_all.iloc[(df_GCD_all['X1']-input).abs().argsort()[:1]]
point_meanCO2 = df_GCD_all.iloc[(df_GCD_all['X2']-input2).abs().argsort()[:1]]

# print(point_meanCO2)
# print(len(df_vars))
print('vars mean cO2',df_vars.iloc[570])
# print('vars mean Prof',df_vars.iloc[743])

# print('vars min cO2',df_vars.iloc[1078])
# print('vars max Prof',df_vars.iloc[773])
# F1_DD = pd.DataFrame(F1_DD, columns=['F1'])
# F2_DD = pd.DataFrame(F2_DD, columns=['F2'])

# df_GCD = pd.DataFrame(df_GCD, columns=['F1'])
# F2_GCD = pd.DataFrame(F2_GCD, columns=['F2'])


# print('indexes of Dataframe:')  
# print(df.loc[df['Marks'] == 100])
# print((df.loc[df['Marks'] == 100]) & df.loc[df['Subject'] == 'Physic'))


plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend',fontsize=12) # using a size in points
plt.rc('legend',fontsize='medium') # using a named size
plt.rc('axes',labelsize=12, titlesize=12) # using a size in points

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Profit [US$]')
ax.set_ylabel('CO2 efficiecy [kg/nm]')


# ax.scatter(df_GCD['X1'], df_GCD['X2'],s=30, facecolors='none', edgecolors='grey',alpha = 0.5,label='GCD solutions')
# ax.scatter(df_DD['X1'], df_DD['X2'],s=30, facecolors='none', edgecolors='skyblue',alpha = 0.5,label='DD solutions')
ax.scatter(df_GCD['X1'], df_GCD['X2'], s=50, facecolors='none',marker= '^',edgecolors='black')
# ax.scatter(df_GCD['X1'][22], df_GCD['X2'][22],s=80, marker= 'o',facecolors='none', edgecolors='skyblue',alpha = 1,label='meanCO2')
# ax.scatter(df_GCD['X1'][26], df_GCD['X2'][26],s=80, marker= 'o',facecolors='none', edgecolors='blue',alpha = 1,label='meanProf')
# ax.scatter(F1_DD, F2_DD, s=50, facecolors='none',marker= 's',edgecolors='black')
ax.set_title("Objective Space")

# ax.scatter(df_vars['X1'], df_vars['X3'],s=30, facecolors='none', edgecolors='grey',alpha = 0.5,label='GCD solutions')

plt.legend(loc='upper left')

plt.show()