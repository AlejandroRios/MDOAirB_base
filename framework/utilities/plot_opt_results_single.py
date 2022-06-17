import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
with open(r"Database/Results_Multi_Optim/functions/case5_profit_pareto.pkl", "rb") as input_file:
    F1_GCD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions/case5_C02_pareto.pkl", "rb") as input_file:
    F2_GCD = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/functions/functions_multi_obj_GCD_profit_CO2.pkl", "rb") as input_file:
    df_GCD_all = pickle.load(input_file)

with open(r"Database/Results_Multi_Optim/variables/vars_multi_obj_GCD_profit_CO2.pkl", "rb") as input_file:
    df_vars = pickle.load(input_file)

print(F1_GCD)

df_GCD = pd.DataFrame(
    {'X1': F1_GCD,
     'X2': F2_GCD,})


print(df_GCD_all)
# print(df_GCD_all.head())

# print('max prof',np.min(df_GCD['X1']))
# print('min cost',np.min(df_GCD['X2']))

# print('mean prof',np.mean(df_GCD['X1']))
# print('mean cost',np.mean(df_GCD['X2']))



maxprofit= np.min(df_GCD['X1'])
mainC02 = np.min(df_GCD['X2'])


# print(mainC02 ,maxprofit)


input1 = maxprofit
input2 =mainC02 
point_maxProf = df_GCD_all.iloc[(df_GCD_all['X1']-input1).abs().argsort()[:1]]
point_minCO2 = df_GCD_all.iloc[(df_GCD_all['X2']-input2).abs().argsort()[:1]]
print('minCO2',df_GCD_all.iloc[(df_GCD_all['X2']-input2).abs().argsort()[:3]])
# print('maxProf ',df_GCD_all.iloc[(df_GCD_all['X2']-input2).abs().argsort()[:1]])

print('max profit',point_maxProf)

print('aquiii',)
input3 = np.mean(df_GCD['X1'])
print('meanProfit value',input3)
input4 = np.mean(df_GCD['X2'])
point_meanProf = df_GCD.iloc[(df_GCD['X1']-input3).abs().argsort()[:1]]
point_meanCO2 = df_GCD.iloc[(df_GCD['X2']-input4).abs().argsort()[:1]]



point_meanProf = df_GCD_all.iloc[(df_GCD_all['X1']-(-1758594)).abs().argsort()[:10]]

print('meanProfit',point_meanProf)

# print(point_meanCO2)
# print(len(df_vars))
print('vars mean cO2',df_vars.iloc[944])
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


ax.scatter(df_GCD_all['X1'], df_GCD_all['X2'],s=20, facecolors='none', edgecolors='red',alpha = 0.5,label='GCD solutions')
# ax.scatter(df_DD['X1'], df_DD['X2'],s=30, facecolors='none', edgecolors='skyblue',alpha = 0.5,label='DD solutions')
ax.scatter(F1_GCD, F2_GCD, s=50, facecolors='none',marker= '^',edgecolors='blue',label='Pareto solutions')
ax.scatter(point_maxProf._values[0][0],point_maxProf._values[0][1], s=100, facecolors='none',marker= 's',edgecolors='black',label='Pareto extreme solutions')
ax.scatter(point_minCO2._values[0][0],point_minCO2._values[0][1], s=100, facecolors='none',marker= 's',edgecolors='black')
# ax.scatter(point_meanProf._values[0][0],point_meanProf._values[0][1], s=100, facecolors='none',marker= 's',edgecolors='yellow',label='meanProf')
ax.scatter(point_meanCO2._values[0][0],point_meanCO2._values[0][1], s=100, facecolors='none',marker= 's',edgecolors='yellow',label='Pareto mean CO2 solution')

ax.set_title("Objective Space")

# ax.scatter(df_vars['X1'], df_vars['X3'],s=30, facecolors='none', edgecolors='grey',alpha = 0.5,label='GCD solutions')

plt.legend(loc='upper left')

plt.show()