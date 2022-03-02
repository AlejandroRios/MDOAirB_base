import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics


import numpy as np
import matplotlib.pyplot as pl
from sklearn import neural_network 
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



x = np.load("X_data.npy")
y1 = np.load("y1_data.npy")
y2 = np.load("y2_data.npy")

# y1 = np.where(y1<0, 0, y1)
# y2 = np.where(y2<0, 0, y2)



# x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=12)
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=42)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# scaler2 = StandardScaler()
# scaler2.fit(y_train)
# y_test = scaler2.transform(y_test)
# y_train = scaler2.transform(y_train)
# nn_unit = neural_network.MLPRegressor(activation='relu', solver='lbfgs')

nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(90,90), activation='relu', 
                                      solver='lbfgs', max_iter=1000, learning_rate = 'adaptive', learning_rate_init = 0.001,random_state=11)

# 3 20
# 11 20 38
# 12 20

regressormodel=nn_unit.fit(x_train,y_train)

yp =nn_unit.predict(x_test)

# print(nn_unit.predict(scaler.transform([(0.1,32100,0.8)])))

rmse =mean_squared_error(y_test,yp)

#Calculation 10-Fold CV
yp_cv = cross_val_predict(regressormodel, x_test, y_test, cv=10)
rmsecv=np.sqrt(mean_squared_error(y_test,yp_cv))



# plt.figure(0)
# plt.xlabel("#E poca")
# plt.ylabel("Magnitud de perdida")
# plt.plot(regressormodel.history["loss"])


from joblib import dump, load
dump(nn_unit, 'nn_force_PW120.joblib') 

dump(scaler, 'scaler_force_PW120_in.bin', compress=True)
# dump(scaler2, 'scaler_force_PW120_out.bin', compress=True)

print('Method: ReLU')
print('RMSE on the data: %.4f' %rmse)
print('RMSE on 10-fold CV: %.4f' %rmsecv)


n_fig=0
plt.figure(n_fig)
plt.plot(yp, y_test,'ro')
# plt.plot(y_test,'bo')
plt.plot(yp_cv, y_test,'bo', alpha=0.25, label='10-folds CV')
plt.xlabel('predicted')
plt.title('Method: ReLU')
plt.ylabel('real')
plt.grid(True)
plt.show()

