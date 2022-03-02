import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as pl
from sklearn import neural_network 
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

x = np.load("X_data2.npy")
y2 = np.load("y2_data2.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=42)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


scaler2 = StandardScaler()
scaler2.fit(y_train)
y_test = scaler2.transform(y_test)
y_train = scaler2.transform(y_train)


nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(90,30), activation='relu', 
                                      solver='lbfgs', max_iter=1000, learning_rate = 'adaptive', learning_rate_init = 0.01,random_state=11)

regressormodel=nn_unit.fit(x_train,y_train)

yp =nn_unit.predict(x_test)

# print(nn_unit.predict(scaler.transform([(0.1,32100,0.8)])))

rmse =mean_squared_error(y_test,yp)

#Calculation 10-Fold CV
yp_cv = cross_val_predict(regressormodel, x_test, y_test, cv=10)
rmsecv=mean_squared_error(y_test,yp_cv)



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
plt.plot(yp, y_test,'bo')
# plt.plot(y_test,'bo')
# plt.plot(yp_cv, y_test,'bo', alpha=0.25, label='10-folds CV')
plt.xlabel('predicted')
plt.title('Method: ReLU')
plt.ylabel('real')
plt.grid(True)
plt.show()

