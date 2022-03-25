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

from matplotlib.offsetbox import AnchoredText


def poly2latex(poly, variable="x", width=2):

    t = ["{0:0.{width}e}"]
    t.append(t[-1] + " {variable}")
    t.append(t[-1] + "^{1}")

    def f():
        for i, v in enumerate(reversed(poly)):
            idx = i if i < 2 else 2
            yield t[idx].format(v, i, variable=variable, width=width)

    return "${}$".format("+".join(f()))

def polyfit(x, y, degree):
    '''
    Description: This function calculates parameters from regression

    Inputs: 
        - x: x data
        - y; y data
        - degree: polyinomial degree interval [1-3]

    Outputs:
        - dictionary containing results (coefficients and R^2 value)
    '''

    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x) # or [p(z) for z in x]
    ybar = np.sum(y)/len(y) # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2) # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2) # or sum([ (yi - ybar)**2 for yi in y])
    results['R'] = ssreg / sstot

    return results


x = np.load("Performance/Engine/Turboprop/X_data2.npy")
y1 = np.load("Performance/Engine/Turboprop/y1_data2.npy")
y2 = np.load("Performance/Engine/Turboprop/y2_data2.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=42)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# scaler2 = StandardScaler()
# scaler2.fit(y_train)
# y_test = scaler2.transform(y_test)
# y_train = scaler2.transform(y_train)


nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(30,70), activation='relu', 
                                      solver='lbfgs', max_iter=1000, learning_rate = 'adaptive', learning_rate_init = 0.01,random_state=11)

regressormodel=nn_unit.fit(x_train,y_train)

yp =nn_unit.predict(x_test)

# print(nn_unit.predict(scaler.transform([(0.1,32100,0.8)])))

rmse =mean_squared_error(y_test,yp)

#Calculation 10-Fold CV
yp_cv = cross_val_predict(regressormodel, x_test, y_test, cv=10)
rmsecv=mean_squared_error(y_test,yp_cv)


y_test = y_test.flatten()

x_new = np.linspace(min(y_test),max(y_test),50)
p = np.polyfit(yp, y_test, 1)
y_est = p[0]*x_new + p[1]
results = polyfit(yp, y_test, 1)



plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)

anchored_text = AnchoredText('$R^2$: {:.4f} \n'.format(results['R']) +  '$MSE $: {:.2f}'.format(rmse*100) + "%"  , loc=4)
ax1.add_artist(anchored_text)

from joblib import dump, load
dump(nn_unit, 'Performance/Engine/Turboprop/ANN_skl_ff/nn_ff_PW120.joblib') 

dump(scaler, 'Performance/Engine/Turboprop/ANN_skl_ff/scaler_ff_PW120_in.bin', compress=True)
# dump(scaler2, 'scaler_force_PW120_out.bin', compress=True)

print('Method: ReLU')
print('RMSE on the data: %.4f' %rmse)
print('RMSE on 10-fold CV: %.4f' %rmsecv)



ax1.plot(yp, y_test,'bo',alpha=0.5)
ax1.plot(x_new, y_est, 'k-',label='linear regression',linewidth=2)

ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Activation function: ReLU')

ax1.set_xlim([0,600])
ax1.set_ylim([0,600])

plt.grid(True)
plt.show()

