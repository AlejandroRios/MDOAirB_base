'''

'''
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import inverse_transform
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------
# Load data and
x = np.load("Performance/Engine/Turboprop/X_data2.npy")
y1 = np.load("Performance/Engine/Turboprop/y1_data2.npy")


x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# y_test = np.reshape(y_test, (len(y_test), 1))
# generated_target = np.reshape(y_train, (len(y_train), 1))


# np.savez('TF_train', input=generated_inputs, targets=generated_target)

# -----------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
# training_data = np.load('TF_train.npz')




input_size=2
output_size=1


hidden1 = tf.keras.layers.Dense(units=90, input_shape=[3], activation = 'relu')
hidden2 = tf.keras.layers.Dense(units=100, activation = 'relu')
hidden3 = tf.keras.layers.Dense(units=50, activation = 'relu')
hidden4 = tf.keras.layers.Dense(units=10, activation = 'relu')
hidden5 = tf.keras.layers.Dense(units=10, activation = 'relu')
salida = tf.keras.layers.Dense(units=1)
models = tf.keras.Sequential([hidden1,hidden2,salida])
models.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss = 'mean_squared_error')

regressormodel = models.fit(x_train,y_train,epochs=1000,verbose=False)

yp =models.predict(x_test)

# print(models.predict(scaler.transform([(0,400000)])))


rmsecv=mean_squared_error(y_test,yp)

print('Method: ReLU')
print('RMSE on the data: %.4f' %rmsecv)


plt.figure(0)
plt.xlabel("#E poca")
plt.ylabel("Magnitud de perdida")
plt.plot(regressormodel.history["loss"])



plt.figure(1)
plt.plot(yp, y_test,'ro')
# plt.plot(y_test,'bo')
plt.xlabel('predicted')
plt.title('Method: ReLU')
plt.ylabel('real')
plt.grid(True)
plt.show()


# print(scaler2.inverse_transform(y_test))

# model_json = models.to_json()
# with open("force_ANN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# models.save_weights("force_ANN.h5")
# print("Saved model to disk")


# from joblib import dump, load
# dump(scaler, 'force_in_scaler.bin', compress=True)
# dump(scaler2, 'force_out_scaler.bin', compress=True)