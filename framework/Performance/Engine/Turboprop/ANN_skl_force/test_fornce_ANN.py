from keras.models import model_from_json
from joblib import dump, load
import numpy as np
# import tensorflow as tf
from framework.Performance.Engine.Turboprop.PW120model import PW120model
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import inverse_transform
from sklearn.model_selection import train_test_split

scaler = load('force_in_scaler.bin') 
scaler2 = load('force_out_scaler.bin')

# load json and create model
json_file = open('force_ANN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("force_ANN.h5")
print("Loaded model from disk")
 

X_data = np.load('X_data.npy')
y_data = np.load('y1_data.npy')
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
generated_inputs = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler2 = StandardScaler()
scaler2.fit(y_train)
y_test = scaler2.transform(y_test)

# evaluate loaded model on test data


test = loaded_model.predict(scaler.transform([(25000,0.1,1)]))
test = scaler2.inverse_transform(test)

print(test)




M0 = np.linspace(0.01,0.7,101)
altitude = np.linspace(0,35000,11)
throttle_position = np.linspace(0.1,1,11)



F_vec = []
fuel_flow_vec = []

# X_data = [[i, j] for i in altitude for j in M0]
X_data = [[i] for i in M0]

X_data = np.asarray(X_data)
# print(X_data[0:,0])

for i in X_data:
    
    test = loaded_model.predict(scaler.transform([(25000, i[0], 1)]))
    test = scaler2.inverse_transform(test)

    # F, fuel_flow = PW120model(25000, i[0], 1)

    F_vec.append(float(test))
    # fuel_flow_vec.append(fuel_flow)

F_vec = np.asarray(F_vec)
y1 = np.asarray(F_vec)
# y2 = np.asarray(fuel_flow_vec)

x = np.load("X_datat.npy")

# x = np.load("X_datat.npy")
# y1 = np.load("y1_datat.npy")
# y2 = np.load("y2_datat.npy")



import matplotlib.pyplot as plt

plt.plot(M0,F_vec)
plt.show()