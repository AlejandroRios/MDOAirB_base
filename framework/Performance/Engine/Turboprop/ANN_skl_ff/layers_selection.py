import numpy as np
import seaborn as sns; sns.set_theme()
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neural_network 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

x = np.load("X_data2.npy")
# y1 = np.load("y1_data2.npy")
y2 = np.load("y2_data2.npy")

def layer_selection(hidden_1,hidden_2):

    hidden_1 = int(hidden_1)
    hidden_2 = int(hidden_2)

    x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # scaler2 = StandardScaler()
    # scaler2.fit(y_train)
    # y_test = scaler2.transform(y_test)
    # y_train = scaler2.transform(y_train)

    nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(hidden_1,hidden_2), activation='relu', 
                                        solver='lbfgs', max_iter=1000, learning_rate = 'adaptive', learning_rate_init = 0.01,random_state=11)

    regressormodel=nn_unit.fit(x_train,y_train)

    yp =nn_unit.predict(x_test)

    mse =mean_squared_error(y_test,yp)
    print(mse)

    return float("{:.2f}".format(mse*100))

hidden_1 = np.linspace(10,100,10)
hidden_2 = np.linspace(10,100,10)

hidden_1_labels = [str(x) for x in hidden_1]
hidden_2_labels = [str(x) for x in hidden_2]

try:
    mse_matrix = np.load("mse_data.npy")
except:
    mse_matrix = []

    for i in hidden_1:
        for j in hidden_2:
            mse = layer_selection(i,j)
            mse_matrix.append(mse)

    mse_matrix = np.array(mse_matrix).reshape(10,10)
    np.save("mse_data.npy", mse_matrix)


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(1, 1, 1)



# im = ax1.imshow(mse_matrix,cmap ='viridis',norm=LogNorm())
im = ax1.imshow(mse_matrix,cmap ='viridis')

cb = plt.colorbar(im)

cb.set_label('MSE %')
cb.formatter.set_powerlimits((0, 0))
# We want to show all ticks...
ax1.set_xticks(np.arange(len(hidden_1_labels)))
ax1.set_yticks(np.arange(len(hidden_2_labels)))
# # ... and label them with the respective list entries
ax1.set_xticklabels(hidden_1_labels)
ax1.set_yticklabels(hidden_2_labels)

# # # Loop over data dimensions and create text annotations.
for i in range(len(hidden_1_labels)):
    for j in range(len(hidden_2_labels)):
        text = ax1.text(j, i, round(mse_matrix[i, j],2),
                       ha="center", va="center", color="w")


ax1.xaxis.set_ticks_position('top')
ax1.grid(False)
ax1.set_xlabel('Neurons layer 2')
ax1.set_ylabel('Neurons layer 1')

fig1.tight_layout()
plt.show()