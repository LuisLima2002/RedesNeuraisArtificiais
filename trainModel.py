import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras
from keras.utils.vis_utils import plot_model

#load data
df = pd.read_csv('data.csv')
df.head()
X=df['X']
Y=df['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


#create model
model=keras.Sequential()

model.add(keras.layers.Dense(2, input_dim = 1, activation = 'relu'))


model.add(keras.layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mae'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#train model
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=1000,verbose=1)

Y_pred = model.predict(X_test)
print("Score with test Data: ",r2_score(Y_test,Y_pred))

Y_pred = Y_pred.flatten()

a = plt.axes(aspect='equal')
plt.scatter(Y_test, Y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
maxValue =max(int(number) for number in Y_test)
plt.plot([0, maxValue], [0, maxValue])
plt.plot()
plt.show()

Y_pred = model.predict(X).flatten()
plt.plot(X,Y)
plt.plot(X,Y_pred)
plt.show()

answer= input("To sabe yor model type a name: (to exit type 0)")
if answer!='0':
    model.save(answer)