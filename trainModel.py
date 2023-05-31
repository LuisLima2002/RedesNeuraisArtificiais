import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras
#todo: refazer sigmoid, refazer uma linear  
#load data
df = pd.read_csv('data.csv')
df.head()
X=df['X']/60
Y=df['Y']/802

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#create model
model=keras.Sequential()

model.add(keras.layers.Dense(256, input_dim = 1, activation = 'sigmoid'))
model.add(keras.layers.Dense(128, activation = 'sigmoid'))
model.add(keras.layers.Dense(64, activation = 'sigmoid'))
model.add(keras.layers.Dense(64, activation = 'sigmoid'))


model.add(keras.layers.Dense(1))
#metrics - mse (mean square error)  -  mae (mean absolute error)
#optmaier
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse','mae'])
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#train model
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=1500,verbose=1)

plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.savefig("4-256-128-64-64-sigmoid")
plt.show()
Y_pred = model.predict(X_test)
print("Score with test Data: ",r2_score(Y_test,Y_pred))

#Y_pred = Y_pred.flatten()

#maxValue =max(int(number) for number in Y_test)
#minValue =min(int(number) for number in Y_test)
#a = plt.axes(aspect='equal')
#plt.scatter(Y_test, Y_pred)
#plt.xlabel('True values')
#plt.ylabel('Predicted values')
#plt.plot([minValue, maxValue], [minValue, maxValue])
#plt.plot()
#plt.show()
Y_pred = model.predict(X).flatten()
Y_pred = Y_pred*802
X=X*60
Y=Y*802
plt.plot(X,Y)
plt.plot(X,Y_pred)
plt.show()

model.save("4-256-128-64-64-sigmoid")