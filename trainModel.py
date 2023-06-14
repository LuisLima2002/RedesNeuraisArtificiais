import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import d2_absolute_error_score
from tensorflow import keras

# load data
df = pd.read_csv('data.csv')
X = df['X']
Y = df['Y']
""" X = df['X']/60
Y = df['Y']/802 """

# splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# create model
model = keras.Sequential()

model.add(keras.layers.Dense(128, input_dim = 1, activation = 'relu'))
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(1))

# metrics - mse (mean square error) - mae (mean absolute error)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mae'])

# train model
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=6000, verbose=1)

# plot the training loss
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("4-128-64-32-32-relu-trainingloss")
plt.show()

# plot the validation loss
plt.plot(history.history['val_loss'])
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("4-128-64-32-32-relu-validationloss")
plt.show()

# score in the test dataset
Y_pred = model.predict(X_test)
print("Score with test Data: ", d2_absolute_error_score(Y_test, Y_pred))
file = open("4-128-64-32-32-relu-score.txt", "w")
file.write("Score with test Data: " + str(d2_absolute_error_score(Y_test, Y_pred)))

# plot the result in all dataset
Y_pred = model.predict(X).flatten()
""" Y_pred = Y_pred * 802
X = X * 60
Y = Y * 802 """
plt.title("Result")
plt.plot(X, Y, label="Real Function")
plt.plot(X, Y_pred, label="Predicted Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("4-128-64-32-32-relu-result")
plt.show()

# saving model
model.save("4-128-64-32-32-relu")