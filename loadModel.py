import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import d2_absolute_error_score
from tensorflow import keras

# receive the model
answer = input("Type the name of the model to load: ")

# load data
df = pd.read_csv('data.csv')
X = df['X']
Y = df['Y']
""" X = df['X']/60
Y = df['Y']/802 """

# load model
model = keras.models.load_model(answer)

# score in all dataset
Y_pred = model.predict(X)
print("Score with test Data: ", d2_absolute_error_score(Y, Y_pred))

# Comparation between Real Y and Predicted Y
Y_pred = Y_pred.flatten()
# a = plt.axes(aspect='equal')
plt.title("Comparation between Real Y and Predicted Y")
plt.scatter(Y, Y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
maxValue = max(int(number) for number in Y)
plt.plot([0, maxValue], [0, maxValue], ls="--")
plt.show()

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
plt.savefig("4-256-128-64-64-relu-result")
plt.show()