import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow import keras

answer=input("Type the name of the model to load: ")


df = pd.read_csv('data.csv')
df.head()
X=df['X']
Y=df['Y']
model = keras.models.load_model(answer)
Y_pred = model.predict(X)
print("Score with test Data: ",r2_score(Y,Y_pred))

Y_pred = Y_pred.flatten()

a = plt.axes(aspect='equal')
plt.scatter(Y, Y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
maxValue =max(int(number) for number in Y)
plt.plot([0, maxValue], [0, maxValue])
plt.plot()
plt.show()

Y_pred = model.predict(X).flatten()
plt.plot(X,Y)
plt.plot(X,Y_pred)
plt.show()