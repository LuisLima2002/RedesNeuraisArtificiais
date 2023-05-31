from tensorflow import keras

x = int(input("Type the value of X to predict: "))

model = keras.models.load_model("4-126-64-32-32-relu3")

Y_pred = model.predict([x])

print("The predict value is ", Y_pred[0][0])