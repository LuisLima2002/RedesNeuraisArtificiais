from tensorflow import keras

# receive the model
answer = input("Type the name of the model to load: ")

# receive the x value to predict
x = int(input("Type the value of X to predict: "))

# load the model
model = keras.models.load_model(answer)

# predict and print the x value requested
Y_pred = model.predict([x])
print("The predict value is ", Y_pred[0][0])