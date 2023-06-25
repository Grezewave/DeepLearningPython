import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

inputs = pd.read_csv("entradas_breast.csv")
outputs = pd.read_csv("saidas_breast.csv")

trainingInputs, testInputs, trainingClasses, testClasses = \
    train_test_split(inputs, outputs, test_size=0.25)

classifier = Sequential()
classifier.add(Dense())
