import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

inputs = pd.read_csv("BreastCancer/entradas_breast.csv")
outputs = pd.read_csv("BreastCancer/saidas_breast.csv")

trainingInputs, testInputs, trainingClasses, testClasses = \
    train_test_split(inputs, outputs, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(
    units=int((30*1)/2),
    activation='relu',
    kernel_initializer='random_uniform',
    input_dim=30
))
# units = (number_of_features + number_of_output_neurons)/2
classifier.add(Dense(
    units=1,
    activation='sigmoid'  # critical parameter
))

classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

classifier.fit(
    trainingInputs,
    trainingClasses,
    batch_size=10,  # how many inputs will be reaed at time
    epochs=100  # how many times the weights will be adjusted)
)
