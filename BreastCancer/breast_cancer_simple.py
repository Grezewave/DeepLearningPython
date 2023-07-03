import pandas as pd
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix
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
    units=int((30*1)/2),
    activation='relu',
    kernel_initializer='random_uniform',
))

classifier.add(Dense(
    units=1,
    activation='sigmoid'  # critical parameter
))

optimizer = optimizers.Adam(
    learning_rate=0.001,  # higher values increase speed
                          # but can causes missing target
    decay=0.0001,
    clipvalue=0.5
)

classifier.compile(
    optimizer=optimizer,
    # optimizer = 'adam'
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

classifier.fit(
    trainingInputs,
    trainingClasses,
    batch_size=10,  # how many inputs will be reaed at time
    epochs=100  # how many times the weights will be adjusted)
)

weigths = [
    classifier.layers[0].get_weights(),
    classifier.layers[1].get_weights(),
    classifier.layers[2].get_weights()
]

predictions = classifier.predict(testInputs)
predictions = (predictions > 0.5)

accuracy = accuracy_score(testClasses, predictions)
cMatrix = confusion_matrix(testClasses, predictions)

print(predictions)
print(accuracy)
print(cMatrix)


results = classifier.evaluate(testInputs, testClasses)
