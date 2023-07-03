import pandas as pd
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

inputs = pd.read_csv("BreastCancer/entradas_breast.csv")
outputs = pd.read_csv("BreastCancer/saidas_breast.csv")


def createNetwork():
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

    return classifier


classifier = KerasClassifier(
    build_fn=createNetwork,
    epochs=100,
    batch_size=10
)

results = cross_val_score(
    estimator=classifier,
    X=inputs, y=outputs,
    cv=10,  # k from k-cross validation
    scoring='accuracy'
)

mean_accuracy = results.mean()
