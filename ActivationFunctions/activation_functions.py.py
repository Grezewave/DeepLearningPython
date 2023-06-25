import numpy as np


def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0


def sigmoidFunction(soma):
    return 1/(1 + np.exp(-soma))


def tahnFunction(soma):
    return (
        np.exp(soma) - np.exp(-soma))\
        / (np.exp(soma) + np.exp(-soma)
           )


def relu(soma):
    return max(0, soma)


def linear(soma):
    return soma


def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()


test = [stepFunction(2)]
test.append(sigmoidFunction(2.1))
test.append(tahnFunction(2.1))
test.append(relu(2.1))
test.append(linear(2.1))
test.append(softmax([1, 5, 8, 4]))

print(test)
