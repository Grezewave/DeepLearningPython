import numpy as np


def absoluteError(target, x):
    return sum(abs(target-x))/len(x)


def meanSquarredError(target, x):
    return (1/len(target))*sum((target-x)**2)


def rootMeanSquaredError(target, x):
    return np.sqrt(meanSquarredError(target, x))


def correctRate(target, x):
    return 1 - absoluteError(target, x)


target = np.array([1, 0, 1, 0])
x = np.array([0.3, 0.02, 0.89, 0.32])


print(absoluteError(target, x))
print(meanSquarredError(target, x))
print(rootMeanSquaredError(target, x))
print(correctRate(target, x))
