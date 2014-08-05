from sklearn import linear_model
# from sklearn import datasets
import numpy as np
import math
import pandas as p
import matplotlib.pyplot as plt


def sq_loss(p, y):
    return 0.5 * (p - y) * (p - y)


def hinge_loss(p, y):
    z = p * y
    if z <= 1.0:
        return (1.0 - z)
    return 0.0


def log_loss(p, y):
    z = p * y
    if z > 18:
        return math.exp(-z)
    if z < -18:
        return -z
    return math.log(1.0 + math.exp(-z))

if __name__ == '__main__':

    plt.close('all')

    chunks = 1
    n_iter = 20

    # iris
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # X = X[y < 2]
    # y = y[y < 2]

    X = np.array(p.read_csv('./data/leon_small_data.csv',
                            nrows=5000,
                            skipinitialspace=True,
                            index_col=False,
                            header=None,
                            na_values=['<null>']).dropna())

    y = np.array(p.read_csv('./data/leon_small_label.csv',
                            nrows=5000,
                            skipinitialspace=True,
                            index_col=False,
                            header=None,
                            na_values=['<null>']).dropna()).ravel()

    # random
    # rng = np.random.RandomState(42)
    # n_samples, n_features = 1000, 100
    # X = rng.normal(size=(n_samples, n_features))
    # w = rng.normal(size=n_features)
    # y = np.dot(X, w)
    # y = np.sign(y)

    classes = np.unique(y)

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    pobj = []
    average_pobj = []

    model = linear_model.SGDClassifier(loss='log',
                                       learning_rate='constant',
                                       eta0=.0001,
                                       fit_intercept=True,
                                       n_iter=1, average=False)

    avg_model = linear_model.SGDClassifier(loss='log',
                                           learning_rate='constant',
                                           eta0=.0006,
                                           fit_intercept=True,
                                           n_iter=1, average=True)

    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            model.partial_fit(x_chunk, y_chunk, classes=classes)
            avg_model.partial_fit(x_chunk, y_chunk, classes=classes)

            est = np.dot(X, model.coef_.T)
            est += model.intercept_
            ls = list(map(log_loss, est, y))
            pobj.append(np.mean(ls))

            est = np.dot(X, avg_model.coef_.T)
            est += avg_model.intercept_
            ls = list(map(log_loss, est, y))
            average_pobj.append(np.mean(ls))

    plt.plot(average_pobj, label='ASGD')
    plt.plot(pobj, label='SGD')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
