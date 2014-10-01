from sklearn import linear_model
# from sklearn import datasets
import numpy as np
import math
import time
from datetime import datetime
import pandas as p
import scipy.sparse as sp
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

    X_test = X[4000:]
    y_test = y[4000:]
    X = X[:4000]
    y = y[:4000]

    # random
    # rng = np.random.RandomState(42)
    # n_samples, n_features = 1000, 100
    # X = rng.normal(size=(n_samples, n_features))
    # w = rng.normal(size=n_features)
    # y = np.dot(X, w)
    # y = np.sign(y)

    # random
    # rng = np.random.RandomState(42)
    # n_samples, n_features = 1000, 30000
    # X = rng.normal(size=(n_samples, n_features))
    # w = rng.normal(size=n_features)
    # y = np.dot(X, w)
    # y = np.sign(y)

    # n_samples, n_features = 2000, 20000
    # rng = np.random.RandomState(42)
    # X = np.eye(n_samples, n_features)
    # w = rng.normal(size=n_features)
    # y = np.dot(X, w)
    # y = np.sign(y).ravel()
    # y[y == -1] = 2

    # X_test = X[500:]
    # y_test = y[500:]
    # X = X[:1500]
    # y = y[:1500]
    # X = sp.csr_matrix(X)

    classes = np.unique(y)

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    # alpha = .09
    alpha = 1.
    fit_intercept = True

    model_c = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='constant',
                                         eta0=.00001,
                                         alpha=alpha,
                                         fit_intercept=fit_intercept,
                                         n_iter=1, average=False)

    avg_model_c = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='constant',
                                             eta0=.0006,
                                             alpha=alpha,
                                             fit_intercept=fit_intercept,
                                             n_iter=1, average=True)

    model_o = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='optimal',
                                         alpha=alpha,
                                         fit_intercept=fit_intercept,
                                         n_iter=1, average=False)

    avg_model_o = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='optimal',
                                             alpha=alpha,
                                             fit_intercept=fit_intercept,
                                             n_iter=1, average=True)

    model_i = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='invscaling',
                                         eta0=.1,
                                         power_t=.6,
                                         alpha=alpha,
                                         fit_intercept=fit_intercept,
                                         n_iter=1, average=False)

    avg_model_i = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='invscaling',
                                             eta0=.005,
                                             power_t=.3,
                                             alpha=alpha,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    models = [
        (model_c, [], [], {'color': 'red', 'label': r"$eta(t)=.00001$"}),
        (avg_model_c, [], [], {"color": 'red', "linestyle": 'dashed', "label": r'$eta(t)=.0006$'}),
        (model_o, [], [], {"color": 'blue', "label": r'$eta(t) = 1/(\alpha * t)$'}),
        (avg_model_o, [], [], {"color": 'blue', "linestyle": 'dashed', "label": r'$eta(t) = 1/(\alpha * t)$'}),
        (model_i, [], [], {"color": 'green', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=.1, power_t=.6$'}),
        (avg_model_i, [], [], {"color": 'green', "linestyle": 'dashed', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=.005, power_t=.3$'})
    ]

    for clf, timing, scores, plot_params in models:
        time1 = time.time()
        for i in range(n_iter):
            for x_chunk, y_chunk in zip(x_chunks, y_chunks):
                clf.partial_fit(x_chunk, y_chunk, classes)
                timing.append(time.time() - time1)
                scores.append(clf.score(X_test, y_test))

        plt.plot(timing, scores, **plot_params)


    plt.rc('text', usetex=True)
    plt.xlabel('time (seconds)')
    plt.ylabel('score')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()
