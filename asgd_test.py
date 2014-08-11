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

    chunks = 100
    n_iter = 2

    # iris
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # X = X[y < 2]
    # y = y[y < 2]

    # X = np.array(p.read_csv('./data/leon_small_data.csv',
    #                         nrows=5000,
    #                         skipinitialspace=True,
    #                         index_col=False,
    #                         header=None,
    #                         na_values=['<null>']).dropna())

    # y = np.array(p.read_csv('./data/leon_small_label.csv',
    #                         nrows=5000,
    #                         skipinitialspace=True,
    #                         index_col=False,
    #                         header=None,
    #                         na_values=['<null>']).dropna()).ravel()

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

    n_samples, n_features = 1000, 20000
    X = np.eye(n_samples, n_features)
    y = np.random.random_integers(0, 1, n_samples).ravel()
    y[y == 0] = 2

    for i, j in enumerate(y):
        X[i][i] = j
    classes = np.unique(y)

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    X = sp.csr_matrix(X)

    pobj_c = []
    pobj_ac = []
    pobj_o = []
    pobj_ao = []
    pobj_i = []
    pobj_ai = []

    times_c = []
    times_ac = []
    times_o = []
    times_ao = []
    times_i = []
    times_ai = []

    model_c = linear_model.SGDClassifier(loss='log',
                                         learning_rate='constant',
                                         eta0=.0001,
                                         fit_intercept=False,
                                         n_iter=1, average=False)

    avg_model_c = linear_model.SGDClassifier(loss='log',
                                             learning_rate='constant',
                                             eta0=.0006,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    model_o = linear_model.SGDClassifier(loss='log',
                                         learning_rate='optimal',
                                         eta0=.0001,
                                         fit_intercept=True,
                                         n_iter=1, average=False)

    avg_model_o = linear_model.SGDClassifier(loss='log',
                                             learning_rate='optimal',
                                             eta0=.0006,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    model_i = linear_model.SGDClassifier(loss='log',
                                         learning_rate='invscaling',
                                         eta0=.0001,
                                         fit_intercept=True,
                                         n_iter=1, average=False)

    avg_model_i = linear_model.SGDClassifier(loss='log',
                                             learning_rate='invscaling',
                                             eta0=.0006,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            model_c.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_c.append(time2 - time1)

            # est = np.dot(X, model_c.coef_.T)
            # est += model_c.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_c.append(np.mean(ls))
            est = model_c.score(X, y)
            pobj_c.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            avg_model_c.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ac.append(time2 - time1)

            # est = np.dot(X, avg_model_c.coef_.T)
            # est += model_c.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ac.append(np.mean(ls))
            est = avg_model_c.score(X, y)
            pobj_ac.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            model_i.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_i.append(time2 - time1)

            # est = np.dot(X, model_i.coef_.T)
            # est += model_i.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_i.append(np.mean(ls))
            est = model_i.score(X, y)
            pobj_i.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            avg_model_i.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ai.append(time2 - time1)

            # est = np.dot(X, avg_model_i.coef_.T)
            # est += avg_model_i.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ai.append(np.mean(ls))
            est = avg_model_i.score(X, y)
            pobj_ai.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            model_o.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_o.append(time2 - time1)

            # est = np.dot(X, model_o.coef_.T)
            # est += model_o.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_o.append(np.mean(ls))
            est = model_o.score(X, y)
            pobj_o.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            x_chunk = sp.csr_matrix(x_chunk)
            avg_model_o.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ao.append(time2 - time1)

            # est = np.dot(X, avg_model_o.coef_.T)
            # est += avg_model_o.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ao.append(np.mean(ls))
            est = avg_model_o.score(X, y)
            pobj_ao.append(est)

    plt.rc('text', usetex=True)
    plt.plot(times_c, pobj_c, label=r"SGD $f(eta) = eta$")
    plt.plot(times_ac, pobj_ac, label=r'ASGD $f(eta) = eta$')
    plt.plot(times_o, pobj_o, label=r'SGD $f(eta) = 1/(\alpha * t)$')
    plt.plot(times_ao, pobj_ao, label=r'ASGD $f(eta) = 1/(\alpha * t)$')
    plt.plot(times_i, pobj_i, label=r'SGD $f(eta) = eta/t^{power\_t}$')
    plt.plot(times_ai, pobj_ai, label=r'ASGD $f(eta) = eta/t^{power\_t}$')
    plt.xlabel('time (seconds)')
    plt.ylabel('score')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()
