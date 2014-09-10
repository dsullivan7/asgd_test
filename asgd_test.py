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

    alpha = .09

    model_c = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='constant',
                                         eta0=.00001,
                                         alpha=alpha,
                                         fit_intercept=True,
                                         n_iter=1, average=False)

    avg_model_c = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='constant',
                                             eta0=.0006,
                                             alpha=alpha,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    model_o = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='optimal',
                                         alpha=alpha,
                                         fit_intercept=True,
                                         n_iter=1, average=False)

    avg_model_o = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='optimal',
                                             alpha=alpha,
                                             eta0=0.001,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    model_i = linear_model.SGDClassifier(loss='hinge',
                                         learning_rate='invscaling',
                                         eta0=.1,
                                         power_t=.6,
                                         alpha=alpha,
                                         fit_intercept=True,
                                         n_iter=1, average=False)

    avg_model_i = linear_model.SGDClassifier(loss='hinge',
                                             learning_rate='invscaling',
                                             eta0=1.,
                                             power_t=.3,
                                             alpha=alpha,
                                             fit_intercept=True,
                                             n_iter=1, average=True)

    # XXX : please use for loop rather than copy pasting
    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            model_c.partial_fit(x_chunk, y_chunk, classes)
            time2 = time.time()
            times_c.append(time2 - time1)

            # est = np.dot(X, model_c.coef_.T)
            # est += model_c.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_c.append(np.mean(ls))
            est = model_c.score(X_test, y_test)
            pobj_c.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            avg_model_c.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ac.append(time2 - time1)

            # est = np.dot(X, avg_model_c.coef_.T)
            # est += model_c.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ac.append(np.mean(ls))
            est = avg_model_c.score(X_test, y_test)
            pobj_ac.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            model_i.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_i.append(time2 - time1)

            # est = np.dot(X, model_i.coef_.T)
            # est += model_i.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_i.append(np.mean(ls))
            est = model_i.score(X_test, y_test)
            pobj_i.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            avg_model_i.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ai.append(time2 - time1)

            # est = np.dot(X, avg_model_i.coef_.T)
            # est += avg_model_i.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ai.append(np.mean(ls))
            est = avg_model_i.score(X_test, y_test)
            pobj_ai.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            model_o.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_o.append(time2 - time1)

            # est = np.dot(X, model_o.coef_.T)
            # est += model_o.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_o.append(np.mean(ls))
            est = model_o.score(X_test, y_test)
            pobj_o.append(est)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            avg_model_o.partial_fit(x_chunk, y_chunk, classes=classes)
            time2 = time.time()
            times_ao.append(time2 - time1)

            # est = np.dot(X, avg_model_o.coef_.T)
            # est += avg_model_o.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ao.append(np.mean(ls))
            est = avg_model_o.score(X_test, y_test)
            pobj_ao.append(est)

    plt.rc('text', usetex=True)
    plt.plot(times_c, pobj_c, color='red',
             label=r"$eta(t)=eta0, eta0=.00001$")
    plt.plot(times_ac, pobj_ac, color='red',
             linestyle='dashed', label=r'$eta(t) = eta0, eta0=.0006$')
    plt.plot(times_o, pobj_o, color='blue',
             label=r'$eta(t) = 1/(\alpha * t), \alpha=.09$')
    plt.plot(times_ao, pobj_ao, color='blue',
             linestyle='dashed', label=r'$eta(t) = 1/(\alpha * t), \alpha=.0000001$')
    plt.plot(times_i, pobj_i, color='green',
             label=r'$eta(t) = eta0/t^{power\_t}, eta0=.1, power_t=.6$')
    plt.plot(times_ai, pobj_ai, color='green',
             linestyle='dashed', label=r'$eta(t) = eta0/t^{power\_t}, eta0=1, power_t=.3$')
    plt.xlabel('time (seconds)')
    plt.ylabel('score')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()
