from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
# from sklearn import datasets
import numpy as np
import math
import time
from datetime import datetime
import pandas as p
import scipy.sparse as sp
import matplotlib.pyplot as plt

# XXX : I cannot run it as I don't have the data

if __name__ == '__main__':

    plt.close('all')
    mms = MinMaxScaler()

    chunks = 1
    n_iter = 20

    train = np.array(p.read_csv('./data/compactiv.dat',
                            skipinitialspace=True,
                            index_col=False,
                            header=None,
                            na_values=['<null>']).dropna())

    # test = np.array(p.read_csv('./data/parkinsins_test.txt',
    #                         skipinitialspace=True,
    #                         index_col=False,
    #                         header=None,
    #                         na_values=['<null>']).dropna()).ravel()

    X = train[:, :21]
    X = mms.fit_transform(X)
    y = train[:, 21]

    X_test = X[7000:]
    y_test = y[7000:]
    X = X[:7000]
    y = y[:7000]

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)


    model_c = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='constant',
                                        eta0=.2,
                                        fit_intercept=False,
                                        n_iter=1, average=False)

    avg_model_c = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='constant',
                                            eta0=.2,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    model_o = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='optimal',
                                        alpha=.03,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_o = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='optimal',
                                            alpha=.03,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    model_i = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='invscaling',
                                        eta0=2.5,
                                        power_t=.6,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_i = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='invscaling',
                                            eta0=2.5,
                                            power_t=.6,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    models = [
        (model_c, [], [], {'color': 'red', 'label': r"$eta(t)=eta0, eta0=.2$"}),
        (avg_model_c, [], [], {"color": 'red', "linestyle": 'dashed', "label": r'$eta(t) = eta0, eta0=.2$'}),
        (model_o, [], [], {"color": 'blue', "label": r'$eta(t) = 1/(\alpha * t), \alpha=.03$'}),
        (avg_model_o, [], [], {"color": 'blue', "linestyle": 'dashed', "label": r'$eta(t) = 1/(\alpha * t), \alpha=.03$'}),
        (model_i, [], [], {"color": 'green', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=2.5, power_t=.6$'}),
        (avg_model_i, [], [], {"color": 'green', "linestyle": 'dashed', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=2.5, power_t=.6$'})
    ]

    for model in models:
        time1 = time.time()
        for i in range(n_iter):
            for x_chunk, y_chunk in zip(x_chunks, y_chunks):
                model[0].partial_fit(x_chunk, y_chunk)
                time2 = time.time()
                model[1].append(time2 - time1)
                est = model[0].score(X_test, y_test)
                model[2].append(est)

        plt.plot(model[1], model[2], **model[3])

    plt.rc('text', usetex=True)
    plt.xlabel('time (seconds)')
    plt.ylabel('score')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()
