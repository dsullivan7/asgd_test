from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import datasets
import numpy as np
import math
import time
from datetime import datetime
import pandas as p
import scipy.sparse as sp
import matplotlib.pyplot as plt

if __name__ == '__main__':

    plt.close('all')
    mms = StandardScaler()

    chunks = 1
    n_iter = 30

    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1,
                          smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    train = np.array(p.read_csv('./data/Train_rev1.csv',
                                skipinitialspace=True,
                                nrows=7000
                                # header=None,
                                #na_values=['<null>']).dropna())
                                ))

    # test = np.array(p.read_csv('./data/parkinsins_test.txt',
    #                         skipinitialspace=True,
    #                         index_col=False,
    #                         header=None,
    #                         na_values=['<null>']).dropna()).ravel()

    X = tfv.fit_transform(train[:, 2])
    y = train[:, 10]
    X_test = X[6000:7000]
    y_test = y[6000:7000]
    X = X[:6000]
    y = y[:6000]

    x_chunks = [X]  # np.array_split(X, chunks)
    y_chunks = [y]  # np.array_split(y, chunks)

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

    model_c = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='constant',
                                        eta0=.7,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_c = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='constant',
                                            eta0=.7,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    # model_o = linear_model.SGDRegressor(loss='squared_loss',
    #                                     learning_rate='pa1',
    #                                     eta0=.00001,
    #                                     fit_intercept=True,
    #                                     n_iter=1, average=False)

    # avg_model_o = linear_model.SGDRegressor(loss='squared_loss',
    #                                         learning_rate='optimal',
    #                                         alpha=1.,
    #                                         fit_intercept=True,
    #                                         n_iter=1, average=True)

    model_i = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='invscaling',
                                        eta0=2.,
                                        power_t=.1,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_i = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='invscaling',
                                            eta0=2.,
                                            power_t=.1,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    time1 = time.time()
    for i in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            # x_chunk = sp.csr_matrix(x_chunk)
            model_c.partial_fit(x_chunk, y_chunk)
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
            avg_model_c.partial_fit(x_chunk, y_chunk)
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
            model_i.partial_fit(x_chunk, y_chunk)
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
            avg_model_i.partial_fit(x_chunk, y_chunk)
            time2 = time.time()
            times_ai.append(time2 - time1)

            # est = np.dot(X, avg_model_i.coef_.T)
            # est += avg_model_i.intercept_
            # ls = list(map(log_loss, est, y))
            # pobj_ai.append(np.mean(ls))
            est = avg_model_i.score(X_test, y_test)
            pobj_ai.append(est)

    # time1 = time.time()
    # for i in range(n_iter):
    #     for x_chunk, y_chunk in zip(x_chunks, y_chunks):
    #         # x_chunk = sp.csr_matrix(x_chunk)
    #         model_o.partial_fit(x_chunk, y_chunk)
    #         time2 = time.time()
    #         times_o.append(time2 - time1)

    #         # est = np.dot(X, model_o.coef_.T)
    #         # est += model_o.intercept_
    #         # ls = list(map(log_loss, est, y))
    #         # pobj_o.append(np.mean(ls))
    #         est = model_o.score(X_test, y_test)
    #         pobj_o.append(est)

    # time1 = time.time()
    # for i in range(n_iter):
    #     for x_chunk, y_chunk in zip(x_chunks, y_chunks):
    #         # x_chunk = sp.csr_matrix(x_chunk)
    #         avg_model_o.partial_fit(x_chunk, y_chunk)
    #         time2 = time.time()
    #         times_ao.append(time2 - time1)

    #         # est = np.dot(X, avg_model_o.coef_.T)
    #         # est += avg_model_o.intercept_
    #         # ls = list(map(log_loss, est, y))
    #         # pobj_ao.append(np.mean(ls))
    #         est = avg_model_o.score(X_test, y_test)
    #         pobj_ao.append(est)

    plt.rc('text', usetex=True)
    plt.plot(times_c, pobj_c, color='red',
             label=r"$eta(t)=eta0, eta0=.7$")
    plt.plot(times_ac, pobj_ac, color='red',
             linestyle='dashed', label=r'$eta(t) = eta0, eta0=.7$')
    # plt.plot(times_o, pobj_o, color='blue',
    #          label=r'$eta(t) = 1/(\alpha * t), \alpha=.1$')
    # plt.plot(times_ao[10:], pobj_ao[10:], color='blue',
    #          linestyle='dashed', label=r'$eta(t) = 1/(\alpha * t), \alpha=.0000001$')
    plt.plot(times_i, pobj_i, color='green',
             label=r'$eta(t) = eta0/t^{power\_t}, eta0=2., power_t=.1$')
    plt.plot(times_ai, pobj_ai, color='green',
             linestyle='dashed', label=r'$eta(t) = eta0/t^{power\_t}, eta0=2., power_t=.1$')
    plt.xlabel('time (seconds)')
    plt.ylabel('score')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()
