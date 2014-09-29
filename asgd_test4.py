from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
# from sklearn import datasets
import numpy as np
import math
import time
from datetime import datetime
import pandas as p
import scipy.sparse as sp
import matplotlib.pyplot as plt

# XXX : I cannot run it as I don't have the data
# if you want demo on text you should use the 20 newsgroup

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

    train = fetch_20newsgroups(subset='train')
    X = tfv.fit_transform(train.data)
    y = train.target
    X_test = X[:-1000]
    y_test = y[:-1000]
    X = X[-1000:]
    y = y[-1000:]

    x_chunks = [X]  # np.array_split(X, chunks)
    y_chunks = [y]  # np.array_split(y, chunks)

    model_c = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='constant',
                                        eta0=.9,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_c = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='constant',
                                            eta0=2.,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    model_o = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='optimal',
                                        alpha=.0004,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_o = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='optimal',
                                            alpha=.0003,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    model_i = linear_model.SGDRegressor(loss='squared_loss',
                                        learning_rate='invscaling',
                                        eta0=2.,
                                        power_t=.01,
                                        fit_intercept=True,
                                        n_iter=1, average=False)

    avg_model_i = linear_model.SGDRegressor(loss='squared_loss',
                                            learning_rate='invscaling',
                                            eta0=1.,
                                            power_t=.01,
                                            fit_intercept=True,
                                            n_iter=1, average=True)

    models = [
        (model_c, [], [], {'color': 'red', 'label': r"$eta(t)=eta0, eta0=.7$"}),
        (avg_model_c, [], [], {"color": 'red', "linestyle": 'dashed', "label": r'$eta(t) = eta0, eta0=.7$'}),
        (model_o, [], [], {"color": 'blue', "label": r'$eta(t) = 1/(\alpha * t), \alpha=.1$'}),
        (avg_model_o, [], [], {"color": 'blue', "linestyle": 'dashed', "label": r'$eta(t) = 1/(\alpha * t), \alpha=.0000001$'}),
        (model_i, [], [], {"color": 'green', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=2., power_t=.1$'}),
        (avg_model_i, [], [], {"color": 'green', "linestyle": 'dashed', "label": r'$eta(t) = eta0/t^{power\_t}, eta0=2., power_t=.1$'})
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
