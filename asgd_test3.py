import timeit

if __name__ == '__main__':

    setup1 = '''
from sklearn import linear_model
import numpy as np
import pandas as p

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
classes = np.unique(y)

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
'''
    setup2 = '''
from sklearn import linear_model
import numpy as np
import pandas as p

# random
rng = np.random.RandomState(42)
n_samples, n_features = 10000, 20000
X1 = rng.normal(size=(n_samples, n_features))
w = rng.normal(size=n_features)
y1 = np.dot(X1, w)
y1 = np.sign(y1)

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
    '''

    setup3 = '''
from sklearn import linear_model
import numpy as np
import pandas as p
import scipy.sparse as sp

# random
n_samples, n_features = 10000, 20000
X2 = np.zeros([n_samples, n_features])
y2 = np.random.random_integers(0, 1, n_samples).ravel()
y2[y2 == 0] = -1
X2[:, -1] = y2
X2 = sp.csr_matrix(X2)

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
'''

    print("sgd leon data set:" + str(min(timeit.Timer("model.fit(X, y)",
          setup=setup1).repeat(10, 1))) + " seconds")
    print("asgd leon dataset:" + str(min(timeit.Timer("avg_model.fit(X, y)",
          setup=setup1).repeat(10, 1))) + " seconds")
    print("sgd dense matrix with 10000 samples of 20000 features : " +
          str(min(timeit.Timer("model.fit(X1, y1)",
              setup=setup2).repeat(10, 1))) +
          " seconds")
    print("asgd dense matrix with 10000 samples of 20000 features : " +
          str(min(timeit.Timer("avg_model.fit(X1, y1)",
              setup=setup2).repeat(10, 1))) +
          " seconds")
    print("sgd sparse matrix with 10000 samples of 20000 features : " +
          str(min(timeit.Timer("model.fit(X2, y2)",
                  setup=setup3).repeat(10, 1))) +
          " seconds")
    print("asgd sparse matrix with 10000 samples of 20000 features : " +
          str(min(timeit.Timer("avg_model.fit(X2, y2)",
                  setup=setup3).repeat(10, 1))) +
          " seconds")
