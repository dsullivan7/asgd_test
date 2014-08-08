from sklearn import linear_model
import time
import numpy as np
import scipy.sparse as sp

if __name__ == '__main__':

    n_samples = 5000
    n_features = 20000

    # X = np.zeros([n_samples, n_features])
    # y = np.random.random_integers(0, 1, n_samples).ravel()
    # y[y == 0] = -1
    # X[:, -1] = y
    # X = sp.csr_matrix(X)

    rng = np.random.RandomState(42)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w)
    y = np.sign(y)

    classes = np.unique(y)

    pobj = []
    average_pobj = []

    for i in range(10):
        avg_model = linear_model.SGDClassifier(loss='log',
                                               eta0=.001,
                                               learning_rate='constant',
                                               fit_intercept=True,
                                               n_iter=1, average=True)
        time1 = time.time()
        avg_model.fit(X, y)
        time2 = time.time()
        print("the fit took: " + str(time2 - time1) + " seconds")
