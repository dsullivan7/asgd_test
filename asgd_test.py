from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


def loss(p, y):
            return 0.5 * (p - y) * (p - y)

if __name__ == '__main__':

    plt.close('all')

    rng = np.random.RandomState(42)
    n_samples, n_features = 10000, 10

    chunks = 100

    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # Define a ground truth on the scaled data
    y = np.dot(X, w)
    y = np.sign(y)

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    pobj = []
    average_pobj = []

    model = linear_model.SGDRegressor(loss='squared_loss',
                                      learning_rate='constant',
                                      eta0=.01, alpha=0,
                                      fit_intercept=True,
                                      n_iter=1, average=False)

    avg_model = linear_model.SGDRegressor(loss='squared_loss',
                                          learning_rate='constant',
                                          eta0=.01, alpha=0,
                                          fit_intercept=True,
                                          n_iter=1, average=True)

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        model.partial_fit(x_chunk, y_chunk)
        avg_model.partial_fit(x_chunk, y_chunk)

        est = np.dot(X, model.coef_.T)
        est += model.intercept_
        ls = list(map(loss, est, y))
        pobj.append(np.mean(ls))

        est = np.dot(X, avg_model.coef_.T)
        est += avg_model.intercept_
        ls = list(map(loss, est, y))
        average_pobj.append(np.mean(ls))

    plt.plot(average_pobj, label='SGD')
    plt.plot(pobj, label='ASGD')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
