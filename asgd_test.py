from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':


    plt.close('all')

    eta = .001
    rng = np.random.RandomState(42)
    n_samples, n_features = 1000, 10

    chunks = 10

    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # Define a ground truth on the scaled data
    y = np.dot(X, w)
    y = np.sign(y)

    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    model = linear_model.SGDClassifier(loss='squared_loss',
                                       learning_rate='constant',
                                       eta0=eta, alpha=0,
                                       fit_intercept=True,
                                       n_iter=1, average=False)

    avg_model = linear_model.SGDClassifier(loss='squared_loss',
                                           learning_rate='constant',
                                           eta0=eta, alpha=0,
                                           fit_intercept=True,
                                           n_iter=1, average=True)


    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        model.fit(x_chunk, y_chunk)
        avg_model.fit(x_chunk, y_chunk)

    print(model.pobj_)
    plt.plot(model.pobj_, label='SGD')
    plt.plot(avg_model.pobj_, label='ASGD')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
