import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X = X[y < 2]
y = y[y < 2]

classifiers = [
    ("SGD", SGDClassifier(n_iter=1, alpha=.1)),
    ("ASGD", SGDClassifier(average=True, eta0=.001, n_iter=1)),
]


class SquaredLoss():
    """Squared loss traditional used in linear regression."""
    def loss(self, p, y):
        return 0.5 * (p - y) * (p - y)


class Hinge():
    def loss(self, p, y):
            z = p * y
            if z <= 1.0:
                return 1.0 - z
            return 0.0

loss = Hinge()

chunks = 3
n_iter = 30
X_train = X
y_train = y
classes = np.unique(y)
for name, clf in classifiers:

    # X_train, X_test, y_train, y_test = train_test_split(digits.data,
    #                                                     digits.target,
    #                                                     test_size=.10)


    x_chunks = np.array_split(X_train, chunks)
    y_chunks = np.array_split(y_train, chunks)
    yy = []

    for j in range(n_iter):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            clf.partial_fit(x_chunk, y_chunk, classes=classes)
            y_pred = clf.decision_function(X_train).ravel()
            # yy.append(1 - np.mean(y_pred == y_train))
            avg_score = np.mean(list(map(loss.loss, y_pred, y_train)))
            avg_score += clf.alpha * np.linalg.norm(clf.coef_)
            yy.append(avg_score)
    plt.plot(yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("iteration")
plt.ylabel("Average cost")
plt.show()
