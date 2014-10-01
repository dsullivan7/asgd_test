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

n_iter = 1000  # XXX looks like it has no influence ... I don't get it.

alpha = .001

classifiers = [
    ("SGD", SGDClassifier(n_iter=n_iter, alpha=alpha)),
    ("ASGD", SGDClassifier(n_iter=n_iter, alpha=alpha, average=True)),
    ("ASGD2", SGDClassifier(n_iter=n_iter, alpha=alpha, average=True, learning_rate='constant', eta0=.001)),
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
n_epochs = 30
X_train = X
y_train = y
classes = np.unique(y)

# X_train, X_test, y_train, y_test = train_test_split(digits.data,
#                                                     digits.target,
#                                                     test_size=.10)

for name, clf in classifiers:
    x_chunks = np.array_split(X_train, chunks)
    y_chunks = np.array_split(y_train, chunks)
    yy = []

    for j in range(n_epochs):
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            clf.partial_fit(x_chunk, y_chunk, classes=classes)
            df = clf.decision_function(X_train).ravel()
            avg_score = np.mean(list(map(loss.loss, df, y_train)))
            avg_score += clf.alpha * np.linalg.norm(clf.coef_) ** 2
            yy.append(avg_score)
    plt.plot(yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("iteration")
plt.ylabel("Average cost")
plt.show()
