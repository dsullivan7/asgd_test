from sklearn import datasets
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

# shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
classes = np.unique(y)

clf1 = SGDClassifier(alpha=0.1, n_iter=1, average=True, eta0=0.001)
clf2 = SGDClassifier(alpha=0.1, n_iter=1, average=True, eta0=0.001)
score_1 = []
score_2 = []

XX = np.r_[X, X]
yy = np.r_[y, y]

clf1.partial_fit(X, y, classes=classes)
clf1.partial_fit(X, y, classes=classes)
# clf2.partial_fit(XX, yy, classes=classes)
clf2.fit(XX, yy)

print clf1.score(X, y)
print clf2.score(X, y)
