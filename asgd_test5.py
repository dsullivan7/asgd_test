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

alpha = 1.

clf1 = SGDClassifier(alpha=alpha, n_iter=1, average=True)
clf2 = SGDClassifier(alpha=alpha, n_iter=1)
score_1 = []
score_2 = []

# XXX average = 10 to activate averaging after 10 epochs???

for _ in range(500):
    clf1.partial_fit(X, y, classes=classes)
    clf2.partial_fit(X, y, classes=classes)

    score_1.append(clf1.score(X, y))
    score_2.append(clf2.score(X, y))


plt.plot(score_1, color='red',
         label=r'ASGD $\alpha=%s$' % clf1.alpha)

plt.plot(score_2, color='red',
         linestyle='dashed', label=r'SGD $\alpha=%s$' % clf2.alpha)

plt.legend(loc=0, prop={'size': 11})
plt.show()
