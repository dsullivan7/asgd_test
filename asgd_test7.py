import numpy as np
import pandas as p
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier


def log_loss(p, y):
    z = p * y
    if z > 18:
        return math.exp(-z)
    if z < -18:
        return -z
    return math.log(1.0 + math.exp(-z))


def main():
    print("loading")
    n_epochs = 20
    avg_start = 250000

    clfs = [(SGDClassifier(loss="log"), [], [], "SGD"),
            (SGDClassifier(average=avg_start, loss="log"), [], [], "ASGD")]

    for n in range(n_epochs):
        reader = p.read_csv("./data/pascal/alpha.txt", sep=" ", iterator=True)
        print("epoch:", n)
        for i in range(100):
            if i % 10 == 0:
                print("    chunk:", i)

            X = np.array(reader.get_chunk(5000))
            y = X[:, 0]
            X = X[:, 1:]
            for idx, row in enumerate(X):
                X[idx] = [np.double(k.split(":")[1]) for k in row]

            for clf, loss, _, _ in clfs:
                if i < 50:
                    clf.partial_fit(X, y, classes=[-1, 1])
                else:
                    pred = clf.decision_function(X)
                    loss.append(np.mean(list(map(log_loss, pred, y))))

        for clf, loss, total_loss, name in clfs:
            total_loss.append(np.mean(loss))
            print(name, total_loss)
            loss[:] = []

    for clf, _, total_loss, name in clfs:
        plt.plot(total_loss, label=name, marker=".")

    plt.xlabel('epoch')
    plt.ylabel('average cost')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()

if __name__ == "__main__":
    main()
