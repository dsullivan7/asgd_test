import numpy as np
import pandas as p


def main():
    print("converting alpha.txt to labels and plain data")

    reader = p.read_csv("./data/alpha.txt", iterator=True)
    for i in range(200):

        print("chunk:", i)
        X = np.array(reader.get_chunk(2500))
        y = X[:, 0]
        X = X[:, 1:]
        for idx1, row in enumerate(X):
            for idx2, col in enumerate(row):
                X[idx1, idx2] = np.double(X[idx1, idx2].split(":")[1])

        print("  writing")
        fi = open("data_alpha.txt", "a")
        fil = open("labels_alpha.txt", "a")
        df = p.DataFrame(X)
        dfl = p.DataFrame(y.reshape(-1, 1))
        df.to_csv(fi, header=False, index=False)
        dfl.to_csv(fil, header=False, index=False)

    print("plain data to numpy binaries")
    readerX = p.read_csv("./data_alpha.txt", iterator=True)
    readery = p.read_csv("./labels_alpha.txt", iterator=True)

    for i in range(10):
        print("chunk:", i)
        X = np.array(readerX.get_chunk(50000))
        y = np.array(readery.get_chunk(50000))
        np.save("./leon_data/labels_batch" + str(i) + ".npy", y)
        np.save("./leon_data/data_batch" + str(i) + ".npy", X)


if __name__ == "__main__":
    main()
