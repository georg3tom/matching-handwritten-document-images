"""
https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6
"""

import faiss
import numpy as np
import cv2


class L2ExactIndex:
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype("float32")
        self.labels = labels

    def build(self):
        self.index = faiss.IndexFlatL2(
            self.dimension,
        )
        if not self.index.is_trained:
            self.index.tran(data)
        self.index.add(self.vectors)

    def query(self, vectors, k=6):
        k = int(min(k, self.labels.shape[0]))
        vectors = vectors.astype("float32")
        distances, indices = self.index.search(vectors, k)
        return distances, self.labels[np.array(indices)]

    def score(self, vectors, labels, k=3):
        _, pred = self.query(vectors)
        k = min(k, pred.shape[0])
        pred = np.char.split(pred, sep="_")
        labels = np.char.split(labels, sep="_")
        acc = 0
        for i in range(pred.shape[0]):
            x = 0
            for j in range(k):
                if pred[i][j][0].split(".")[0] == labels[i][0].split(".")[0]:
                    x += 1
            x /= k
            acc += x
        acc /= pred.shape[0]
        return acc

    def write(self, indexName="./index", labelName="./labels"):
        faiss.write_index(self.index, indexName)
        np.save(labelName, self.labels)

    def load_index(self, indexName="./index"):
        self.index = self.read_index(indexName)


if __name__ == "__main__":
    d = 64
    nb = 100
    nq = 10
    np.random.seed(1234)
    labels = np.array([i for i in range(nb)])
    xb = np.random.random((nb, d)).astype("float32")
    xb[:, 0] += np.arange(nb) / 1000.0
    xq = np.random.random((nq, d)).astype("float32")
    xq[:, 0] += np.arange(nq) / 1000.0

    index = L2ExactIndex(xb, labels)
    index.build()

    print(index.query(xq[:1]))
