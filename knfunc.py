import numpy as np


def knn2D(X, kn, x1, x2):
    [n, d] = np.shape(x1)
    [l, N] = np.shape(X)
    print(l)
    p = np.zeros_like(x1)
    for i in np.arange(0, n):
        for j in np.arange(0, d):
            x = np.array([[x1[i, j]], [x2[i, j]]])
            x = np.tile(x, N)
            difference = X - x
            R = np.sqrt(
                np.diag(np.dot(difference.T, difference)).reshape(N, 1))
            index = np.argsort(R, 0)
            r = R[index[kn - 1]][0][0]
            Vn = np.pi * r**2
            p[i, j] = (kn/N)/Vn
    return p


def knndD(X, kn, Xin):
    [d, n] = np.shape(Xin)
    [l, N] = np.shape(X)
    p = np.zeros(n)
    for i in np.arange(0, n-1):
        x = np.array(Xin[:, i]).reshape(l, 1)
        x = np.tile(x, N)
        difference = X - x
        R = np.sqrt(np.diag(np.dot(difference.T, difference)))
        index = np.argsort(R, 0)
        r = R[index[kn - 1]]
        Vn = np.pi * r**2
        p[i] = (kn/N)/Vn
    return p
