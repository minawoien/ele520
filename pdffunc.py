import numpy as np


def norm1D(my, Sgm, x):
    [n, d] = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * Sgm) * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p


def norm_2D(my, Smg, x_1, x_2):
    [n1, d1] = np.shape(x_1)
    p = np.zeros_like(x_1)
    for i in np.arange(0, n1):
        for j in np.arange(0, d1):
            x = np.array([[x_1[i, j]], [x_2[i, j]]])
            p[i, j] = 1 / ((2*np.pi)**(2/2)*(np.linalg.det(Smg))
                           ** (1/2)) * np.exp(-1/2*np.transpose(x - my)@np.linalg.inv(Smg)@(x-my))
    return p


def norm_dD(my, Sgm, X):
    [l, n] = np.shape(X)
    p = np.zeros(n)
    for i in np.arange(0, n):
        x = np.array(X[:, i]).reshape(l, 1)
        iSgm = np.linalg.inv(Sgm)
        p[i] = 1 / (np.power(2*np.pi, l/2)*np.power(np.linalg.det(Sgm), 1/2)) * \
            np.exp(-1/2 *
                   np.linalg.multi_dot([np.transpose(x-my), iSgm, (x-my)]))
    return p
