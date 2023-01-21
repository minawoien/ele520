import numpy as np


def norm_2D(my, Smg, x_1, x_2):
    [n1, d1] = np.shape(x_1)
    [n2, d2] = np.shape(x_2)
    p = np.zeros_like(x_1)
    for i in np.arange(0, n1):
        for j in np.arange(0, n2):
            x = np.array([[x_1[i, j]], [x_2[i, j]]])
            p[i, j] = 1 / ((2*np.pi)**(2/2)*(np.linalg.det(Smg))
                           ** (1/2)) * np.exp((-1/2*x-my).transpose()@np.linalg.inv(Smg)@(x-my))
    return p
