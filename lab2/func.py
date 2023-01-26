import sys
import numpy as np
import matplotlib.pyplot as plt


def norm_2D(my, Smg, x_1, x_2):
    [n1, d1] = np.shape(x_1)
    p = np.zeros_like(x_1)
    for i in np.arange(0, n1):
        for j in np.arange(0, d1):
            x = np.array([[x_1[i, j]], [x_2[i, j]]])
            p[i, j] = 1 / ((2*np.pi)**(2/2)*(np.linalg.det(Smg))
                           ** (1/2)) * np.exp(-1/2*np.transpose(x - my)@np.linalg.inv(Smg)@(x-my))
    return p


def plot(p, x_1, x_2, col, db):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    if db:
        ax.plot_wireframe(x_1, x_2, np.where(
            p[1] < p[0], p[0], np.nan), color=col[0])
        ax.plot_wireframe(x_1, x_2, np.where(
            p[1] >= p[0], p[1], np.nan), color=col[1])
    else:
        for i in range(len(p)):
            ax.plot_surface(x_1, x_2, p[i], color=col[i])
    plt.show()
