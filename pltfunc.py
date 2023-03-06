import matplotlib.pyplot as plt
import numpy as np


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
            ax.plot_wireframe(x_1, x_2, p[i], color=col[i])
    plt.show()
