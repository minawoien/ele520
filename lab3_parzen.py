import pickle
import numpy as np
from pdffunc import *
from pltfunc import plot
from knfunc import knn2D

with open('data/lab3.p', "rb") as fp:
    X = pickle.load(fp)

N = []
for k in range(len(X)):
    N.append(X[k].shape[-1])
    l = X[k].shape[0]
print(N)

x1 = np.arange(-10, 10.5, 0.5).reshape(-1, 1)
x2 = np.arange(-9, 10.5, 0.5).reshape(-1, 1)
x_1, x_2 = np.meshgrid(x1, x2)

h1 = float(input("Tall: "))
p_parzen = []
for i in range(len(X)):
    hn = h1/np.sqrt(N[i])
    hn_matrix = hn**2 * np.eye(l)
    p_parzen.append(0)
    for k in range(N[i]):
        xi = X[i][:, k].reshape(l, 1)
        window_func = norm_2D(xi, hn_matrix, x_1, x_2)
        p_parzen[i] = p_parzen[i] + window_func
    p_parzen[i] = 1/N[i]*p_parzen[i]


col = ['r', 'b']
plot(p_parzen, x_1, x_2, col, False)
