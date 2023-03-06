import pickle
import numpy as np
from pdffunc import *
from pltfunc import plot
from knfunc import knn2D
kn = 1
x1 = np.arange(-10, 10.5, 0.5).reshape(-1, 1)
x2 = np.arange(-9, 10.5, 0.5).reshape(-1, 1)
x_1, x_2 = np.meshgrid(x1, x2)
with open('data/lab3.p', "rb") as fp:
    X = pickle.load(fp)

p = []
for i in range(len(X)):
    p.append(knn2D(X[i], kn, x_1, x_2))

col = ['r', 'b']
plot(p, x_1, x_2, col, False)
