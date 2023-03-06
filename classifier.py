import numpy as np
from pdffunc import norm_dD
from knfunc import knndD
import pandas as pd
import matplotlib.pyplot as plt


def classify(Y, Xin, type):
    M = len(Y)
    N = []
    for i in range(0, M):
        N.append([])
        [d, N[i]] = Y[i].shape
    N = np.array(N)
    pw = N/sum(N)

    if type == "ML":
        mu = []
        sgm = []
    elif type == "PZ":
        # h1 = 0.1
        h1 = 5
    elif type == "KN":
        # knn = 5
        knn = 1

    pxw = np.zeros((M, Xin.shape[1]))
    p = np.zeros((1, Xin.shape[1]))
    for i in range(0, M):
        if type == "ML":
            mu.append(np.mean(Y[i], 1).reshape(d, 1))
            sgm.append(np.cov(Y[i]))
            pxw[i, :] = norm_dD(mu[i], sgm[i], Xin)
        elif type == "PZ":
            hn = h1/np.sqrt(N[i])
            hn_matrix = hn**2 * np.eye(d)
            for k in range(N[i]):
                xi = Y[i][:, k].reshape(d, 1)
                window_func = norm_dD(xi, hn_matrix, Xin)
                pxw[i, :] = pxw[i, :] + window_func
            pxw[i, :] = 1 / N[i] * pxw[i, :]
        elif type == "KN":
            pxw[i, :] = knndD(Y[i], knn, Xin)
        p = p + pw[i] * pxw[i, :]

    g = np.zeros((M, Xin.shape[1]))
    for i in range(0, M):
        g[i, :] = (pw[i]*pxw[i, :]) / p

    c = g.argmax(axis=0)

    return g, c


def calculate_classifier(c, M):
    actual = M*[None]
    cm = M*[None]
    Ac_num = 0
    Ac_dnum = 0
    R = M*[None]
    for i in range(M):
        actual[i] = np.array([i] * len(c[i]))
        cm[i] = pd.crosstab(actual[i], c[i], margins=True, rownames=[
            'Actual'], colnames=['Predicted'], margins_name="Total")
        Ac_num += cm[i][i][i]
        Ac_dnum += cm[i]["Total"]["Total"]
        R[i] = cm[i][i][i] / cm[i]["Total"]["Total"]
        print(f"P(correct|ω{i+1}): {R[i]}")
    print(f"P(error): {1-Ac_num/Ac_dnum}")
