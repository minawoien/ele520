import pickle
from classifier import classify, calculate_classifier
import matplotlib.pyplot as plt

pfile = 'data/lab4_2.p'
with open(pfile, "rb") as fp:
    X_2D3cl, X_2D4cl, X_2D4cl_ms, X_2D4cl_hs, X_3D3cl_ms, Y_2D3cl, Y_2D4cl, Y_2D4cl_ms, Y_2D4cl_hs, Y_3D3cl_ms = pickle.load(
        fp)


def print_results(Y, X):
    M = len(Y)

    gx = M*[None]
    gy = M*[None]
    cx = M*[None]
    cy = M*[None]
    for i in range(0, M):
        gx[i], cx[i] = classify(Y, X[i], "ML")  # Classify training set
        gy[i], cy[i] = classify(Y, Y[i], "ML")  # Classify the test set

    calculate_classifier(cy, M)

    print("\nTest:")
    calculate_classifier(cx, M)


# a)
print("2D 3 classes")
print_results(Y_2D3cl, X_2D3cl)

# b)
print("\n2D 4 classes")
print_results(Y_2D4cl, X_2D4cl)

# c)
print("\n2D 4 classes with medium separability")
print_results(Y_2D4cl_ms, X_2D4cl_ms)

# d)
print("\n2D 4 classes with high separability")
print_results(Y_2D4cl_hs, X_2D4cl_hs)

# e)
print("\n3D 3 classes with medium separability")
print_results(Y_3D3cl_ms, X_3D3cl_ms)
