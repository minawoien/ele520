import pickle
from classifier import classify
import numpy as np
import pandas as pd

# Simulated data for a two class problem
file = 'data/lab4.p'
with open(file, "rb") as f:
    X, Y = pickle.load(f)
M = len(Y)

gx = M*[None]
gy = M*[None]
cx = M*[None]
cy = M*[None]
# Takes in Y, input training data and X, data to be classified as feature vector matrix.
# The function returns p, the class specific density function,  g, containing the discriminant
# function values for each feature, and c, the classification for each feature vector in X
for i in range(0, M):
    gx[i], cx[i] = classify(Y, X[i], "ML")  # Classify training set
    gy[i], cy[i] = classify(Y, Y[i], "ML")  # Classify the test set

actual1 = np.array([0] * len(cy[0]))
actual2 = np.array([1] * len(cy[1]))
print(cy[1])
print(actual2)

cm1 = pd.crosstab(actual1, cy[0], margins=True, rownames=[
    'Actual'], colnames=['Predicted'], margins_name="Total")
cm2 = pd.crosstab(actual2, cy[1], margins=True, rownames=[
    'Actual'], colnames=['Predicted'], margins_name="Total")

# Compute error rate
Ac = (cm1[0][0] + cm2[1][1]) / (cm1["Total"]["Total"]+cm2["Total"]["Total"])
print(f"P(error): {1-Ac}")

R_1 = cm1[0][0] / cm1["Total"]["Total"]
print(f"P(correct|ω1): {R_1}")

R_2 = cm2[1][1] / cm2["Total"]["Total"]
print(f"P(correct|ω2): {R_2}")


# Test
print("\nTest:")
cm1 = pd.crosstab(actual1, cx[0], margins=True, rownames=[
    'Actual'], colnames=['Predicted'], margins_name="Total")
cm2 = pd.crosstab(actual2, cx[1], margins=True, rownames=[
    'Actual'], colnames=['Predicted'], margins_name="Total")

# Compute error rate
Ac = (cm1[0][0] + cm2[1][1]) / (cm1["Total"]["Total"]+cm2["Total"]["Total"])
print(f"P(error): {1-Ac}")

R_1 = cm1[0][0] / cm1["Total"]["Total"]
print(f"P(correct|ω1): {R_1}")

R_2 = cm2[1][1] / cm2["Total"]["Total"]
print(f"P(correct|ω2): {R_2}")
