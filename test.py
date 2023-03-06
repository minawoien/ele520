import numpy as np
import pickle
import pandas as pd
file = 'data/lab4_data.p'
with open(file, "rb") as f:
    pxwx, Pwx, gx, Cx, CNx, pxwy, Pwy, gy, Cy, CNy = pickle.load(f)


confusion_matrix = pd.crosstab(Cy[0], Cx[0], margins=True, rownames=[
                               'Actual'], colnames=['Predicted'], margins_name="Total")
print(confusion_matrix)

# Accuracy = (TP+TN)/population = (4+5)/12 = 0.75
Ac = (confusion_matrix[0][0] + confusion_matrix[1][1]) / \
    confusion_matrix["Total"]["Total"]

print(Ac)
# Compute error rate
P_error = 1-Ac
print(P_error)
