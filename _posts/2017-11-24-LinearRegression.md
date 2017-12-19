# Linear Regression

Linear Regression is a standard machine learning algorithm used for prediction. This post is about implementing the analytical method for regression.

```python
import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import seaborn as sns
```


```python
def preprocessLinearRegression(filename, X=[], Y=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        next(lines, None)
        dataset = list(lines)

        for i in range(len(dataset)):
            for j in range(len(dataset[0])):
                dataset[i][j] = float(dataset[i][j])
        dataset1 = [row for row in dataset if row[-1] != 0]
        for row in dataset1:
            X.append(row)
```


```python
def LinearRegressionTrain():
    X = []; Y = []
    preprocessLinearRegression('Data_training.csv', X, Y)
    X = np.array(X)
    Y = [row[-1] for row in X]
    Y = np.array(Y)
    # plt.hist(Y, bins=100); plt.xlabel("Y"); plt.ylabel("Frequency"); plt.show()
    # plt.hist(np.log(Y), bins=100);plt.xlabel("log of Y");plt.ylabel("Frequency");plt.show()
    X = np.delete(X, 12, axis=1)
    X = np.insert(X,0,1,axis=1)
    xTranspose = X.transpose()
    w = np.dot(xTranspose,X)
    w = np.linalg.pinv(w)
    w = np.dot(w,xTranspose)
    w = np.dot(w,Y)
    print("w : "); print(w)
    LinearRegressionTest(w)


```


```python
def LinearRegressionTest(w):
    X = []; Y = []; RSS=0.0
    preprocessLinearRegression('Data_test.csv', X, Y)
    X = np.array(X)
    Y = [row[-1] for row in X]
    Y = np.array(Y)
    X = np.delete(X, 12, axis=1)
    X = np.insert(X, 0, 1, axis=1)
    temp = ((Y - np.dot(X,w)) **2)
    for row in temp:
        RSS += row
    print("RSS: " + repr(RSS))
    print(pearsonr(np.dot(X,w),Y))

LinearRegressionTrain()
```
