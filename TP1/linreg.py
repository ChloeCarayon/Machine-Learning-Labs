import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


def preprocessingData(file):
    data = pd.read_excel(file)
    X = np.asmatrix(data.iloc[:, :-1])
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)
    y = np.asmatrix(data.iloc[:, -1:])
    w = np.array([[1] for i in range(n + 1)])

    normalizeFeatures(X, n)

    return X, y, w, m, n


def normalizeFeatures(X, n):
    for feat in range(1, n + 1):
        min = np.min(X[:, feat])
        max = np.max(X[:, feat])
        X[:, feat] = (X[:, feat] - min) / (max - min)


def costFunction(X, y, w, m):
    return float(1 / (2 * m) * (X * w - y).T * (X * w - y))


def gradientDescent(file, alpha=0.03, threshold=1e-3, iter=1000):
    X, y, w, m, n = preprocessingData(file)

    J = []
    i = 0
    while True:
        J.append(costFunction(X, y, w, m))
        delta = 1 / m * (X.T * X * w - X.T * y)
        w = w - alpha * delta
        i += 1

        if (abs(costFunction(X, y, w, m) - J[-1]) < threshold) or (i == iter):
            break

    print("\nBuild regression")
    print("Coefficients:\n", w)
    print(abs(costFunction(X, y, w, m) - J[-1]))
    print("MAE:", meanAbsoluteError(X, y, w, m))
    print("RMSE:", rootMeanSquaredError(X, y, w, m))
    print("R2:", r2(X, y, w, m))

    print("\nNormal equation:")
    print(normalEquation(X, y))

    plt.plot(np.log(J))
    plt.title("Value of the cost function over iterations")
    plt.xlabel("iteration")
    plt.ylabel("log of cost function value")
    plt.show()


def meanAbsoluteError(X, y, w, m):
    return float(1 / m * sum(abs(X * w - y)))


def rootMeanSquaredError(X, y, w, m):
    return float(1 / m * (X * w - y).T * (X * w - y))**(1 / 2)


def r2(X, y, w, m):
    return float(((X * w - np.mean(y)).T * (X * w - np.mean(y))) / ((y - np.mean(y)).T * (y - np.mean(y))))


def normalEquation(X, y):
    return (X.T * X)**-1 * X.T * y


gradientDescent("../data/CCPP.xlsx", alpha=0.9)

X, y, w, m, n = preprocessingData("../data/CCPP.xlsx")
print("Scikit-learn")
reg = linear_model.LinearRegression()
reg.fit(X, y)
s = reg.coef_
s[0, 0] = reg.intercept_
s = s.T
print(s)
print("MAE:", meanAbsoluteError(X, y, s, m))
print("RMSE:", rootMeanSquaredError(X, y, s, m))
print("R2:", r2(X, y, s, m))
