import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


def preprocessingData(df):
    X = np.asmatrix(df.iloc[:, :-1])
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)
    y = np.asmatrix(df.iloc[:, -1:])
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


def gradientDescent(df, alpha=0.03, threshold=1e-3, iter=1000, autoAlpha=True):
    X, y, w, m, n = preprocessingData(df)

    J = [costFunction(X, y, w, m)]
    i = 0
    while True:
        J.append(costFunction(X, y, w, m))
        delta = 1 / m * (X.T * X * w - X.T * y)
        w = w - alpha * delta
        i += 1

        if autoAlpha and J[-1] > J[-2]:
            alpha /= 1.1

        if (abs(costFunction(X, y, w, m) - J[-1]) < threshold) or (i == iter):
            break

    print("\nGradient descent")
    print("Coefficients:\n", w)
    print(abs(costFunction(X, y, w, m) - J[-1]))
    print("MAE:", meanAbsoluteError(X, y, w, m))
    print("RMSE:", rootMeanSquaredError(X, y, w, m))
    print("R2:", r2(X, y, w, m))

    print("\nNormal equation:")
    w_norm = normalEquation(X, y)
    print(w_norm)
    print("MAE:", meanAbsoluteError(X, y, w_norm, m))
    print("RMSE:", rootMeanSquaredError(X, y, w_norm, m))
    print("R2:", r2(X, y, w_norm, m))

    plt.plot(np.log(J))
    plt.title("Value of the cost function over iterations")
    plt.xlabel("iteration")
    plt.ylabel("log of cost function value")
    plt.show()

    return w_norm


def meanAbsoluteError(X, y, w, m):
    return float(1 / m * sum(abs(X * w - y)))


def rootMeanSquaredError(X, y, w, m):
    return float(1 / m * (X * w - y).T * (X * w - y))**(1 / 2)


def r2(X, y, w, m):
    return float(((X * w - np.mean(y)).T * (X * w - np.mean(y))) / ((y - np.mean(y)).T * (y - np.mean(y))))


def normalEquation(X, y):
    return (X.T * X)**-1 * X.T * y

# data = pd.read_excel("../data/CCPP.xlsx")
# gradientDescent(data, alpha=0.9)

# X, y, w, m, n = preprocessingData(data)
# print("Scikit-learn")
# reg = linear_model.LinearRegression()
# reg.fit(X, y)
# s = reg.coef_
# s[0, 0] = reg.intercept_
# s = s.T
# print(s)
# print("MAE:", meanAbsoluteError(X, y, s, m))
# print("RMSE:", rootMeanSquaredError(X, y, s, m))
# print("R2:", r2(X, y, s, m))
