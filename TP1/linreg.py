import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data):
        self.X, self.y, self.w, self.m, self.n = self.preprocessingData(data)

        self.MAE = None
        self.RMSE = None
        self.R2 = None

    def preprocessingData(self, df):
        X = self.normalizeFeatures(df.iloc[:, :-1])
        X = np.asmatrix(X)
        m, n = X.shape
        X = np.insert(X, 0, 1, axis=1)
        y = np.asmatrix(df.iloc[:, -1:])
        w = np.array([[1] for i in range(n + 1)])

        return X, y, w, m, n

    def normalizeFeatures(self, X):
        self.scaling = list(zip(X.min(), X.max()))

        return (X - X.min()) / (X.max() - X.min())

    def costFunction(self, X, y, w, m):
        return float(1 / (2 * m) * (X * w - y).T * (X * w - y))

    def gradientDescent(self, alpha=0.03, threshold=1e-3, iter=1000, autoAlpha=True):
        i = 0
        self.J = []

        while True:
            i += 1
            self.J.append(self.costFunction(self.X, self.y, self.w, self.m))

            grad = 1 / self.m * (self.X.T * self.X * self.w - self.X.T * self.y)
            self.w = self.w - alpha * grad

            if len(self.J) > 1:
                if autoAlpha and self.J[-1] > self.J[-2]:
                    alpha /= 1.1

                if abs(self.J[-1] - self.J[-2]) < threshold or i == iter:
                    break

        self.updateScores()

    def plotCost(self):
        try:
            plt.plot(np.log(self.J))
            plt.title("Value of the cost function over iterations")
            plt.xlabel("iteration")
            plt.ylabel("log of cost function value")
            plt.show()
        except AttributeError:
            print("No gradient descent was performed")

    def normalEquation(self):
        self.w = (self.X.T * self.X)**-1 * self.X.T * self.y

        self.updateScores()

    def meanAbsoluteError(self, X, y, w, m):
        return float(1 / m * sum(abs(X * w - y)))

    def rootMeanSquaredError(self, X, y, w, m):
        return float(1 / m * (X * w - y).T * (X * w - y))**(1 / 2)

    def r2(self, X, y, w, m):
        return float(((X * w - np.mean(y)).T * (X * w - np.mean(y))) / ((y - np.mean(y)).T * (y - np.mean(y))))

    def updateScores(self):
        self.MAE = self.meanAbsoluteError(self.X, self.y, self.w, self.m)
        self.RMSE = self.rootMeanSquaredError(self.X, self.y, self.w, self.m)
        self.R2 = self.r2(self.X, self.y, self.w, self.m)

    def predict(self, test):
        X_test = np.asmatrix(test)
        minmax = list(zip(*self.scaling))
        min, max = np.array(minmax[0]), np.array(minmax[1])
        X_test = (X_test - min) / (max - min)

        return X_test * self.w[1:] + self.w[0]
