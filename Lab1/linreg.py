import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, data, test_size=0.2, random_state=42):
        self.preprocessingData(data, test_size, random_state)

    def preprocessingData(self, df, test_size, random_state):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.m, self.n = X_train.shape
        self.X_train = self.normalizeFeatures(X_train, fit=True)
        self.X_test = self.normalizeFeatures(X_test)
        self.y_train = np.asmatrix(y_train)
        self.y_test = np.asmatrix(y_test)
        self.w = np.asmatrix([[1] for i in range(self.n + 1)])

    def normalizeFeatures(self, X, fit=False):
        if fit:
            self.scaling = list(zip(X.min(), X.max()))

        minmax = list(zip(*self.scaling))
        min, max = np.array(minmax[0]), np.array(minmax[1])
        X = np.asmatrix((X - min) / (max - min))

        return np.insert(X, 0, 1, axis=1)

    def costFunction(self, X, y, w, m):
        return float(1 / (2 * m) * (X * w - y).T * (X * w - y))

    def gradientDescent(self, alpha=0.03, threshold=1e-3, iter=1000, autoAlpha=True):
        i = 0
        self.J = []

        while True:
            i += 1
            self.J.append(self.costFunction(self.X_train, self.y_train, self.w, self.m))

            grad = 1 / self.m * (self.X_train.T * self.X_train * self.w - self.X_train.T * self.y_train)
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
        self.w = (self.X_train.T * self.X_train)**-1 * self.X_train.T * self.y_train

        self.updateScores()

    def meanAbsoluteError(self, X, y, w, m):
        return float(1 / m * sum(abs(X * w - y)))

    def rootMeanSquaredError(self, X, y, w, m):
        return float(1 / m * (X * w - y).T * (X * w - y))**(1 / 2)

    def r2(self, X, y, w):
        return float(1 - ((y - X * w).T * (y - X * w)) / ((y - y.mean()).T * (y - y.mean())))

    def updateScores(self):
        self.MAE = self.meanAbsoluteError(self.X_train, self.y_train, self.w, self.m)
        self.RMSE = self.rootMeanSquaredError(self.X_train, self.y_train, self.w, self.m)
        self.R2 = self.r2(self.X_train, self.y_train, self.w)

        self.test_MAE = self.meanAbsoluteError(self.X_test, self.y_test, self.w, self.X_test.shape[0])
        self.test_RMSE = self.rootMeanSquaredError(self.X_test, self.y_test, self.w, self.X_test.shape[0])
        self.test_R2 = self.r2(self.X_test, self.y_test, self.w)

    def predict(self, X_pred):
        if X_pred.shape[1] < self.n + 1:
            X_pred = self.normalizeFeatures(X_pred)

        return X_pred * self.w
