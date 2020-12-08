import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Regression:
    def __init__(self, df, test_size=0.2, random_state=42):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.m, self.n = X_train.shape
        self.X_train = self.normalizeFeatures(X_train, fit=True)
        self.X_test = self.normalizeFeatures(X_test)
        self.y_train = np.matrix(y_train)
        self.y_test = np.matrix(y_test)

    def normalizeFeatures(self, X, fit=False):
        if fit:
            self.scaling = list(zip(X.min(), X.max()))

        minmax = list(zip(*self.scaling))
        min, max = np.array(minmax[0]), np.array(minmax[1])
        X = np.matrix((X - min) / (max - min))

        return np.insert(X, 0, 1, axis=1)

    def gradientDescent(self, alpha=0.03, threshold=1e-3, iter=1000, autoAlpha=True):
        i = 0
        self.J = []
        self.w = np.matrix([[1] for i in range(self.n + 1)])

        while True:
            i += 1
            self.J.append(self.costFunction())

            self.w = self.w - alpha * self.gradient()

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


class LinearRegression(Regression):
    def costFunction(self):
        X, y, w, m = self.X_train, self.y_train, self.w, self.m
        return float(1 / (2 * m) * (X * w - y).T * (X * w - y))

    def gradient(self):
        X, y, w, m = self.X_train, self.y_train, self.w, self.m
        return 1 / m * X.T * (X * w - y)

    def normalEquation(self):
        X, y = self.X_train, self.y_train
        self.w = (X.T * X)**-1 * X.T * y

        self.updateScores()

    def meanAbsoluteError(self, X, y):
        w, m = self.w, X.shape[0]
        return float(1 / m * sum(abs(X * w - y)))

    def rootMeanSquaredError(self, X, y):
        w, m = self.w, X.shape[0]
        return float(1 / m * (X * w - y).T * (X * w - y))**(1 / 2)

    def r2(self, X, y):
        w = self.w
        return float(1 - ((y - X * w).T * (y - X * w)) / ((y - y.mean()).T * (y - y.mean())))

    def updateScores(self):
        self.MAE = self.meanAbsoluteError(self.X_train, self.y_train)
        self.RMSE = self.rootMeanSquaredError(self.X_train, self.y_train)
        self.R2 = self.r2(self.X_train, self.y_train)

        self.MAE_test = self.meanAbsoluteError(self.X_test, self.y_test)
        self.RMSE_test = self.rootMeanSquaredError(self.X_test, self.y_test)
        self.R2_test = self.r2(self.X_test, self.y_test)

    def predict(self, X_pred):
        if X_pred.shape[1] < self.n + 1:
            X_pred = self.normalizeFeatures(X_pred)

        return X_pred * self.w


class LogisticRegression(Regression):
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X * self.w))

    def costFunction(self):
        y, m, sigmoid = self.y_train, self.m, self.sigmoid(self.X_train)
        return float(-1 / m * (y.T * np.log(sigmoid) + (1 - y.T) * np.log(1 - sigmoid)))

    def gradient(self):
        X, y = self.X_train, self.y_train
        return X.T * (self.sigmoid(X) - y)

    def confusion(self, X, y):
        y_pred = self.predict(X)
        TN = int((1 - y_pred[y == 0]).sum())
        FP = int(y_pred[y == 0].sum())
        FN = int((1 - y_pred[y == 1]).sum())
        TP = int(y_pred[y == 1].sum())

        return np.matrix([[TN, FP], [FN, TP]])

    def accuracy(self, X, y):
        return (y == self.predict(X)).mean()

    def precision(self, X, y):
        matrix = self.confusion(X, y)
        return matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])

    def recall(self, X, y):
        matrix = self.confusion(X, y)
        return matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])

    def f1(self, X, y):
        precision = self.precision(X, y)
        recall = self.recall(X, y)
        return 2 * precision * recall / (precision + recall)

    def updateScores(self):
        self.matrix = self.confusion(self.X_train, self.y_train)
        self.acc = self.accuracy(self.X_train, self.y_train)
        self.pre = self.precision(self.X_train, self.y_train)
        self.rec = self.recall(self.X_train, self.y_train)
        self.f1 = self.f1(self.X_train, self.y_train)

        self.acc_test = self.accuracy(self.X_test, self.y_test)
        self.pre_test = self.precision(self.X_test, self.y_test)

    def predict(self, X_pred):
        if X_pred.shape[1] < self.n + 1:
            X_pred = self.normalizeFeatures(X_pred)

        return self.sigmoid(X_pred).round()
