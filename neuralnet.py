import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Layer:
    def __init__(self, n_inputs, n_neurons, random_state=42):
        np.random.seed(random_state)
        self.w = np.matrix(np.random.randn(n_neurons, n_inputs + 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return np.multiply(x, (1 - x))

    def forward(self, inputs):
        z = self.w * inputs
        a = self.sigmoid(z)
        self.output = np.insert(a, 0, 1, axis=0)

    def backward(self, next_w, next_error):
        deriv = self.sigmoid_deriv(self.output)
        error = np.multiply(next_w.T * next_error, deriv)
        self.error = np.delete(error, 0, axis=0)


class InputLayer(Layer):
    def __init__(self):
        pass

    def forward(self, inputs):
        self.output = np.insert(inputs, 0, 1, axis=0)


class OutputLayer(Layer):
    def forward(self, inputs):
        z = self.w * inputs
        a = self.sigmoid(z)
        self.output = a

    def backward(self, y):
        self.error = self.output - y


class DeepNeuralNetwork:
    def __init__(self, df, n_neurons, test_size=0.2, random_state=42):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.m, self.n = X_train.shape
        self.n_classes = y.nunique()

        self.X_train = self.normalizeFeatures(X_train, fit=True).T
        self.X_test = self.normalizeFeatures(X_test).T
        self.y_train = np.matrix(pd.get_dummies(y_train)).T
        self.labels_train = np.matrix(y_train)
        self.y_test = np.matrix(pd.get_dummies(y_test)).T
        self.labels_test = np.matrix(y_test)

        self.layers = []
        self.layers.append(InputLayer())
        prev_n = self.n
        for n in n_neurons:
            self.layers.append(Layer(prev_n, n))
            prev_n = n
        self.layers.append(OutputLayer(prev_n, self.n_classes))

    def normalizeFeatures(self, X, fit=False):
        if fit:
            scaling = list(zip(X.min(), X.max()))
            self.scaling = list(map(lambda x: (0, 1) if x[0] == x[1] else (x[0], x[1]), scaling))

        minmax = list(zip(*self.scaling))
        min, max = np.array(minmax[0]), np.array(minmax[1])

        return np.matrix((X - min) / (max - min))

    def gradientDescent(self, alpha=1e-3, threshold=1e-5, epochs=1000):
        i = 0
        self.J = []
        self.train_accuracy = []
        self.test_accuracy = []

        while True:
            i += 1
            self.forward(self.X_train)
            self.backward()

            self.J.append(self.costFunction(self.layers[-1].output))

            grads = self.gradient()

            for l in range(len(self.layers) - 1):
                self.layers[l + 1].w -= alpha * grads[l]

            self.train_accuracy.append(self.accuracy(self.X_train, self.labels_train))
            self.test_accuracy.append(self.accuracy(self.X_test, self.labels_test))

            if len(self.J) > 1:
                if abs(self.J[-1] - self.J[-2]) < threshold or i == epochs:
                    break

    def plotCostAccuracy(self):
        try:
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.plot(self.J)
            plt.title("Value of the cost function over epochs")
            plt.xlabel("epoch")
            plt.ylabel("cost function value")

            plt.subplot(122)
            plt.plot(self.train_accuracy, label="train")
            plt.plot(self.test_accuracy, label="test")
            plt.title("Accuracy over epochs")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.show()
        except AttributeError:
            print("No gradient descent was performed")

    def costFunction(self, y_pred):
        y, m = self.y_train, self.m

        return -1 / m * np.sum(np.multiply(y, np.log(y_pred)) + np.multiply((1 - y), np.log(1 - y_pred)))

    def forward(self, x):
        inputs = x
        for l in range(len(self.layers)):
            self.layers[l].forward(inputs)
            inputs = self.layers[l].output

    def backward(self):
        self.layers[-1].backward(self.y_train)
        next_w, next_error = self.layers[-1].w, self.layers[-1].error
        for l in range(len(self.layers) - 2, 0, -1):
            self.layers[l].backward(next_w, next_error)
            next_w, next_error = self.layers[l].w, self.layers[l].error

    def gradient(self):
        grads = []
        for l in range(len(self.layers) - 1):
            grads.append(self.layers[l + 1].error * self.layers[l].output.T)

        return grads

    def confusion(self, X, y):
        y_pred = self.predict(X)

        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)

        for i in range(len(y_pred)):
            matrix[y.T[i], y_pred[i]] += 1

        return matrix

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

    def predict(self, X):
        self.forward(X)
        return np.array(self.layers[-1].output).argmax(axis=0)
