import pandas as pd
from neuralnet import DeepNeuralNetwork


def neuralnet():
    features = pd.read_csv("Lab4/data/features.txt", header=None)
    labels = pd.read_csv("Lab4/data/labels.txt", header=None).replace(10, 0)
    df = features.copy()
    df["label"] = labels

    NN = DeepNeuralNetwork(df, [50])
    NN.gradientDescent(alpha=1e-4, threshold=1e-4, epochs=1000)
    print("NN 50 neurons")
    print("Train accuracy:", NN.accuracy(NN.X_train, NN.labels_train))
    print("Test accuracy:", NN.accuracy(NN.X_test, NN.labels_test))
    NN.plotCostAccuracy()


neuralnet()
