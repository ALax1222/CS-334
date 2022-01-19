import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA as pca
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def normalize(xTrain, xTest):
    ss = StandardScaler()

    ss.fit(xTrain)

    xTrain = ss.transform(xTrain)
    xTest = ss.transform(xTest)

    return xTrain, xTest


def PCA(xTrain, xTest):
    pca1 = pca(n_components=9)
    pca1.fit(xTrain)

    xTrain = pca1.transform(xTrain)
    xTest = pca1.transform(xTest)

    var = pca1.explained_variance_ratio_
    print("Percent of Variance Explained:", np.sum(var))
    print("Components:", pca1.components_)

    return xTrain, xTest



def log_reg(xTrain, yTrain, xTest, yTest):
    lr = LogisticRegression()

    lr.fit(xTrain, yTrain)

    probability = []

    probs = lr.predict_proba(xTest)
    for i in range(len(probs)):
        probability.append(probs[i][0])
    y_pred = lr.predict(xTest)

    # plot_roc_curve(lr, xTest, yTest)
    # plt.show()

    return y_pred, probability


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="yTest.csv",
                        help="filename for labels associated with the test data")
    # parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    args.epoch = 5
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    roc = []

    xTrain, xTest = normalize(xTrain, xTest)

    yPred, probs = log_reg(xTrain, yTrain, xTest, yTest)

    roc1 = roc_curve(yTest, probs)

    plt.plot(roc1[1], roc1[0])

    xTrain, xTest = PCA(xTrain, xTest)

    yPred, probs = log_reg(xTrain, yTrain, xTest, yTest)

    roc2 = roc_curve(yTest, probs)

    plt.plot(roc2[1], roc2[0])
    plt.show()


if __name__ == "__main__":
    main()