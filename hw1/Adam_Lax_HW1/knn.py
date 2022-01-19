import argparse
import numpy as np
import pandas as pd
from operator import itemgetter


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.X_Train = xFeat
        self.y_train = y
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label

        for i in range(len(xFeat)):
            dis = ((self.X_Train.iloc[:, 0] - xFeat.iloc[i, 0]) ** 2)
            for x in xFeat:
                if x != "f1":
                    dis += ((self.X_Train[x] - xFeat[x][i]) ** 2)
            # Calculates distance

            dis = dis.sort_values()[:self.k]
        
            zero = 0
            one = 0
            for i in range(len(self.X_Train)):
                if i in dis:
                    if self.y_train[i] == 0:
                        zero += 1
                    else:
                        one += 1
            yHat.append(0 if zero > one else 1)
            # Calculates most common y-value from nearest neighbors

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = 0
    right = 0
    wrong = 0

    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)

    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    from PIL import Image

    image = Image.open('Q3.png')
    image.show()
    # prints image of graph


if __name__ == "__main__":
    main()

