import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        start = time.time()
        trainStats = {}

        transpose = np.matrix.transpose(xTrain)
        inv_trans = np.linalg.inv(np.dot(transpose, xTrain))
        half = np.dot(inv_trans,transpose)
        beta = np.dot(half, yTrain)

        # beta = np.dot(np.dot(np.linalg.inv(np.dot(np.matrix.transpose(xTrain), xTrain)),\
        # np.matrix.transpose(xTrain)), yTrain)

        const = [np.mean(yTrain - np.dot(xTrain, beta))]
        for x in beta:
            const.append(x[0])

        self.beta = const

        mse_train = LinearRegression.mse(self, xTrain, yTrain)
        mse_test = LinearRegression.mse(self, xTest, yTest)

        trainStats[0] = {'Time': time.time() - start, 'Train': mse_train, 'Test': mse_test}

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="new_xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="eng_yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="new_xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="eng_yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
