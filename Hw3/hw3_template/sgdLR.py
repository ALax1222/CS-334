import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}

        # xTrain, X_test, yTrain, y_test = train_test_split(xTrain, yTrain, test_size=0.6)

        batches = len(xTrain) // self.bs

        beta = np.zeros((12, 1))

        # transpose = np.matrix.transpose(xTrain)
        # inv_trans = np.linalg.inv(np.dot(transpose, xTrain))
        # half = np.dot(inv_trans,transpose)
        # beta = np.dot(half, yTrain)
        
        # # Closed form solution

        start = time.time()
        for _ in range(1, self.mEpoch + 1):
            shuffled_indices = np.random.permutation(len(xTrain))

            xTrain = xTrain[shuffled_indices]
            yTrain = yTrain[shuffled_indices]

            xTrains = np.array_split(xTrain, batches)
            yTrains = np.array_split(yTrain, batches)

            for i in range(len(xTrains)):
                grads = np.zeros((12, 1))
                for j in range(len(xTrains[i])):
                    transpose = np.matrix.transpose(xTrains[i][j])
                    step1 = np.dot(xTrains[i][j], beta)
                    step2 = yTrains[i][j] - step1
                    gradient = transpose * step2
                    for k in range(len(gradient)):
                        grads[k] += gradient[k]

                grads /= len(xTrains[i])

                beta += self.lr*grads

                const = yTrains[i] - np.dot(xTrains[i], beta)
                const = [np.mean(const)]

                beta_new = np.ndarray.tolist(beta)

                fin_beta = []
                for x in beta_new:
                    fin_beta.append(x[0])

                const.extend(fin_beta)

                self.beta = const

                mse_train = LinearRegression.mse(self, xTrain, yTrain)
                mse_test = LinearRegression.mse(self, xTest, yTest)

                round = (_ - 1) * len(xTrains) + i

                trainStats[round] = {'Time': time.time() - start, 'Train': mse_train, 'Test': mse_test}
                start = time.time()

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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()

    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)

if __name__ == "__main__":
    main()

