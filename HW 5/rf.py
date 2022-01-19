import argparse
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, nest, maxFeat, criterion='gini', maxDepth=5, minLeafSample=5):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.feature_indices = []
        self.forest = None

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """

        forest = []
        predictions = []
        cols = []

        full_list = range(0,10)
        for tree_number in range(self.nest):
            cols = random.sample(full_list, k=self.maxFeat)

            xFeat_subset = xFeat[:, cols]

            n_iterations = 1
            for i in range(n_iterations):
                self.feature_indices.append(cols)
                indices = np.random.choice(range(0, len(xFeat_subset)), size=len(xFeat_subset), replace=True)
                
                indices = indices.tolist()
                indices.sort()

                not_indices = []
                for i in range(len(xFeat_subset)):
                    if i not in indices:
                        not_indices.append(i)

                xTrain = xFeat_subset[indices]
                yTrain = y[indices]

                xTest = xFeat_subset[not_indices]
                yTest = y[not_indices]

                
                dtc = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_samples_leaf=self.minLeafSample)
                dtc.fit(xTrain, yTrain)
                pred = dtc.predict(xTest)

                yPred = []
                j = 0
                for i in range(len(xFeat_subset)):
                    if i in not_indices:
                        yPred.append(pred[j])
                        j += 1
                    else:
                        yPred.append(2)

                forest.append(dtc)
                predictions.append(yPred)

        final_predictions = []
        for i in range(len(predictions[0])):
            one = 0
            zero = 0
            for j in range(len(predictions)):
                if predictions[j][i] == 0:
                    zero += 1
                elif predictions[j][i] == 1:
                    one += 1
            if zero >= one:
                final_predictions.append(0)
            else:
                final_predictions.append(1)
        self.forest = forest
        return accuracy_score(y, final_predictions)

    def predict(self, xFeat, yTest):
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
            Predicted response per sample
        """
        yHat = []
        predictions = []
        for i in range(len(self.feature_indices)):
            xFeat_subset = xFeat[:, self.feature_indices[i]]
            predictions.append(self.forest[i].predict(xFeat_subset))

        for i in range(len(predictions[0])):
            one = 0
            zero = 0
            for j in range(len(predictions)):
                if predictions[j][i] == 0:
                    zero += 1
                elif predictions[j][i] == 1:
                    one += 1
            if zero >= one:
                yHat.append(0)
            else:
                yHat.append(1)

        return yHat


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

    np.random.seed(args.seed)
    nest = [2,4,6,8,10]
    max_feat = [3,4,5,6,9]
    criterion = ["gini"]
    max_depth = [1,5,10,20,25]
    minLeafSample = [1,5,10,25,50]

    # nest=[7]
    # max_feat=[7]
    # criterion=["entropy"]
    # max_depth=[20]
    # minLeafSample=[1]

    i = 0
    curr = [0,0,0]
    for n in nest:
        for mf in max_feat:
            for cr in criterion:
                for md in max_depth:
                    for mls in minLeafSample:

                        model = RandomForest(nest=n, maxFeat=mf, criterion=cr, maxDepth=md, minLeafSample=mls)


                        trainStats = model.train(xTrain, yTrain)
                        yHat = model.predict(xTest, yTest)
                        acc = accuracy_score(yTest, yHat)
                        # print(trainStats)
                        # print(acc)
                        curr[0] = acc
                        curr[1] = trainStats
                        curr[2] = [n,mf,cr,md,mls]
                        print(curr)


if __name__ == "__main__":
    main()