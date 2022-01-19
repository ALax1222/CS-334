import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from collections import Counter


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.tree = None
    
    def best_split(self):
        entropy = 1
        gini = 0
        best_feature = None
        best_value = None
        for feature in self.xFeat:
            if feature != 'y':
                if self.criterion == "gini":
                    x = self.GINI_best_split_in_variable(feature)
                    if x[0] >= gini:
                        gini = x[0]
                        best_feature = x[1]
                        best_value = x[2]
                else:
                    x = self.entropy(feature)
                    if x[0] <= entropy:
                        entropy = x[0]
                        best_feature = x[1]
                        best_value = x[2]

                
        return [best_value, best_feature]

    def GINI_impurity(self, y1_count, y2_count):
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0

        n = y1_count + y2_count
        if n == 0:
            return 0.0

        p1 = y1_count / n
        p2 = y2_count / n
        gini = 1 - (p1 ** 2 + p2 ** 2)
        
        return gini

    def get_GINI(self):
        y1_count, y2_count = Counter(self.xFeat['y']).get(0, 0), Counter(self.xFeat['y']).get(1, 0)
        return self.GINI_impurity(y1_count, y2_count)

    def GINI_best_split_in_variable(self, variable):
        GINI_base = self.get_GINI()

        max_gain = 0
        best_feature = None
        best_value = None
        
        for value in self.xFeat[variable]:
            left_counts = Counter(self.xFeat[self.xFeat[variable]<value]['y'])
            right_counts = Counter(self.xFeat[self.xFeat[variable]>=value]['y'])

            y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

            gini_left = self.GINI_impurity(y0_left, y1_left)
            gini_right = self.GINI_impurity(y0_right, y1_right)

            n_left = y0_left + y1_left
            n_right = y0_right + y1_right

            w_left = n_left / (n_left + n_right)
            w_right = n_right / (n_left + n_right)

            wGINI = w_left * gini_left + w_right * gini_right

            GINIgain = GINI_base - wGINI


            if GINIgain > max_gain:
                best_feature = variable
                best_value = value 

                max_gain = GINIgain

        return [max_gain, best_feature, best_value]

    def entropy(self, variable):
        max_entropy = 1
        best_variable = None
        best_value = None

        for value in self.xFeat[variable]:
            y1 = self.xFeat[self.xFeat[variable] < value]['y']
            if len(y1) == 0:
                a = 0
            else:
                a = len(y1[y1 == 0])/len(y1)
            if a == 0:
                entropy1 = 0
            else:
                entropy1 = -1 * np.sum(a*np.log2(a))

            y2 = self.xFeat[self.xFeat[variable] >= value]['y']
            if len(y2) == 0:
                a = 0
            else:
                a = len(y2[y2 == 0])/len(y2)
            if a == 0:
                entropy2 = 0
            else:
                entropy2 = -1 * np.sum(a*np.log2(a))

            entropy  = 0
            entropy += (len(y1) / (len(y1) + len(y2))) * entropy1
            entropy += (len(y2) / (len(y1) + len(y2))) * entropy2


            if entropy <= max_entropy:
                max_entropy = entropy
                best_variable = variable
                best_value = value

        return [max_entropy, best_variable, best_value]

        

    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        self.xFeat = xFeat
        self.xFeat['y'] = y

        self.counts = Counter(y)

        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat 

        df = xFeat.copy()

        if (self.maxDepth > 0) and (len(y) >= self.minLeafSample):

            x = self.best_split()
            best_feature = x[1]
            best_value = x[0]

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                left_df, right_df = df[df[self.best_feature]<=self.best_value].copy(), df[df[self.best_feature]>self.best_value].copy()

                left = DecisionTree(
                    criterion=self.criterion,
                    maxDepth=self.maxDepth - 1, 
                    minLeafSample=self.minLeafSample, 
                    )

                left.xFeat = left_df.loc[:, ~left_df.columns.isin(['y'])]
                left.y = left_df['y']

                self.left = left 
                self.left.train(xFeat=left.xFeat, y=left.y)

                right = DecisionTree(
                    criterion=self.criterion,
                    maxDepth=self.maxDepth - 1, 
                    minLeafSample=self.minLeafSample, 
                    )

                right.xFeat = right_df.loc[:, ~right_df.columns.isin(['y'])]
                right.y = right_df['y']

                self.right = right 
                self.right.train(xFeat=right.xFeat, y=right.y)
            else:
                self.left = None
                self.right = None
                        
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

        for _, x in xFeat.iterrows():
            values = {}
            for feature in xFeat:
                values.update({feature: x[feature]})
        
            yHat.append(self.predict_obs(values))

        return yHat


    def predict_obs(self, values):
        cur_node = self
        while cur_node.maxDepth > 0 and "best_feature" in cur_node.__dict__:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if len(cur_node.xFeat) < cur_node.minLeafSample:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            if cur_node.yhat == None:
                cur_node.yhat = 0

        return cur_node.yhat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    # parser.add_argument("md",
    #                     type=int,
    #                     help="maximum depth")
    # parser.add_argument("mls",
    #                     type=int,
    #                     help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    mls = [1, 2, 5, 10, 15, 25, 50]

    for mls in mls:
        args = parser.parse_args()
        args.md = 5
        args.mls = mls
        print("MLS Depth: ", mls)
        # load the train and test data
        xTrain = pd.read_csv(args.xTrain)
        yTrain = pd.read_csv(args.yTrain)
        xTest = pd.read_csv(args.xTest)
        yTest = pd.read_csv(args.yTest)
        # create an instance of the decision tree using gini
        dt1 = DecisionTree('gini', args.md, args.mls)
        trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
        print("GINI Criterion ---------------")
        print("Training Acc:", trainAcc1)
        print("Test Acc:", testAcc1)
        dt = DecisionTree('entropy', args.md, args.mls)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        print("Entropy Criterion ---------------")
        print("Training Acc:", trainAcc)
        print("Test Acc:", testAcc)      

if __name__ == "__main__":
    main()
    from PIL import Image

    image = Image.open('Q1C1.png')
    image.show()

    image = Image.open('Q1C2.png')
    image.show()