import numpy as np
import argparse
import pandas as pd
from pandas.core import algorithms
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def search(xFeat, y):
    trainAuc = []
    testAuc = []

    n_neighbors1 = [1, 2, 5, 10, 15, 25]
    weights1 = ["uniform","distance"]
    algorithm1 = ["auto","ball_tree","kd_tree","brute"]
    leaf_size1 = [1, 2, 5, 10, 15, 25]
    metric1 = ["minkowski"]
    n_jobs1 = [1]

    kf = KFold(n_splits=5)

    best = [0,0]

    for n_neighbors in n_neighbors1:
        for weights in weights1:
            for algorithm in algorithm1:
                for leaf_size in leaf_size1:
                    for metric in metric1:
                        for n_jobs in n_jobs1:

                            model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                        weights=weights,
                                                        algorithm=algorithm,
                                                        leaf_size=leaf_size,
                                                        metric=metric,
                                                        n_jobs=n_jobs)

                            testAuc = []

                            for train_index, test_index in kf.split(xFeat, y):
                                X_train, X_test = xFeat.iloc[train_index], xFeat.iloc[test_index]
                                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                                model = model.fit(X_train, y_train)

                                pred_test = model.predict(X_test)

                                fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, pred_test)
                                metrics.roc_curve(y_test, pred_test)
                                testAuc.append(metrics.auc(fpr_test, tpr_test))

                            if sum(testAuc)/len(testAuc) > best[0]:
                                best = [sum(testAuc)/len(testAuc), n_neighbors, weights, algorithm, leaf_size, metric, n_jobs]
    # print(best)

    best = [0,0]

    criterion1 = ["gini", "entropy"]
    max_depth1 = [1, 2, 3, 5, 10, 15, 25, 50, 100]
    min_samples_leaf1 = [1, 2, 3, 5, 10, 15, 25, 50, 100]

    for criterion in criterion1:
        for max_depth in max_depth1:
            for min_samples_leaf in min_samples_leaf1:

                testAuc = []

                model = DecisionTreeClassifier(criterion=criterion, 
                                            max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf)

                for train_index, test_index in kf.split(xFeat, y):
                    X_train, X_test = xFeat.iloc[train_index], xFeat.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = model.fit(X_train, y_train)

                    pred_test = model.predict(X_test)

                    fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, pred_test)
                    metrics.roc_curve(y_test, pred_test)
                    testAuc.append(metrics.auc(fpr_test, tpr_test))

                if sum(testAuc)/len(testAuc) > best[0]:
                    best = [sum(testAuc)/len(testAuc), criterion, max_depth, min_samples_leaf]
                                        
    # print(best)


def test(xTrain, yTrain, xTest, yTest):
    X_train95, X_test5, y_train95, y_test5 = train_test_split(xTrain, yTrain, test_size=0.05)
    X_train90, X_test10, y_train90, y_test10 = train_test_split(xTrain, yTrain, test_size=0.1)
    X_train80, X_test20, y_train80, y_test20 = train_test_split(xTrain, yTrain, test_size=0.2)

    Xdata = [xTrain, X_train95, X_train90, X_train80]
    ydata = [yTrain, y_train95, y_train90, y_train80]

    auc = []
    acc = []

    model = KNeighborsClassifier(n_neighbors=1,
                                weights="uniform",
                                algorithm="auto",
                                leaf_size=1,
                                metric="minkowski",
                                n_jobs=1)

    for i in range(len(Xdata)):
        model = model.fit(Xdata[i], ydata[i])

        pred_test = model.predict(xTest)

        fpr_test, tpr_test, thresholds = metrics.roc_curve(yTest, pred_test)
        metrics.roc_curve(yTest, pred_test)
        auc.append(metrics.auc(fpr_test, tpr_test))
        acc.append(metrics.accuracy_score(yTest, pred_test))



    model = DecisionTreeClassifier(criterion="entropy",
                                max_depth=15,
                                min_samples_leaf=1)

    for i in range(len(Xdata)):
        model = model.fit(Xdata[i], ydata[i])

        pred_test = model.predict(xTest)

        fpr_test, tpr_test, thresholds = metrics.roc_curve(yTest, pred_test)
        metrics.roc_curve(yTest, pred_test)
        auc.append(metrics.auc(fpr_test, tpr_test))
        acc.append(metrics.accuracy_score(yTest, pred_test))


    # for x in auc:
    #     print(x)
    # print()
    # for x in acc:
    #     print(x)

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    search(xTrain, yTrain)
    test(xTrain, yTrain, xTest, yTest)

if __name__ == "__main__":
    main()