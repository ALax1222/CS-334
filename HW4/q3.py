import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename).iloc[:, 1:]
    return df.to_numpy()

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """

    wrong = 0
    for i in range(len(yHat)):
        if yHat[i] != yTrue[i]:
            wrong += 1

    return wrong

def main():
    """
    Main file to run from the command line.
    """

    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        # default="X_train_bin.csv",
                        default="X_train_count.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="y_train.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        # default="X_test_bin.csv",
                        default="X_test_count.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="y_test.csv",
                        help="filename for labels associated with the test data")
    # parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)  
 
    model1 = MultinomialNB()
    model2 = BernoulliNB()
    model3 = LogisticRegression()

    models = [model1, model2, model3]

    for model in models:
        model.fit(xTrain, yTrain)

        yHat = model.predict(xTest)
        # print out the number of mistakes
        print("Number of mistakes on the test dataset")
        print(calc_mistakes(yHat, yTest))

if __name__ == "__main__":
    main()
