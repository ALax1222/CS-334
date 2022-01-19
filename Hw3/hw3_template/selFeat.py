import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler



def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    df = df.drop(columns=['date', 'Windspeed', 'Visibility', 'Tdewpoint'])
    # plt.figure(figsize=(16, 6))

    # heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)

    # plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    return df


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """

    df_new = df[['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out']]

    return df_new


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    scaler = StandardScaler()

    scaler.fit(trainDF)

    trainDF = scaler.transform(trainDF)
    testDF = scaler.transform(testDF)

    trainDF = pd.DataFrame(trainDF)
    testDF = pd.DataFrame(testDF)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    # parser.add_argument("outTrain",
    #                     help="filename of the updated training data")
    # parser.add_argument("outTest",
    #                     help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv('xTrainTr', index=False)
    xTestTr.to_csv('xTestTr', index=False)


if __name__ == "__main__":
    main()
