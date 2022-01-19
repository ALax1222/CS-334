import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """

    y = []
    x = []

    data = pd.read_csv(filename)
    for i in range(len(data)):
        y.append(data.iloc[i, 0][0])
        x.append(data.iloc[i, 0][2:])
    
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    y_train.to_csv('y_train.csv')
    y_test.to_csv('y_test.csv')

    return X_train, X_test, y_train, y_test


def build_vocab_map(X_train):
    temp_words = dict()
    for i in range(len(X_train)):
        contains = set()
        review = X_train.iloc[i, 0].split()
        for word in review:
            if word not in contains:
                contains.add(word)
                if word not in temp_words:
                    temp_words[word] = 1
                else:
                    temp_words[word] += 1

    temp_words = dict(sorted(temp_words.items(), key=lambda item: item[1], reverse=True))
    
    words = dict()
    for word in temp_words:
        if temp_words[word] >= 30:
            words[word] = temp_words[word]
        else:
            break
    
    print(len(words))
    print(words)

    return words


def construct_binary(words, X_train, X_test):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    
    X_train_bin = []
    for i in range(len(X_train)):
        if i % 100 == 0:
            print(i)
        bin_arr = []
        review = X_train.iloc[i, 0].split()
        for word in words:
            if word in review:
                bin_arr.append(1)
            else:
                bin_arr.append(0)
        X_train_bin.append(bin_arr)

    X_test_bin = []
    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        bin_arr = []
        review = X_test.iloc[i, 0].split()
        for word in words:
            if word in review:
                bin_arr.append(1)
            else:
                bin_arr.append(0)
        X_test_bin.append(bin_arr)

    X_train_bin = pd.DataFrame(X_train_bin)
    X_test_bin = pd.DataFrame(X_test_bin)

    X_train_bin.to_csv('X_train_bin.csv')
    X_test_bin.to_csv('X_test_bin.csv')


    return X_train_bin, X_test_bin


def construct_count(words, X_train, X_test):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    X_train_count = []
    for i in range(len(X_train)):
        if i % 100 == 0:
            print(i)
        count_arr = []
        review = X_train.iloc[i, 0].split()
        for word in words:
            count_arr.append(review.count(word))
        X_train_count.append(count_arr)

    X_test_count = []
    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        count_arr = []
        review = X_test.iloc[i, 0].split()
        for word in words:
            count_arr.append(review.count(word))
        X_test_count.append(count_arr)

    X_train_count = pd.DataFrame(X_train_count)
    X_test_count = pd.DataFrame(X_test_count)

    X_train_count.to_csv('X_train_count.csv')
    X_test_count.to_csv('X_test_count.csv')

    return None


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = model_assessment(args.data)
    words = build_vocab_map(X_train)
    construct_binary(words, X_train, X_test)
    construct_count(words, X_train, X_test)



if __name__ == "__main__":
    main()
