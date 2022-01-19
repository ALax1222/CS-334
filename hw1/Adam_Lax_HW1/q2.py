import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
# Load Iris dataset
def download():
    iris = datasets.load_iris()
    return iris


def graph1(iris):
    # Preparing Iris dataset
    iris_data = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_target = pd.DataFrame(data=iris.target, columns=['species'])
    iris_df = pd.concat([iris_data, iris_target], axis=1)

    # Add species name
    iris_df['species_name'] = np.where(iris_df['species'] == 0, 'Setosa', None)
    iris_df['species_name'] = np.where(iris_df['species'] == 1, 'Versicolor', iris_df['species_name'])
    iris_df['species_name'] = np.where(iris_df['species'] == 2, 'Virginica', iris_df['species_name'])

    # Prepare petal length by species datasets
    setosa_petal_length = iris_df[iris_df['species_name'] == 'Setosa']['petal_length']
    versicolor_petal_length = iris_df[iris_df['species_name'] == 'Versicolor']['petal_length']
    virginica_petal_length = iris_df[iris_df['species_name'] == 'Virginica']['petal_length']

    # Prepare petal width by species datasets
    setosa_petal_width= iris_df[iris_df['species_name'] == 'Setosa']['petal_width']
    versicolor_petal_width = iris_df[iris_df['species_name'] == 'Versicolor']['petal_width']
    virginica_petal_width = iris_df[iris_df['species_name'] == 'Virginica']['petal_width']

    # Prepare sepal length by species datasets
    setosa_sepal_length = iris_df[iris_df['species_name'] == 'Setosa']['sepal_length']
    versicolor_sepal_length = iris_df[iris_df['species_name'] == 'Versicolor']['sepal_length']
    virginica_sepal_length = iris_df[iris_df['species_name'] == 'Virginica']['sepal_length']

    # Prepare sepal width by species datasets
    setosa_sepal_width= iris_df[iris_df['species_name'] == 'Setosa']['sepal_width']
    versicolor_sepal_width = iris_df[iris_df['species_name'] == 'Versicolor']['sepal_width']
    virginica_sepal_width = iris_df[iris_df['species_name'] == 'Virginica']['sepal_width']


    # Graphing petal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Distribution of petal length by species')

    # Set species names as labels for the boxplot
    dataset = [setosa_petal_length, versicolor_petal_length, virginica_petal_length]
    labels = iris_df['species_name'].unique()
    ax.boxplot(dataset, labels=labels)
    plt.show()


    # Graphing petal width distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Distribution of petal width by species')

    # Set species names as labels for the boxplot
    dataset = [setosa_petal_width, versicolor_petal_width, virginica_petal_width]
    labels = iris_df['species_name'].unique()
    ax.boxplot(dataset, labels=labels)
    plt.show()


    # Graphing sepal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Distribution of sepal length by species')

    # Set species names as labels for the boxplot
    dataset = [setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length]
    labels = iris_df['species_name'].unique()
    ax.boxplot(dataset, labels=labels)
    plt.show()


    # Graphing sepal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Distribution of sepal width by species')

    # Set species names as labels for the boxplot
    dataset = [setosa_sepal_width, versicolor_sepal_width, virginica_sepal_width]
    labels = iris_df['species_name'].unique()
    ax.boxplot(dataset, labels=labels)
    plt.show()



def graph2(iris):
    iris_data = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_target = pd.DataFrame(data=iris.target, columns=['species'])
    iris_df = pd.concat([iris_data, iris_target], axis=1)

    # Add species name
    iris_df['species_name'] = np.where(iris_df['species'] == 0, 'Setosa', None)
    iris_df['species_name'] = np.where(iris_df['species'] == 1, 'Versicolor', iris_df['species_name'])
    iris_df['species_name'] = np.where(iris_df['species'] == 2, 'Virginica', iris_df['species_name'])

    # Prepare petal length by species datasets
    setosa_petal_length = iris_df[iris_df['species_name'] == 'Setosa']['petal_length']
    versicolor_petal_length = iris_df[iris_df['species_name'] == 'Versicolor']['petal_length']
    virginica_petal_length = iris_df[iris_df['species_name'] == 'Virginica']['petal_length']

    # Prepare petal width by species datasets
    setosa_petal_width= iris_df[iris_df['species_name'] == 'Setosa']['petal_width']
    versicolor_petal_width = iris_df[iris_df['species_name'] == 'Versicolor']['petal_width']
    virginica_petal_width = iris_df[iris_df['species_name'] == 'Virginica']['petal_width']

    # Prepare sepal length by species datasets
    setosa_sepal_length = iris_df[iris_df['species_name'] == 'Setosa']['sepal_length']
    versicolor_sepal_length = iris_df[iris_df['species_name'] == 'Versicolor']['sepal_length']
    virginica_sepal_length = iris_df[iris_df['species_name'] == 'Virginica']['sepal_length']

    # Prepare sepal width by species datasets
    setosa_sepal_width= iris_df[iris_df['species_name'] == 'Setosa']['sepal_width']
    versicolor_sepal_width = iris_df[iris_df['species_name'] == 'Versicolor']['sepal_width']
    virginica_sepal_width = iris_df[iris_df['species_name'] == 'Virginica']['sepal_width']


    # Graphing petal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Petal length and Petal Width Colored by Species')

    # Set species names as labels for the boxplot
    x = setosa_petal_length
    y = setosa_petal_width
    
    ax.scatter(x, y, c='r')

    x = versicolor_petal_length
    y = versicolor_petal_width

    ax.scatter(x, y, c='g')

    x = virginica_petal_length
    y = virginica_petal_width

    ax.scatter(x, y, c='b')

    plt.show()


    # Graphing petal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set plot title
    ax.set_title('Sepal length and Sepal Width Colored by Species')

    # Set species names as labels for the boxplot
    x = setosa_sepal_length
    y = setosa_sepal_width
    
    ax.scatter(x, y, c='r')

    x = versicolor_sepal_length
    y = versicolor_sepal_width

    ax.scatter(x, y, c='g')

    x = virginica_sepal_length
    y = virginica_sepal_width

    ax.scatter(x, y, c='b')

    plt.show()







def main():
    iris = download()
    graph1(iris)
    graph2(iris)

if __name__ == "__main__":
    main()