#AUTHOR: Eric Becerril-Blas
# ML Housing dataset (UCI Machine Learning database)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)

def plot_data(data, title, pathname):
    '''
        Plot the data with given data from task 1 and task 2 and then use plt.clf() to clear current figure
    '''
    i = 0
    k = 0
    while i < 1000:
        j = i + 100
        plt.scatter(data[i:j, 0], data[i:j, 1], marker='$'+str(k)+'$')
        k = k + 1
        i = i + 100
    plt.title(title)
    plt.savefig(pathname)
    plt.clf()


def task1(data, labels, images):
    '''
        Visualize the MNIST data using PCA. Reduce the data dimension to two or three and plot the
        data of reduced dimension. Must plot all the data of ten groups (0 to 9)
    '''
    pca = PCA(n_components=2)
    pca.fit(images)
    PCAX = pca.transform(images)
    plot_data(PCAX, "2D PCA of MNIST",'MNISTpca2D.png')


def task2(data, labels, images):
    '''
        Visualize the MNIST data using t-SNE library.
    '''
    tsne = TSNE(n_components=2)
    tsneX = tsne.fit_transform(images)
    plot_data(tsneX, "2D t-SNE of MNIST",'MNISTt_SNE2D.png')


def task3(data, X):
    '''
        Visualize the housing data using violin plot.
    '''
    plt.plot(X, 'ko')
    plt.title("Violin Plot of MNIST")
    plt.savefig('ViolinPlot.png')


def main():
    '''
        The main body of the program which runs all 3 tasks
    '''
    print("Hello World from HW1.py Script!")

    # Create Dataframe for MNIST data
    MNIST_df = pd.read_csv("MNIST_100.csv")

    # make two variables - X and y for the MNIST dataset
    y = MNIST_df.iloc[:, 0] # label data
    X = MNIST_df.drop('label', axis=1) # pixel values for image

    # Create Dataframe for housing dataset
    HOUSING_df = pd.read_csv("housing_training.csv")
    Z = HOUSING_df.iloc[:, 0:2]

    # Call functions for each task 
    task1(MNIST_df,y,X)
    task2(MNIST_df,y,X)
    task3(HOUSING_df,Z)

if __name__== "__main__":
    main()
