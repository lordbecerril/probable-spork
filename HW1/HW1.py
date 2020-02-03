#AUTHOR: Eric Becerril-Blas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def task1(data):
    '''
        Visualize the MNIST data using PCA. Reduce the data dimension to two or three and plot the
        data of reduced dimension. Must plot all the data of ten groups (0 to 9)
    '''
    # make two variables - images and labels
    labels = data.iloc[:, 0]
    images = data.drop('label', axis=1)

    pca = PCA(n_components=2)
    pca.fit(images)
    PCAX = pca.transform(images)
    i = 0
    k = 0
    while i < 1000:
        j = i + 100
        plt.scatter(PCAX[i:j, 0], PCAX[i:j, 1], marker='$'+str(k)+'$')
        k = k + 1
        i = i + 100
    plt.title("2D PCA of MNIST")
    plt.savefig('MNISTpca2D.png')

def task2(data):
    '''
        Visualize the MNIST data using t-SNE library.
    '''
    # make two variables - images and labels
    labels = data.iloc[:, 0]
    images = data.drop('label', axis=1)

    tsneX = TSNE(n_components=2).fit_transform(images)
    tsneX.shape
    i = 0
    k = 0
    while i < 1000:
        j = i + 100
        plt.plot(tsneX[i:j, 0], tsneX[i:j, 1], marker='$'+str(k)+'$')
        k = k + 1
        i = i + 100
    plt.title("2D t-SNE of MNIST")
    plt.savefig('MNISTt_SNE2D.png')

def task3(data):
    '''
        Visualize the housing data using violin plot.
    '''


def main():
    '''
        The main body of my program which consists of all 3 tasks
    '''
    print("Hello from HW1.py Script!")

    MNIST_df = pd.read_csv("MNIST_100.csv") # Create Dataframe for MNIST data
    HOUSING_df = pd.read_csv("housing_training.csv")

    task1(MNIST_df)
    task2(MNIST_df)
    task3(HOUSING_df)

if __name__== "__main__":
    main()
