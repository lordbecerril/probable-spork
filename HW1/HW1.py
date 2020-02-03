#AUTHOR: Eric Becerril-Blas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D

def task1(data):
    '''
    Visualize the MNIST data using PCA. Reduce the data dimension to two or three and plot the
    data of reduced dimension. Must plot all the data of ten groups (0 to 9)
    '''

def task2(data):
    '''
    Visualize the MNIST data using t-SNE library.
    '''

def task3(data):
    '''
    Visualize the housing data using violin plot.
    '''


def main():
    '''
        The Main MEAT of my program buddy. Heck yeah!
    '''
    print("Hello from HW1.py Script!")

    MNIST_df = pd.read_csv("MNIST_100.csv") # Create Dataframe for MNIST data
    HOUSING_df = pd.read("housing_training.csv")

    task1(MNIST_df)
    task2(MNIST_df)
    task3(HOUSINGdf)

if __name__== "__main__":
    main()
