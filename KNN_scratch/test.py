#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
import pandas as pd
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode
from statistics import mean
import operator
from scipy.spatial import distance
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

#Global Variables
# Create Dataframe for MNIST training data-------------
MNISTtrain_df = pd.read_csv("MNIST_training.csv")

# Create Dataframe for MNIST test data-----------------
MNISTtest_df = pd.read_csv("MNIST_test.csv")

def get_eucladian_distance(data1, data2):
    """
        Returns eucaladian distance of the data
            data1: np.array of row 1 of the test data
            data2: np.array of a row in the train data
    """
    dist = np.linalg.norm(data1 - data2) # using numpy linalg function
    return(dist)



def compute_mode(numbers):
    """
        statistic mode function throws an exception so built this to handle that
    """
    counts = {}
    maxcount = 0
    for number in numbers:
        if number not in counts:
            counts[number] = 0
        counts[number] += 1
        if counts[number] > maxcount:
            maxcount = counts[number]

    for number, count in counts.items():
        if count == maxcount:
            return(number)


def subtractor(test_row,data,labels):
    distances = []
    for index, row in data.iterrows():
        dist = get_eucladian_distance(test_row, row.values)
        distances.append(dist)
    columns = ['distance'] # declare column names
    df1 = pd.DataFrame(data=distances, columns=columns) # Make reduced data into a dataframe
    df = pd.concat([labels, df1], axis=1) # Make a new dataframe with the labels now added to it
    df = df.sort_values(by=['distance'], ascending=True)
    df = df.reset_index()
    return(df)



def main():
    K = 5
    train_labels = MNISTtrain_df.iloc[:, 0] # label data
    train_images = MNISTtrain_df.drop('label', axis=1) # pixel values for image

    test_labels = MNISTtest_df.iloc[:, 0] # label data
    test_images = MNISTtest_df.drop('label', axis=1) # pixel values for image
    neighbors = []
    num_correct = 0 # variable used to count the number of correct predictions

    for index, row in test_images.iterrows():
        #print(row.values) #get first row
        print(index)
        df = subtractor(row.values, train_images,train_labels) #subtract to all vals in train data
        print(df)
        for i in range(K): # for range in the k value
            neighbors.append(df['label'][i]) # append the label to the neighbors list
        print(neighbors)
        prediction = compute_mode(neighbors)
        neighbors = []
        print('Expected '+str(test_labels[index])+', Got '+str(prediction)+" for K = "+str(K))
        if test_labels[index] == prediction:
            num_correct = num_correct + 1
            print("Number correct is ", num_correct)
    accur = (num_correct/50) * 100 # Calculate the accuracy
    print('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(K)) # Print to command line





if __name__== "__main__":
    main()
