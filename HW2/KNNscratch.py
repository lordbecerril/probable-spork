#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode

def reduce_data(data,labels):
    """
        Reduce dimensionality of the data to 3D.
    """
    tsne = TSNE(n_components=3)
    tsneX = tsne.fit_transform(data)
    columns = [ 'x', 'y','z']
    df = pd.DataFrame(data=tsneX, columns=columns)
    df.append(labels, ignore_index = True)
    return df


def get_eucladian_distance(data1, data2):
    """
        Returns eucaladian distance of the data
        https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    """
    dist = np.linalg.norm(data1 - data2)
    return(dist)


def get_neighbors(train, row, k):
    distances = list()
    neighbors = list()
    index = 0
    for x1,y1,z1 in zip(train['x'],train['y'],train['z']):
        dist = get_eucladian_distance(row, np.array((x1 ,y1, z1)))
        distances.append((index, dist))
        index = index + 1
    distances.sort(key=lambda tup: tup[1])
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train,train_labels, row, k):
    neighbors = get_neighbors(train, row, k)
    print(neighbors)
    arr = []
    for i in neighbors:
        arr.append(train_labels[i])
    print(arr)
    try:
        return(mode(arr))
    except:
        print("No unique mode")
        return(-1)



def main():
    """
        Implement KNN from scratch using Python. You are given two data sets:
        MNIST_training.csv and MNIST_test.csv, where “MNIST_training.csv” contains training data
        that you will find the K-nearest neighbors, whereas “MNIST_test.csv” consists of test data that you need
        to predict labels. The training data contains 10 classes (i.e., 0, 1, 2, …, 9), each of which has 95 samples,
        while there are 5 samples on each class in the test data set
    """
    print("Hello World from KNNscratch.py Script!")

    # Create Dataframe for MNIST training data-------------
    MNISTtrain_df = pd.read_csv("MNIST_training.csv")
    # make two vectors from the MNIST Train data
    train_labels = MNISTtrain_df.iloc[:, 0] # label data
    images = MNISTtrain_df.drop('label', axis=1) # pixel values for image
    # Reduce the dimensionality to 3 dimensions
    reduced_df_train = reduce_data(images,train_labels)
    # Write to CSV to see output of dimensionality reduction
    reduced_df_train.to_csv("3dMNISTtrain_data.csv", index = False)

    # Create Dataframe for MNIST test data-----------------
    MNISTtest_df = pd.read_csv("MNIST_test.csv")
    # make two vectors from the MNIST test data
    test_labels = MNISTtest_df.iloc[:, 0] # label data
    images = MNISTtest_df.drop('label', axis=1) # pixel values for image
    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data(images,test_labels)
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv("3dMNISTtest_data.csv", index = False)

    # KNN time
    # Iterate through the test data to get Predctions  numpy.array((x1 ,y1, z1))
    index = 0 #the current index
    num_correct = 0 #number of classifications correct
    for x1,y1,z1 in zip(reduced_df_test['x'],reduced_df_test['y'],reduced_df_test['z']):
        prediction = predict_classification(reduced_df_train,train_labels, np.array((x1 ,y1, z1)), 15)
        print('Expected %d, Got %d.' % (test_labels[index], prediction))
        if test_labels[index] == prediction:
            num_correct = num_correct + 1
        index = index + 1
    accur = (num_correct/50) * 100
    print('Accuracy of KNN was ', accur, "%")

    print("KNNscratch.py has finished running, have a good day!")


if __name__== "__main__":
    main()
