#AUTHOR: Eric Becerril-Blas <becere1@unlv.nevada.edu>
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode
from scipy.spatial import distance

#Global Variables
# Create Dataframe for MNIST training data-------------
MNISTtrain_df = pd.read_csv("MNIST_training.csv")

# Create Dataframe for MNIST test data-----------------
MNISTtest_df = pd.read_csv("MNIST_test.csv")

def get_eucladian_distance(data1, data2):
    """
        Returns eucaladian distance of the data
        https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    """
    dist = np.linalg.norm(data1 - data2)
    return(dist)


def get_neighbors(train, row, k, dimensionality):
    distances = list()
    neighbors = list()
    # 3 dimensions
    if dimensionality == 3:
        for i,x1,y1,z1 in zip(train['label'],train['x'],train['y'],train['z']):
            dist = get_eucladian_distance(row, np.array((x1 ,y1, z1)))
            distances.append((i, dist))
    # 2 dimensions
    else:
        for i,x1,y1 in zip(train['label'],train['x'],train['y']):
            dist = get_eucladian_distance(row, np.array((x1 ,y1)))
            distances.append((i, dist))
    distances.sort(key=lambda tup: tup[1])
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train, row, k, file, dimensionality):
    neighbors = get_neighbors(train, row, k, dimensionality)
    try:
        return(mode(neighbors))
    except:
        file.write(" No unique mode ")
        return(-1)

def KNN(train, test, dimensionality, fname):
    """
        KNN written from scratch homie
    """
    # write an output file
    file = open(fname, "w+")
    k = 15
    # For 3 Dimensions
    if dimensionality == 3:
        num_correct = 0
        for i,x1,y1,z1 in zip(test['label'], test['x'], test['y'], test['z']):
            prediction = predict_classification(train, np.array((x1 ,y1, z1)), k, file, dimensionality)
            file.write('Expected '+str(i)+', Got '+str(prediction)+'.\n')
            if i == prediction:
                num_correct = num_correct + 1
        accur = (num_correct/50) * 100
        file.write('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n')
        print('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n')
    # For 2 Dimensions
    else:
        num_correct = 0
        for i,x1,y1 in zip(test['label'], test['x'], test['y']):
            prediction = predict_classification(train, np.array((x1 ,y1)), k, file, dimensionality)
            file.write('Expected '+str(i)+', Got '+str(prediction)+'.\n')
            if i == prediction:
                num_correct = num_correct + 1
        accur = (num_correct/50) * 100
        file.write('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n')
        print('Accuracy of KNN was '+str(accur)+'% with k equal to '+str(k)+'\n')

    file.close()

def reduce_data_to_2d(data, type):
    """
        Reduces the data to 3 dimensions using t-SNE
        Takes 2 variables: data, 2 or 3 dimensions
            # if type == "PCA", do PCA in 2D
            # if type == "t-SNE", do using t-SNE in 2D
    """
    # make two vectors from the MNIST Train data
    labels = data.iloc[:, 0] # label data
    images = data.drop('label', axis=1) # pixel values for image
    if type == "PCA":
        # Reduce the dimensionality to 3 dimensions using PCA
        pca = PCA(n_components=2)
        pca.fit(images)
        PCAX = pca.transform(images)
        columns = [ 'x', 'y']
        df1 = pd.DataFrame(data=PCAX, columns=columns)
        df = pd.concat([labels, df1], axis=1)
    if type == "t-SNE":
        # Reduce the dimensionality to 3 dimensions using t-SNE
        tsne = TSNE(n_components=2)
        tsneX = tsne.fit_transform(images)
        columns = [ 'x', 'y']
        df1 = pd.DataFrame(data=tsneX, columns=columns)
        df = pd.concat([labels, df1], axis=1)
    return df


def reduce_data_to_3d(data, type):
    """
        Reduces the data to 3 dimensions using t-SNE
        Takes 2 variables: data, 2 or 3 dimensions
            # if type == "PCA", do PCA in 3D
            # if type == "t-SNE", do using t-SNE in 3D
    """
    # make two vectors from the MNIST Train data
    labels = data.iloc[:, 0] # label data
    images = data.drop('label', axis=1) # pixel values for image
    if type == "PCA":
        # Reduce the dimensionality to 3 dimensions using PCA
        pca = PCA(n_components=3)
        pca.fit(images)
        PCAX = pca.transform(images)
        columns = [ 'x', 'y','z']
        df1 = pd.DataFrame(data=PCAX, columns=columns)
        df = pd.concat([labels, df1], axis=1)
    if type == "t-SNE":
        # Reduce the dimensionality to 3 dimensions using t-SNE
        tsne = TSNE(n_components=3)
        tsneX = tsne.fit_transform(images)
        columns = [ 'x', 'y','z']
        df1 = pd.DataFrame(data=tsneX, columns=columns)
        df = pd.concat([labels, df1], axis=1)
    return df

def reduction_2D(reduc_method, to_csv_name_train, to_csv_name_test, master_txt_file):
    reduced_df_train = reduce_data_to_2d(MNISTtrain_df,reduc_method)
    reduced_df_train.to_csv(to_csv_name_train, index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_2d(MNISTtest_df,reduc_method)
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv(to_csv_name_test, index = False)

    KNN(reduced_df_train, reduced_df_test, 2, master_txt_file)

def reduction_3D(reduc_method, to_csv_name_train, to_csv_name_test, master_txt_file):
    # reduce the dimensionality to 3 dimensions
    reduced_df_train = reduce_data_to_3d(MNISTtrain_df,reduc_method)
    # Write to CSV to see output of dimensionality reduction
    reduced_df_train.to_csv(to_csv_name_train, index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_3d(MNISTtest_df,reduc_method)
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv(to_csv_name_test, index = False)

    KNN(reduced_df_train, reduced_df_test, 3, master_txt_file)


def main():
    """
        Implement KNN from scratch using Python. You are given two data sets:
        MNIST_training.csv and MNIST_test.csv, where “MNIST_training.csv” contains training data
        that you will find the K-nearest neighbors, whereas “MNIST_test.csv” consists of test data that you need
        to predict labels. The training data contains 10 classes (i.e., 0, 1, 2, …, 9), each of which has 95 samples,
        while there are 5 samples on each class in the test data set
    """
    print("Hello World from KNNscratch.py Script!\n")


    # ---------------
    print("Commencing KNN using MNIST reduced to 2D by PCA")
    reduction_2D("PCA", "2dPCA_MNISTtrain_data.csv", "2dPCA_MNISTtest_data.csv", "AccuracyWithPCA2D.txt")
    """
    reduced_df_train = reduce_data_to_2d(MNISTtrain_df,"PCA")
    reduced_df_train.to_csv("2dPCA_MNISTtrain_data.csv", index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_2d(MNISTtest_df,"PCA")
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv("2dPCA_MNISTtest_data.csv", index = False)

    KNN(reduced_df_train, reduced_df_test, 2, "AccuracyWithPCA2D.txt")
    print("2D PCA finished\n")
    """
    # ---------------
    print("Commencing KNN using MNIST reduced to 3D by PCA")
    # reduce the dimensionality to 3 dimensions
    reduced_df_train = reduce_data_to_3d(MNISTtrain_df,"PCA")
    # Write to CSV to see output of dimensionality reduction
    reduced_df_train.to_csv("3dPCA_MNISTtrain_data.csv", index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_3d(MNISTtest_df,"PCA")
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv("3dPCA_MNISTtest_data.csv", index = False)

    KNN(reduced_df_train, reduced_df_test, 3, "AccuracyWithPCA3D.txt")
    print("3D PCA finished\n")
    # ---------------
    print("Commencing KNN using MNIST reduced to 2D by t-SNE")
    reduced_df_train = reduce_data_to_2d(MNISTtrain_df,"t-SNE")
    reduced_df_train.to_csv("2dtsne_MNISTtrain_data.csv", index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_2d(MNISTtest_df,"t-SNE")
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv("2dtsne_MNISTtest_data.csv", index = False)

    KNN(reduced_df_train, reduced_df_test, 2, "AccuracyWithtSNE2D.txt")
    print("2D t-SNE finished\n")

    # ---------------
    print("Commencing KNN using MNIST reduced to 3D by t-SNE")
    reduced_df_train = reduce_data_to_3d(MNISTtrain_df,"t-SNE")
    reduced_df_train.to_csv("3dtsne_MNISTtrain_data.csv", index = False)

    # reduce the dimensionality to 3 dimensions
    reduced_df_test = reduce_data_to_3d(MNISTtest_df,"t-SNE")
    # Write to CSV to see output of dimensionality reduction
    reduced_df_test.to_csv("3dtsne_MNISTtest_data.csv", index = False)

    KNN(reduced_df_train, reduced_df_test, 2, "AccuracyWithtSNE3D.txt")
    print("3D t-SNE finished\n")

    # ---------------
    print("KNNscratch.py is finished, have a great day! :)")


if __name__== "__main__":
    main()
